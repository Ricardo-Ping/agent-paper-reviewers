from __future__ import annotations

from collections import Counter
from typing import Any

from ..executors.base import ExecutorAdapter
from ..models import TaskSpec
from ..services.venue_recommender import recommend_venues
from .base import PipelineContext, PipelineStep


class VenueRecommenderStep(PipelineStep):
    name = "VenueRecommender"

    def __init__(self, executor: ExecutorAdapter | None = None) -> None:
        self.executor = executor

    def run(self, ctx: PipelineContext) -> None:
        paper_structured = ctx.artifacts.get("paper_structured", {})
        claims_normalized = ctx.artifacts.get("claims_normalized", {})
        evidence_index = ctx.artifacts.get("evidence_index", {})
        target_year = int(ctx.input_data.venue.year or 0)

        payload = recommend_venues(
            ctx.repo_root or ctx.run_dir,
            target_year=target_year,
            paper_structured=paper_structured if isinstance(paper_structured, dict) else {},
            claims_normalized=claims_normalized if isinstance(claims_normalized, dict) else {},
            evidence_index=evidence_index if isinstance(evidence_index, dict) else {},
            top_k=5,
        )

        recommended = self._recommend_with_executor(
            ctx=ctx,
            payload=payload,
            paper_structured=paper_structured if isinstance(paper_structured, dict) else {},
            claims_normalized=claims_normalized if isinstance(claims_normalized, dict) else {},
        )
        if isinstance(recommended, dict):
            payload = recommended
        else:
            # Fallback mode: keep rule ranking, but let agent refine reasons.
            refined = self._refine_with_executor(
                ctx=ctx,
                payload=payload,
                paper_structured=paper_structured if isinstance(paper_structured, dict) else {},
                claims_normalized=claims_normalized if isinstance(claims_normalized, dict) else {},
            )
            if isinstance(refined, dict):
                payload = refined

        ctx.artifacts["venue_recommendations"] = payload
        ctx.dump_json("artifacts/venue_recommendations.json", payload)

    def _recommend_with_executor(
        self,
        *,
        ctx: PipelineContext,
        payload: dict[str, Any],
        paper_structured: dict[str, Any],
        claims_normalized: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.executor is None:
            return None

        rows = payload.get("recommended_venues", [])
        if not isinstance(rows, list) or not rows:
            return None

        venue_priors = self._build_agent_candidate_priors(rows)

        sections = paper_structured.get("sections", [])
        section_briefs: list[dict] = []
        if isinstance(sections, list):
            for sec in sections[:8]:
                if not isinstance(sec, dict):
                    continue
                section_briefs.append(
                    {
                        "name": str(sec.get("name", "")),
                        "text": str(sec.get("text", ""))[:520],
                    }
                )

        claim_briefs: list[dict] = []
        claims = claims_normalized.get("claims", [])
        if isinstance(claims, list):
            for row in claims[:10]:
                if not isinstance(row, dict):
                    continue
                claim_briefs.append(
                    {
                        "claim_id": str(row.get("claim_id", "")),
                        "claim_type": str(row.get("claim_type", "")),
                        "claim_text": str(row.get("claim_text", ""))[:260],
                    }
                )

        spec_task = TaskSpec(
            task_type="venue_recommend",
            prompt=(
                "Select and rank the best-fit venues for this paper draft.\n"
                "Important rules:\n"
                "1) Compute scores from paper-venue semantic fit, not by copying rule prior scores.\n"
                "2) Return both semantic_fit_score and review_risk_score with explicit meaning.\n"
                "3) match_score must be your own judged value in [0,1], where higher means better submission fit now.\n"
                "4) Avoid identical scores across venues unless they are truly indistinguishable; explain tie-break reason."
            ),
            context={
                "paper_title": str(paper_structured.get("title", "")),
                "paper_summary": str(paper_structured.get("summary", ""))[:900],
                "section_briefs": section_briefs,
                "claims": claim_briefs,
                "candidate_venues": venue_priors,
                "target_year": int(ctx.input_data.venue.year or 0),
            },
            output_schema={
                "recommended_venues": [
                    {
                        "venue": "iclr",
                        "year": 2026,
                        "semantic_fit_score": 0.78,
                        "review_risk_score": 0.31,
                        "match_score": 0.73,
                        "reasons": ["string"],
                        "fit_summary": "string",
                        "specific_gap_summary": "string",
                        "tie_break_basis": "string",
                        "required_check_passed_count": 3,
                        "required_check_total": 5,
                        "passed_checks": ["baseline_coverage"],
                        "failed_checks": ["statistical_significance"],
                    }
                ],
                "method_note": "string",
            },
            model_profile="judge",
        )

        result = self.executor.execute(spec_task)
        fallback_detected = False
        for warning in result.warnings:
            ctx.add_qa_issue(f"venue_recommender_agent_warning:{warning}")
            lower = str(warning).lower()
            if "fallback" in lower or "api_key_missing" in lower:
                fallback_detected = True
        if fallback_detected:
            ctx.add_qa_issue("venue_recommender_agent_fallback_detected_use_rule_reco")
            return None
        if not result.ok:
            return None

        data: Any = result.output
        if isinstance(data, dict) and isinstance(data.get("response"), dict):
            data = data.get("response")
        if not isinstance(data, dict):
            return None

        rows_raw = data.get("recommended_venues", [])
        if not isinstance(rows_raw, list) or not rows_raw:
            return None

        baseline_by_key = {
            (
                str(r.get("venue", "")).strip().lower(),
                int(r.get("year", 0) or 0),
            ): r
            for r in rows
            if isinstance(r, dict)
        }
        rows_out: list[dict[str, Any]] = []
        for row in rows_raw[:5]:
            if not isinstance(row, dict):
                continue
            venue = str(row.get("venue", "")).strip().lower()
            if not venue:
                continue
            try:
                year = int(row.get("year", 0) or 0)
            except (TypeError, ValueError):
                year = 0
            try:
                score = float(row.get("match_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            semantic_fit = self._safe_score(row.get("semantic_fit_score"), fallback=score)
            review_risk = self._safe_score(row.get("review_risk_score"), fallback=max(0.0, 1.0 - score))
            prior = baseline_by_key.get((venue, year), {})
            prior_score = self._safe_score(prior.get("match_score", 0.0), fallback=0.0)
            blended = self._blend_agent_match_score(
                semantic_fit_score=semantic_fit,
                review_risk_score=review_risk,
                model_match_score=self._safe_score(score, fallback=0.0),
                rule_prior_score=prior_score,
            )
            reasons_raw = row.get("reasons", [])
            reasons = []
            if isinstance(reasons_raw, list):
                for item in reasons_raw:
                    text = str(item).strip()
                    if text:
                        reasons.append(text)
            rows_out.append(
                {
                    "venue": venue,
                    "year": year,
                    "semantic_fit_score": round(semantic_fit, 3),
                    "review_risk_score": round(review_risk, 3),
                    "rule_prior_score": round(prior_score, 3),
                    "match_score": round(blended, 3),
                    "reasons": reasons[:10],
                    "fit_summary": str(row.get("fit_summary", "")).strip(),
                    "specific_gap_summary": str(row.get("specific_gap_summary", "")).strip(),
                    "tie_break_basis": str(row.get("tie_break_basis", "")).strip(),
                    "required_check_passed_count": int(row.get("required_check_passed_count", 0) or 0),
                    "required_check_total": int(row.get("required_check_total", 0) or 0),
                    "passed_checks": row.get("passed_checks", []) if isinstance(row.get("passed_checks", []), list) else [],
                    "failed_checks": row.get("failed_checks", []) if isinstance(row.get("failed_checks", []), list) else [],
                    "agent_refined": True,
                }
            )

        if not rows_out:
            return None

        rows_out = self._enforce_score_dispersion(rows_out)
        unique_score_count = len({round(float(x.get("match_score", 0.0) or 0.0), 3) for x in rows_out})
        if unique_score_count <= 1 and len(rows_out) > 1:
            ctx.add_qa_issue("venue_recommender_warning:agent_scores_not_discriminative")
        rows_out.sort(key=lambda x: float(x.get("match_score", 0.0)), reverse=True)
        return {
            "method": "agent_primary+rule_candidates",
            "fallback_used": bool(payload.get("fallback_used", False)),
            "target_year": int(payload.get("target_year", ctx.input_data.venue.year)),
            "candidate_venues_considered": int(payload.get("candidate_venues_considered", len(rows))),
            "recommended_venues": rows_out[:5],
            "agent_note": str(data.get("method_note", "")).strip(),
            "rule_baseline_top": rows[:2],
        }

    def _refine_with_executor(
        self,
        *,
        ctx: PipelineContext,
        payload: dict[str, Any],
        paper_structured: dict[str, Any],
        claims_normalized: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.executor is None:
            return None
        rows = payload.get("recommended_venues", [])
        if not isinstance(rows, list) or not rows:
            return None

        sections = paper_structured.get("sections", [])
        section_briefs: list[dict] = []
        if isinstance(sections, list):
            for sec in sections[:6]:
                if not isinstance(sec, dict):
                    continue
                section_briefs.append(
                    {
                        "name": str(sec.get("name", "")),
                        "text": str(sec.get("text", ""))[:420],
                    }
                )

        claim_briefs: list[str] = []
        claims = claims_normalized.get("claims", [])
        if isinstance(claims, list):
            for row in claims[:8]:
                if not isinstance(row, dict):
                    continue
                text = str(row.get("claim_text", "")).strip()
                if text:
                    claim_briefs.append(text[:220])

        spec_task = TaskSpec(
            task_type="venue_recommend_refine",
            prompt=(
                "Refine venue recommendation reasons to be specific, paper-aware, and actionable. "
                "Keep venue ordering stable unless a score adjustment is clearly justified."
            ),
            context={
                "paper_title": str(paper_structured.get("title", "")),
                "section_briefs": section_briefs,
                "claims": claim_briefs,
                "recommended_venues": rows[:5],
            },
            output_schema={
                "reason_overrides": [
                    {
                        "venue": "iclr",
                        "year": 2026,
                        "match_score_adjust": 0.02,
                        "reasons": ["string"],
                        "specific_gap_summary": "string",
                        "fit_summary": "string",
                    }
                ]
            },
            model_profile="judge",
        )

        result = self.executor.execute(spec_task)
        fallback_detected = False
        for warning in result.warnings:
            ctx.add_qa_issue(f"venue_recommender_agent_warning:{warning}")
            lower = str(warning).lower()
            if "fallback" in lower or "api_key_missing" in lower:
                fallback_detected = True
        if fallback_detected:
            ctx.add_qa_issue("venue_recommender_refine_fallback_detected_skip_refine")
            return None
        if not result.ok:
            return None

        data: Any = result.output
        if isinstance(data, dict) and isinstance(data.get("response"), dict):
            data = data.get("response")
        if not isinstance(data, dict):
            return None
        overrides = data.get("reason_overrides", [])
        if not isinstance(overrides, list) or not overrides:
            return None

        by_key = {
            (
                str(row.get("venue", "")).strip().lower(),
                int(row.get("year", 0) or 0),
            ): row
            for row in payload.get("recommended_venues", [])
            if isinstance(row, dict)
        }
        changed = 0
        for ov in overrides:
            if not isinstance(ov, dict):
                continue
            venue = str(ov.get("venue", "")).strip().lower()
            year = int(ov.get("year", 0) or 0)
            base = by_key.get((venue, year))
            if not isinstance(base, dict):
                continue

            delta = ov.get("match_score_adjust", 0.0)
            try:
                delta_f = float(delta)
            except (TypeError, ValueError):
                delta_f = 0.0
            delta_f = max(-0.08, min(0.08, delta_f))
            base_score = float(base.get("match_score", 0.0) or 0.0)
            base["match_score"] = round(max(0.0, min(1.0, base_score + delta_f)), 3)

            reasons_raw = ov.get("reasons", [])
            if isinstance(reasons_raw, list):
                existing = list(base.get("reasons", [])) if isinstance(base.get("reasons", []), list) else []
                for r in reasons_raw:
                    text = str(r).strip()
                    if text and text not in existing:
                        existing.append(text)
                base["reasons"] = existing[:10]

            gap_summary = str(ov.get("specific_gap_summary", "")).strip()
            fit_summary = str(ov.get("fit_summary", "")).strip()
            if gap_summary:
                base["agent_specific_gap_summary"] = gap_summary
            if fit_summary:
                base["agent_fit_summary"] = fit_summary
            base["agent_refined"] = True
            changed += 1

        if changed <= 0:
            return None

        rows = payload.get("recommended_venues", [])
        if isinstance(rows, list):
            rows = self._enforce_score_dispersion(rows)
            rows.sort(key=lambda x: float(x.get("match_score", 0.0)), reverse=True)
            payload["recommended_venues"] = rows
        payload["method"] = str(payload.get("method", "rule_based")) + "+agent_refine"
        payload["agent_refine"] = {"used": True, "changed_rows": changed}
        return payload

    @staticmethod
    def _safe_score(value: object, *, fallback: float) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float(fallback)
        return max(0.0, min(1.0, v))

    @staticmethod
    def _blend_agent_match_score(
        *,
        semantic_fit_score: float,
        review_risk_score: float,
        model_match_score: float,
        rule_prior_score: float,
    ) -> float:
        semantic_component = 0.70 * semantic_fit_score + 0.30 * (1.0 - review_risk_score)
        blended = 0.72 * semantic_component + 0.18 * model_match_score + 0.10 * rule_prior_score
        return max(0.55, min(0.90, blended))

    @staticmethod
    def _build_agent_candidate_priors(rows: list[dict]) -> list[dict]:
        out: list[dict] = []
        for row in rows[:8]:
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "venue": str(row.get("venue", "")).strip().lower(),
                    "year": int(row.get("year", 0) or 0),
                    "rule_prior_score": float(row.get("match_score", 0.0) or 0.0),
                    "rule_readiness": row.get("rule_readiness", {}),
                    "strength_checks": row.get("passed_checks", [])[:4] if isinstance(row.get("passed_checks", []), list) else [],
                    "gap_checks": row.get("failed_checks", [])[:4] if isinstance(row.get("failed_checks", []), list) else [],
                    "reason_snippets": row.get("reasons", [])[:3] if isinstance(row.get("reasons", []), list) else [],
                }
            )
        return out

    @staticmethod
    def _enforce_score_dispersion(rows: list[dict]) -> list[dict]:
        if not isinstance(rows, list) or len(rows) <= 1:
            return rows
        clean_rows = [dict(x) for x in rows if isinstance(x, dict)]
        if len(clean_rows) <= 1:
            return clean_rows

        scores = [float(x.get("match_score", 0.0) or 0.0) for x in clean_rows]
        if max(scores) - min(scores) >= 0.015 and len({round(s, 3) for s in scores}) >= 2:
            return clean_rows

        def discr(row: dict) -> tuple:
            semantic = float(row.get("semantic_fit_score", row.get("match_score", 0.0)) or 0.0)
            risk = float(row.get("review_risk_score", 1.0 - float(row.get("match_score", 0.0) or 0.0)) or 0.0)
            prior = float(row.get("rule_prior_score", 0.0) or 0.0)
            passed = len(row.get("passed_checks", [])) if isinstance(row.get("passed_checks", []), list) else 0
            failed = len(row.get("failed_checks", [])) if isinstance(row.get("failed_checks", []), list) else 0
            return (semantic, -risk, prior, passed - failed, -failed)

        clean_rows.sort(key=discr, reverse=True)
        anchor = max(0.62, min(0.86, float(clean_rows[0].get("match_score", 0.72) or 0.72)))
        counts = Counter(round(float(x.get("match_score", 0.0) or 0.0), 3) for x in clean_rows)
        tie_detected = any(v > 1 for v in counts.values())
        for idx, row in enumerate(clean_rows):
            adjusted = max(0.55, min(0.90, anchor - 0.016 * idx))
            row["match_score"] = round(adjusted, 3)
            if tie_detected:
                reasons = row.get("reasons", [])
                if isinstance(reasons, list):
                    hint = "Score tie-break applied using semantic fit, risk burden, and venue-specific gaps."
                    if hint not in reasons:
                        reasons.append(hint)
                    row["reasons"] = reasons[:10]
        return clean_rows
