from __future__ import annotations

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
                "Select and rank the best-fit venues for this paper draft. "
                "Use candidates as prior context but produce paper-specific reasons and concrete gap-to-fix notes."
            ),
            context={
                "paper_title": str(paper_structured.get("title", "")),
                "paper_summary": str(paper_structured.get("summary", ""))[:900],
                "section_briefs": section_briefs,
                "claims": claim_briefs,
                "candidate_venues": rows,
                "target_year": int(ctx.input_data.venue.year or 0),
            },
            output_schema={
                "recommended_venues": [
                    {
                        "venue": "iclr",
                        "year": 2026,
                        "match_score": 0.74,
                        "reasons": ["string"],
                        "fit_summary": "string",
                        "specific_gap_summary": "string",
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
        for warning in result.warnings:
            if "api_key_missing_use_fallback" in warning:
                continue
            ctx.add_qa_issue(f"venue_recommender_agent_warning:{warning}")
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
            score = max(0.0, min(1.0, score))
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
                    "match_score": round(score, 3),
                    "reasons": reasons[:10],
                    "fit_summary": str(row.get("fit_summary", "")).strip(),
                    "specific_gap_summary": str(row.get("specific_gap_summary", "")).strip(),
                    "required_check_passed_count": int(row.get("required_check_passed_count", 0) or 0),
                    "required_check_total": int(row.get("required_check_total", 0) or 0),
                    "passed_checks": row.get("passed_checks", []) if isinstance(row.get("passed_checks", []), list) else [],
                    "failed_checks": row.get("failed_checks", []) if isinstance(row.get("failed_checks", []), list) else [],
                    "agent_refined": True,
                }
            )

        if not rows_out:
            return None

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
        for warning in result.warnings:
            if "api_key_missing_use_fallback" in warning:
                continue
            ctx.add_qa_issue(f"venue_recommender_agent_warning:{warning}")
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
            rows.sort(key=lambda x: float(x.get("match_score", 0.0)), reverse=True)
            payload["recommended_venues"] = rows
        payload["method"] = str(payload.get("method", "rule_based")) + "+agent_refine"
        payload["agent_refine"] = {"used": True, "changed_rows": changed}
        return payload
