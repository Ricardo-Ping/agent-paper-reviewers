from __future__ import annotations

from pathlib import Path
import re

from ..executors.base import ExecutorAdapter
from ..models import EvidenceRef, RiskItem, ScoreBundle, TaskSpec
from ..services.feedback_store import (
    apply_feedback_profile,
    build_feedback_profile,
    load_feedback_records,
)
from .base import PipelineContext, PipelineStep


class RiskRankerStep(PipelineStep):
    name = "RiskRanker"

    def __init__(self, executor: ExecutorAdapter | None = None) -> None:
        self.executor = executor

    def run(self, ctx: PipelineContext) -> None:
        alignments = ctx.artifacts["claim_evidence_matrix"]["alignments"]
        gaps = ctx.artifacts["gaps"]["gaps"]
        venue_profile = ctx.artifacts.get("venue_profile", {}).get("profile", {})

        payload = self._rank_with_executor(ctx, alignments, gaps)
        if payload is None:
            payload = self._rank_rule_based(ctx, alignments, gaps, venue_profile)

        payload = self._apply_feedback_loop(ctx, payload, venue_profile)
        payload = self._apply_stage_strategy(ctx, payload, venue_profile)
        payload["score_explanations"] = self._ensure_score_explanations(
            payload,
            alignments=alignments,
            gaps=gaps,
        )
        ctx.artifacts["risk_ranking"] = payload
        ctx.dump_json("artifacts/risk_ranking.json", payload)

    def _apply_feedback_loop(
        self,
        ctx: PipelineContext,
        payload: dict,
        venue_profile: dict,
    ) -> dict:
        risks = payload.get("risks")
        if not isinstance(risks, list) or not risks:
            payload["feedback_loop"] = {
                "records_loaded": 0,
                "profiles_loaded": 0,
                "matched_risks": 0,
                "adjustments": [],
                "scores_recomputed": False,
            }
            return payload

        repo_root = ctx.repo_root or Path.cwd()
        venue_name = str(ctx.input_data.venue.name or "")
        venue_year = int(ctx.input_data.venue.year or 0)

        try:
            records = load_feedback_records(repo_root, venue_name, venue_year)
            profile = build_feedback_profile(records)
            adjusted, signals = apply_feedback_profile(risks, profile)
        except Exception as exc:  # noqa: BLE001
            ctx.add_qa_issue(f"feedback_loop_apply_failed:{exc}")
            payload["feedback_loop"] = {
                "records_loaded": 0,
                "profiles_loaded": 0,
                "matched_risks": 0,
                "adjustments": [],
                "scores_recomputed": False,
            }
            return payload

        payload["risks"] = adjusted
        feedback_meta = {
            "venue": venue_name,
            "year": venue_year,
            "records_loaded": len(records),
            "profiles_loaded": int(signals.get("profiles_loaded", 0) or 0),
            "matched_risks": int(signals.get("matched_risks", 0) or 0),
            "adjustments": signals.get("adjustments", []),
            "scores_recomputed": False,
        }

        if feedback_meta["adjustments"]:
            payload["scores"] = self._build_scores(
                payload["risks"],
                venue_profile.get("weights", {}),
            ).model_dump()
            feedback_meta["scores_recomputed"] = True
            explanations = payload.get("score_explanations")
            if isinstance(explanations, dict):
                payload["score_explanations"] = self._sync_explanation_scores(
                    explanations,
                    payload["scores"],
                )

        payload["feedback_loop"] = feedback_meta
        ctx.artifacts["feedback_profile"] = feedback_meta
        ctx.dump_json("artifacts/feedback_profile.json", feedback_meta)
        return payload

    def _apply_stage_strategy(
        self,
        ctx: PipelineContext,
        payload: dict,
        venue_profile: dict,
    ) -> dict:
        stage = ctx.input_data.review_context.manuscript_stage.value
        comments = ctx.input_data.review_context.reviewer_comments
        risks_raw = payload.get("risks", [])
        if not isinstance(risks_raw, list):
            risks_raw = []
        risks: list[dict] = [dict(r) for r in risks_raw if isinstance(r, dict)]

        stage_meta: dict = {
            "manuscript_stage": stage,
            "reviewer_comments_count": len(comments),
            "reviewer_comment_alignment": {"items": [], "unmatched_comments": []},
            "used_for_ranking": False,
        }

        if not comments or not risks:
            payload["stage_strategy"] = stage_meta
            payload["focus_risks"] = risks[:]
            return payload

        alignment_items: list[dict] = []
        unmatched_comments: list[dict] = []
        matched_risk_ids: list[str] = []

        for comment in comments:
            concern = str(comment.concern or "").strip()
            review_id = str(comment.review_id or "").strip() or f"R{len(alignment_items) + 1}"
            best_idx = -1
            best_score = 0.0
            for idx, risk in enumerate(risks):
                score = self._concern_match_score(concern, str(risk.get("reason", "")))
                if score > best_score:
                    best_idx = idx
                    best_score = score

            if best_idx < 0 or best_score < 0.08:
                unmatched_comments.append({"review_id": review_id, "concern": concern})
                continue

            risk = risks[best_idx]
            matched_risk_ids.append(str(risk.get("id", "")))
            alignment_items.append(
                {
                    "review_id": review_id,
                    "concern": concern,
                    "matched_risk_id": risk.get("id"),
                    "match_score": round(best_score, 3),
                }
            )
            if stage in {"rejected_after_reviews", "meta_review_discussion"}:
                risk["score"] = round(min(0.99, float(risk.get("score", 0.45)) + 0.08), 3)
                tags = risk.get("matched_reviewer_ids", [])
                if not isinstance(tags, list):
                    tags = []
                if review_id not in tags:
                    tags.append(review_id)
                risk["matched_reviewer_ids"] = tags

        if stage in {"rejected_after_reviews", "meta_review_discussion"}:
            stage_meta["used_for_ranking"] = True
            for idx, item in enumerate(unmatched_comments, start=1):
                synth = RiskItem(
                    id=f"RISK-RV-{idx:03d}",
                    severity="P1" if stage == "rejected_after_reviews" else "P0",
                    score=0.72 if stage == "rejected_after_reviews" else 0.78,
                    reason=f"Reviewer {item['review_id']} concern needs direct response: {item['concern']}",
                    evidence_refs=[],
                    likely_reject_phrase="This reviewer concern is not directly answered in the current draft.",
                    fix_hint="Provide a point-by-point response with explicit new evidence and exact paper change locations.",
                ).model_dump()
                synth["source"] = "reviewer_comment"
                synth["matched_reviewer_ids"] = [item["review_id"]]
                risks.append(synth)
                matched_risk_ids.append(synth["id"])

            risks.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            payload["scores"] = self._build_scores(
                risks,
                venue_profile.get("weights", {}),
            ).model_dump()
            explanations = payload.get("score_explanations")
            if isinstance(explanations, dict):
                payload["score_explanations"] = self._sync_explanation_scores(
                    explanations,
                    payload["scores"],
                )
        else:
            stage_meta["reviewer_comment_alignment"] = {
                "items": alignment_items,
                "unmatched_comments": unmatched_comments,
            }
            payload["risks"] = risks
            payload["focus_risks"] = risks[:]
            payload["stage_strategy"] = stage_meta
            return payload

        focus_ids = set(matched_risk_ids)
        focus_risks = [r for r in risks if str(r.get("id", "")) in focus_ids]
        if not focus_risks:
            focus_risks = risks[:]
        focus_risks.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        stage_meta["reviewer_comment_alignment"] = {
            "items": alignment_items,
            "unmatched_comments": unmatched_comments,
        }
        payload["risks"] = risks
        payload["focus_risks"] = focus_risks
        payload["stage_strategy"] = stage_meta
        return payload

    @staticmethod
    def _concern_match_score(concern: str, risk_reason: str) -> float:
        a = RiskRankerStep._tokens(concern)
        b = RiskRankerStep._tokens(risk_reason)
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stop = {
            "the",
            "this",
            "that",
            "with",
            "from",
            "have",
            "has",
            "been",
            "were",
            "which",
            "into",
            "reviewer",
            "concern",
            "needs",
            "direct",
            "response",
            "current",
            "draft",
        }
        parts = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return {p for p in parts if p not in stop}

    def _rank_with_executor(
        self,
        ctx: PipelineContext,
        alignments: list[dict],
        gaps: list[dict],
    ) -> dict | None:
        if self.executor is None:
            return None

        spec = TaskSpec(
            task_type="risk_ranking",
            prompt=(
                "You are a strict conference reviewer. Rank rejection risks using the provided claim-evidence "
                "alignment and detected gaps. Return JSON with fields: risks, scores."
            ),
            context={
                "alignments": alignments,
                "gaps": gaps,
                "score_scale": "risk score in [0,1], where higher means higher rejection risk",
                "severity_levels": ["P0", "P1", "P2"],
            },
            output_schema={
                "risks": [
                    {
                        "id": "RISK-001",
                        "severity": "P1",
                        "score": 0.61,
                        "reason": "string",
                        "evidence_refs": [
                            {
                                "section": "string",
                                "passage_id": "string",
                                "excerpt": "string",
                            }
                        ],
                        "likely_reject_phrase": "string",
                        "fix_hint": "string",
                    }
                ],
                "scores": {
                    "novelty": 0.0,
                    "soundness": 0.0,
                    "experiment": 0.0,
                    "clarity": 0.0,
                    "overall": 0.0,
                },
                "score_explanations": {
                    "novelty": {"score": 0.0, "reasoning": "string", "signals": ["string"]},
                    "soundness": {"score": 0.0, "reasoning": "string", "signals": ["string"]},
                    "experiment": {"score": 0.0, "reasoning": "string", "signals": ["string"]},
                    "clarity": {"score": 0.0, "reasoning": "string", "signals": ["string"]},
                },
            },
            model_profile="judge",
        )

        result = self.executor.execute(spec)
        for w in result.warnings:
            ctx.add_qa_issue(f"risk_ranker_executor_warning:{w}")

        if not result.ok:
            ctx.add_qa_issue("risk_ranker_executor_not_ok_use_rule_fallback")
            return None

        risks = self._normalize_executor_risks(result.output.get("risks"))
        if risks is None:
            ctx.add_qa_issue("risk_ranker_executor_output_invalid_use_rule_fallback")
            return None

        scores = self._normalize_executor_scores(result.output.get("scores"), risks)
        explanations = self._normalize_score_explanations(
            raw=result.output.get("score_explanations"),
            scores=scores.model_dump(),
            risks=risks,
            alignments=alignments,
            gaps=gaps,
        )
        return {"scores": scores.model_dump(), "risks": risks, "score_explanations": explanations}

    @staticmethod
    def _normalize_executor_risks(raw: object) -> list[dict] | None:
        if not isinstance(raw, list):
            return None

        risks: list[dict] = []
        for idx, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue

            score = RiskRankerStep._coerce_score(item.get("score"))
            severity = str(item.get("severity", "")).upper().strip()
            if severity not in {"P0", "P1", "P2"}:
                severity = RiskRankerStep._severity_from_score(score)

            refs = []
            raw_refs = item.get("evidence_refs", [])
            if isinstance(raw_refs, list):
                for ref in raw_refs:
                    if isinstance(ref, dict):
                        try:
                            refs.append(EvidenceRef.model_validate(ref))
                        except Exception:  # noqa: BLE001
                            continue

            reason = str(item.get("reason") or "Insufficient evidence for core claims.").strip()
            likely_reject_phrase = str(
                item.get("likely_reject_phrase")
                or "Experimental evidence does not yet meet venue expectations."
            ).strip()
            fix_hint = str(
                item.get("fix_hint") or "Add focused experiments and clearer validation evidence."
            ).strip()

            risk = RiskItem(
                id=str(item.get("id") or f"RISK-{idx:03d}"),
                severity=severity,
                score=round(score, 3),
                reason=reason,
                evidence_refs=refs,
                likely_reject_phrase=likely_reject_phrase,
                fix_hint=fix_hint,
            )
            risks.append(risk.model_dump())

        if not risks:
            return None

        risks.sort(key=lambda x: x["score"], reverse=True)
        return risks

    @staticmethod
    def _normalize_executor_scores(raw: object, risks: list[dict]) -> ScoreBundle:
        if not isinstance(raw, dict):
            return RiskRankerStep._build_scores(risks)

        def to_score(v: object, default: float) -> float:
            try:
                value = float(v)
            except (TypeError, ValueError):
                value = default
            if value < 0:
                return 0.0
            if value > 10:
                return 10.0
            return round(value, 2)

        novelty = to_score(raw.get("novelty"), 6.0)
        soundness = to_score(raw.get("soundness"), 6.0)
        experiment = to_score(raw.get("experiment"), 6.0)
        clarity = to_score(raw.get("clarity"), 6.0)
        overall = to_score(raw.get("overall"), round((novelty + soundness + experiment + clarity) / 4.0, 2))

        return ScoreBundle(
            novelty=novelty,
            soundness=soundness,
            experiment=experiment,
            clarity=clarity,
            overall=overall,
        )

    @staticmethod
    def _coerce_score(value: object) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.45
        if score > 1.0 and score <= 10.0:
            score = score / 10.0
        return max(0.0, min(1.0, score))

    @staticmethod
    def _severity_from_score(score: float) -> str:
        if score >= 0.75:
            return "P0"
        if score >= 0.45:
            return "P1"
        return "P2"

    def _rank_rule_based(
        self,
        ctx: PipelineContext,
        alignments: list[dict],
        gaps: list[dict],
        venue_profile: dict,
    ) -> dict:
        risks: list[dict] = []
        index = 1

        for item in alignments:
            if item["strength"] in {"None", "Weak"}:
                base_score = 0.82 if item["strength"] == "None" else 0.55
                score = min(0.97, max(base_score, 1.0 - float(item.get("score", 0.0))))
                severity = "P0" if item["strength"] == "None" else "P1"
                refs = [EvidenceRef.model_validate(x) for x in item.get("evidence_refs", [])]
                section_hint = self._section_hint(refs)
                reason = (
                    f"Claim {item['claim_id']} has {item['strength'].lower()} evidence support."
                    f"{section_hint}"
                )
                risks.append(
                    RiskItem(
                        id=f"RISK-{index:03d}",
                        severity=severity,
                        score=round(score, 3),
                        reason=reason,
                        evidence_refs=refs,
                        likely_reject_phrase=self._likely_reject_phrase_for_claim(item),
                        fix_hint=self._fix_hint_for_claim(item),
                    ).model_dump()
                )
                if self._refs_look_noisy(refs):
                    ctx.add_qa_issue(
                        f"risk_ranker_warning:{item['claim_id']}:aligned_evidence_looks_noisy_verify_experiments_section"
                    )
                index += 1

        gap_score_map = {
            "weak_claim_alignment": 0.71,
            "claim_evidence_contradiction": 0.83,
            "missing_baseline": 0.66,
            "missing_significance": 0.62,
            "missing_ablation": 0.58,
            "missing_reproducibility": 0.52,
            "missing_error_analysis": 0.41,
            "missing_robustness": 0.56,
            "missing_contribution_alignment": 0.60,
            "missing_limitations": 0.40,
            "missing_ethics_limitations": 0.38,
            "missing_practical_impact": 0.44,
            "missing_qualitative_analysis": 0.43,
        }
        for gap in gaps:
            score = gap_score_map.get(gap["code"], 0.45)
            severity = "P0" if score >= 0.75 else "P1" if score >= 0.45 else "P2"
            risks.append(
                RiskItem(
                    id=f"RISK-{index:03d}",
                    severity=severity,
                    score=round(score, 3),
                    reason=gap["description"],
                    evidence_refs=[EvidenceRef.model_validate(x) for x in gap.get("evidence_refs", [])],
                    likely_reject_phrase=self._likely_reject_phrase_for_gap(gap),
                    fix_hint=self._fix_hint_for_gap(gap),
                ).model_dump()
            )
            index += 1

        for reason in venue_profile.get("common_reject_reasons", [])[:2]:
            if any(reason.lower() in r["reason"].lower() for r in risks):
                continue
            risks.append(
                RiskItem(
                    id=f"RISK-{index:03d}",
                    severity="P2",
                    score=0.38,
                    reason=reason,
                    evidence_refs=[],
                    likely_reject_phrase="Current draft still leaves reviewer concerns about contribution quality.",
                    fix_hint="Tighten claim-to-evidence linkage and clarify contribution boundaries.",
                ).model_dump()
            )
            index += 1

        risks.sort(key=lambda x: x["score"], reverse=True)

        scores = self._build_scores(risks, venue_profile.get("weights", {}))
        explanations = self._build_score_explanations(
            scores=scores.model_dump(),
            risks=risks,
            alignments=alignments,
            gaps=gaps,
        )
        return {
            "scores": scores.model_dump(),
            "risks": risks,
            "score_explanations": explanations,
        }

    @staticmethod
    def _section_hint(refs: list[EvidenceRef]) -> str:
        if not refs:
            return " No high-quality evidence anchor was found."
        sections = []
        for ref in refs:
            sec = str(ref.section or "").strip().lower()
            if sec and sec not in sections:
                sections.append(sec)
        if not sections:
            return ""
        return f" Current anchors are mostly in sections: {', '.join(sections[:2])}."

    @staticmethod
    def _refs_look_noisy(refs: list[EvidenceRef]) -> bool:
        if not refs:
            return False
        noisy = 0
        for ref in refs:
            text = str(ref.excerpt or "")
            token_count = len(text.split())
            numeric_tokens = sum(1 for t in text.split() if t.replace(".", "", 1).replace("%", "").isdigit())
            alpha_tokens = sum(1 for t in text.split() if any(ch.isalpha() for ch in t))
            if token_count > 0 and numeric_tokens / token_count > 0.45 and alpha_tokens / token_count < 0.35:
                noisy += 1
        return noisy >= 1

    @staticmethod
    def _likely_reject_phrase_for_claim(item: dict) -> str:
        claim_type = str(item.get("claim_type", "novelty")).lower()
        mapping = {
            "baseline": "Comparisons are not convincing under fair baseline settings.",
            "statistical": "Reported gains may not be statistically reliable yet.",
            "ablation": "Component contributions are not convincingly isolated.",
            "reproducibility": "Results are hard to reproduce from the current draft.",
            "novelty": "Core novelty claims are not yet supported by direct evidence.",
        }
        return mapping.get(claim_type, mapping["novelty"])

    @staticmethod
    def _fix_hint_for_claim(item: dict) -> str:
        claim_type = str(item.get("claim_type", "novelty")).lower()
        mapping = {
            "baseline": "Add matched-setting comparisons against strongest baselines and clarify fairness settings.",
            "statistical": "Add multi-seed mean/std, confidence intervals, and paired significance tests.",
            "ablation": "Add component-level and interaction ablations that isolate each design choice.",
            "reproducibility": "Add full implementation/configuration details and deterministic rerun instructions.",
            "novelty": "Add direct empirical evidence that maps one-to-one to the claim statement.",
        }
        return mapping.get(claim_type, mapping["novelty"])

    @staticmethod
    def _likely_reject_phrase_for_gap(gap: dict) -> str:
        code = str(gap.get("code", "")).lower()
        mapping = {
            "missing_baseline": "Baseline comparisons are not strong or fair enough for this venue.",
            "missing_significance": "Improvements are not statistically validated to reviewer standards.",
            "missing_ablation": "Ablation evidence is insufficient to justify component claims.",
            "missing_reproducibility": "Reproducibility details are below expected submission quality.",
            "claim_evidence_contradiction": "Some reported results appear to conflict with the stated claim direction.",
            "missing_reference_coverage": "Related work positioning is not sufficiently grounded in prior literature.",
            "missing_top_venue_related_work_coverage": "Recent top-venue positioning is underdeveloped.",
            "missing_workload_diversity": "Evaluation workloads are not representative enough.",
            "missing_scalability_evaluation": "Scalability evidence is incomplete for system-level claims.",
            "missing_efficiency_tradeoff": "Efficiency trade-off reporting is not convincing.",
            "weak_claim_alignment": "Key claims remain weakly grounded in direct evidence.",
        }
        return mapping.get(code, "Experimental evidence does not yet meet venue expectations.")

    @staticmethod
    def _fix_hint_for_gap(gap: dict) -> str:
        code = str(gap.get("code", "")).lower()
        mapping = {
            "missing_baseline": "Add at least two strong baselines under matched compute/data settings and discuss fairness.",
            "missing_significance": "Report multi-seed statistics with significance tests on all primary metrics.",
            "missing_ablation": "Add full ablation table covering each key component and interactions.",
            "missing_reproducibility": "Provide full hyperparameters, environment, data processing, and code release plan.",
            "claim_evidence_contradiction": (
                "Resolve claim-result conflicts explicitly: identify conflicting table/figure anchors, "
                "correct claim scope/direction, and add reconciled analysis."
            ),
            "missing_reference_coverage": "Expand related work with stronger baseline and nearest-neighbor prior methods.",
            "missing_top_venue_related_work_coverage": "Add recent top-venue papers (last 2-3 years) and position differences explicitly.",
            "missing_workload_diversity": "Add heterogeneous workloads/benchmarks and explain representativeness.",
            "missing_scalability_evaluation": "Add scale-up/scale-out results across data and cluster size axes.",
            "missing_efficiency_tradeoff": "Report throughput-latency-resource trade-off curves, not just single-point metrics.",
            "weak_claim_alignment": "Add claim-specific evidence blocks so each major claim has explicit supporting results.",
        }
        return mapping.get(code, "Address this with a focused experiment or analysis update.")

    def _ensure_score_explanations(
        self,
        payload: dict,
        *,
        alignments: list[dict],
        gaps: list[dict],
    ) -> dict:
        scores = payload.get("scores", {})
        risks = payload.get("risks", [])
        raw = payload.get("score_explanations")
        return self._normalize_score_explanations(
            raw=raw,
            scores=scores if isinstance(scores, dict) else {},
            risks=risks if isinstance(risks, list) else [],
            alignments=alignments,
            gaps=gaps,
        )

    def _normalize_score_explanations(
        self,
        *,
        raw: object,
        scores: dict,
        risks: list[dict],
        alignments: list[dict],
        gaps: list[dict],
    ) -> dict:
        built = self._build_score_explanations(
            scores=scores,
            risks=risks,
            alignments=alignments,
            gaps=gaps,
        )
        if not isinstance(raw, dict):
            return built

        normalized: dict[str, dict] = {}
        for axis in ("novelty", "soundness", "experiment", "clarity"):
            item = raw.get(axis)
            fallback = built.get(axis, {})
            score_value = float(scores.get(axis, fallback.get("score", 0.0)) or 0.0)
            if not isinstance(item, dict):
                normalized[axis] = {
                    "score": round(score_value, 2),
                    "reasoning": str(fallback.get("reasoning", "")),
                    "signals": fallback.get("signals", []),
                }
                continue

            reasoning = str(item.get("reasoning", "")).strip()
            if not reasoning:
                reasoning = str(fallback.get("reasoning", ""))
            raw_signals = item.get("signals")
            signals: list[str] = []
            if isinstance(raw_signals, list):
                for s in raw_signals:
                    t = str(s).strip()
                    if t:
                        signals.append(t)
            if not signals:
                signals = list(fallback.get("signals", []))

            normalized[axis] = {
                "score": round(score_value, 2),
                "reasoning": reasoning,
                "signals": signals,
            }
        return normalized

    @staticmethod
    def _sync_explanation_scores(explanations: dict, scores: dict) -> dict:
        out: dict[str, dict] = {}
        for axis in ("novelty", "soundness", "experiment", "clarity"):
            item = explanations.get(axis, {})
            score = float(scores.get(axis, 0.0) or 0.0)
            if isinstance(item, dict):
                out[axis] = {
                    "score": round(score, 2),
                    "reasoning": str(item.get("reasoning", "")).strip(),
                    "signals": list(item.get("signals", [])) if isinstance(item.get("signals"), list) else [],
                }
            else:
                out[axis] = {"score": round(score, 2), "reasoning": "", "signals": []}
        return out

    def _build_score_explanations(
        self,
        *,
        scores: dict,
        risks: list[dict],
        alignments: list[dict],
        gaps: list[dict],
    ) -> dict:
        weak_claims = [
            str(a.get("claim_id", ""))
            for a in alignments
            if str(a.get("strength", "")).lower() in {"none", "weak"}
        ]
        medium_claims = [
            str(a.get("claim_id", ""))
            for a in alignments
            if str(a.get("strength", "")).lower() == "medium"
        ]
        strong_count = sum(
            1
            for a in alignments
            if str(a.get("strength", "")).lower() == "strong"
        )

        gap_codes = {str(g.get("code", "")).strip().lower() for g in gaps if isinstance(g, dict)}
        top_risk_reasons = [
            str(r.get("reason", "")).strip()
            for r in risks[:3]
            if isinstance(r, dict)
        ]

        def axis_score(axis: str) -> float:
            try:
                return round(float(scores.get(axis, 0.0) or 0.0), 2)
            except (TypeError, ValueError):
                return 0.0

        novelty_signals = []
        if weak_claims:
            novelty_signals.append(f"weak_claims={','.join(weak_claims[:3])}")
        if "missing_top_venue_related_work_coverage" in gap_codes:
            novelty_signals.append("missing_top_venue_related_work_coverage")
        if "missing_reference_coverage" in gap_codes:
            novelty_signals.append("missing_reference_coverage")
        if "missing_contribution_alignment" in gap_codes:
            novelty_signals.append("missing_contribution_alignment")
        novelty_reason = (
            f"Novelty reflects contribution positioning with {len(weak_claims)} weak/none claim-evidence links "
            f"and {strong_count} strong links. "
            "The score is lowered when novelty claims are supported mainly by implementation/integration-level evidence "
            "without clear methodological differentiation."
        )

        soundness_signals = []
        for key in ("missing_significance", "missing_reproducibility", "weak_claim_alignment", "missing_error_analysis"):
            if key in gap_codes:
                soundness_signals.append(key)
        if medium_claims:
            soundness_signals.append(f"medium_claims={','.join(medium_claims[:3])}")
        soundness_reason = (
            "Soundness depends on statistical reliability and reproducibility completeness. "
            f"Current signals indicate {len(soundness_signals)} soundness-related concerns, "
            "so the score reflects technical risk even when headline results look promising."
        )

        experiment_signals = []
        for key in (
            "missing_baseline",
            "missing_ablation",
            "missing_workload_diversity",
            "missing_scalability_evaluation",
            "missing_efficiency_tradeoff",
        ):
            if key in gap_codes:
                experiment_signals.append(key)
        experiment_reason = (
            "Experiment score measures whether evaluation is complete enough for reviewer confidence "
            "(strong baselines, ablation, and robust workloads). "
            f"Detected experiment gaps: {', '.join(experiment_signals[:4]) or 'none major'}."
        )

        clarity_signals = []
        for key in ("missing_limitations", "missing_ethics_limitations", "missing_practical_impact", "missing_qualitative_analysis"):
            if key in gap_codes:
                clarity_signals.append(key)
        if top_risk_reasons:
            clarity_signals.append("top_risk_reason_present")
        clarity_reason = (
            "Clarity score focuses on how easily reviewers can verify claims from writing structure and disclosure. "
            "A higher clarity score means presentation is relatively understandable, not that the paper is free from technical risk."
        )

        return {
            "novelty": {
                "score": axis_score("novelty"),
                "reasoning": novelty_reason,
                "signals": novelty_signals,
            },
            "soundness": {
                "score": axis_score("soundness"),
                "reasoning": soundness_reason,
                "signals": soundness_signals,
            },
            "experiment": {
                "score": axis_score("experiment"),
                "reasoning": experiment_reason,
                "signals": experiment_signals,
            },
            "clarity": {
                "score": axis_score("clarity"),
                "reasoning": clarity_reason,
                "signals": clarity_signals,
            },
        }

    @staticmethod
    def _build_scores(risks: list[dict], weights: dict | None = None) -> ScoreBundle:
        p0 = sum(1 for r in risks if r["severity"] == "P0")
        p1 = sum(1 for r in risks if r["severity"] == "P1")
        p2 = sum(1 for r in risks if r["severity"] == "P2")

        novelty = max(0.0, 8.5 - 0.8 * p1 - 1.2 * p0)
        soundness = max(0.0, 8.0 - 1.0 * p1 - 1.5 * p0)
        experiment = max(0.0, 8.2 - 1.1 * p1 - 1.4 * p0)
        clarity = max(0.0, 8.8 - 0.5 * p2 - 0.6 * p1)
        venue_weights = weights if isinstance(weights, dict) else {}
        w_n = float(venue_weights.get("novelty", 0.25))
        w_s = float(venue_weights.get("soundness", 0.25))
        w_e = float(venue_weights.get("experiment", 0.25))
        w_c = float(venue_weights.get("clarity", 0.25))
        total_w = w_n + w_s + w_e + w_c
        if total_w <= 0:
            total_w = 1.0
            w_n = w_s = w_e = w_c = 0.25
        overall = round(
            (novelty * w_n + soundness * w_s + experiment * w_e + clarity * w_c) / total_w,
            2,
        )

        return ScoreBundle(
            novelty=round(novelty, 2),
            soundness=round(soundness, 2),
            experiment=round(experiment, 2),
            clarity=round(clarity, 2),
            overall=overall,
        )
