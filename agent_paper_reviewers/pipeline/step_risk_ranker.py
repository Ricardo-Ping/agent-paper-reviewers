from __future__ import annotations

from ..executors.base import ExecutorAdapter
from ..models import EvidenceRef, RiskItem, ScoreBundle, TaskSpec
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
            payload = self._rank_rule_based(alignments, gaps, venue_profile)

        ctx.artifacts["risk_ranking"] = payload
        ctx.dump_json("artifacts/risk_ranking.json", payload)

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
        return {"scores": scores.model_dump(), "risks": risks}

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

    def _rank_rule_based(self, alignments: list[dict], gaps: list[dict], venue_profile: dict) -> dict:
        risks: list[dict] = []
        index = 1

        for item in alignments:
            if item["strength"] in {"None", "Weak"}:
                base_score = 0.82 if item["strength"] == "None" else 0.55
                score = min(0.97, max(base_score, 1.0 - float(item.get("score", 0.0))))
                severity = "P0" if item["strength"] == "None" else "P1"
                risks.append(
                    RiskItem(
                        id=f"RISK-{index:03d}",
                        severity=severity,
                        score=round(score, 3),
                        reason=f"Claim {item['claim_id']} has {item['strength']} evidence support.",
                        evidence_refs=[EvidenceRef.model_validate(x) for x in item["evidence_refs"]],
                        likely_reject_phrase="Core claims are not sufficiently supported by rigorous evidence.",
                        fix_hint="Add direct experiments and statistical validation tied to this claim.",
                    ).model_dump()
                )
                index += 1

        gap_score_map = {
            "weak_claim_alignment": 0.71,
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
                    likely_reject_phrase="Experimental evidence does not yet meet venue expectations.",
                    fix_hint="Address this with a focused experiment or analysis update.",
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
        return {
            "scores": scores.model_dump(),
            "risks": risks,
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
