from __future__ import annotations

from ..models import EvidenceRef, RiskItem, ScoreBundle
from .base import PipelineContext, PipelineStep


class RiskRankerStep(PipelineStep):
    name = "RiskRanker"

    def run(self, ctx: PipelineContext) -> None:
        alignments = ctx.artifacts["claim_evidence_matrix"]["alignments"]
        gaps = ctx.artifacts["gaps"]["gaps"]

        risks: list[dict] = []
        index = 1

        for item in alignments:
            if item["strength"] in {"None", "Weak"}:
                score = 0.86 if item["strength"] == "None" else 0.56
                severity = "P0" if item["strength"] == "None" else "P1"
                risks.append(
                    RiskItem(
                        id=f"RISK-{index:03d}",
                        severity=severity,
                        score=score,
                        reason=f"Claim {item['claim_id']} has {item['strength']} evidence support.",
                        evidence_refs=[EvidenceRef.model_validate(x) for x in item["evidence_refs"]],
                        likely_reject_phrase="Core claims are not sufficiently supported by rigorous evidence.",
                        fix_hint="Add direct experiments and statistical validation tied to this claim.",
                    ).model_dump()
                )
                index += 1

        gap_score_map = {
            "missing_baseline": 0.66,
            "missing_significance": 0.62,
            "missing_ablation": 0.58,
            "missing_reproducibility": 0.52,
            "missing_error_analysis": 0.41,
        }
        for gap in gaps:
            score = gap_score_map.get(gap["code"], 0.45)
            severity = "P0" if score >= 0.75 else "P1" if score >= 0.45 else "P2"
            risks.append(
                RiskItem(
                    id=f"RISK-{index:03d}",
                    severity=severity,
                    score=score,
                    reason=gap["description"],
                    evidence_refs=[EvidenceRef.model_validate(x) for x in gap.get("evidence_refs", [])],
                    likely_reject_phrase="Experimental evidence does not yet meet venue expectations.",
                    fix_hint="Address this with a focused experiment or analysis update.",
                ).model_dump()
            )
            index += 1

        risks.sort(key=lambda x: x["score"], reverse=True)

        scores = self._build_scores(risks)
        payload = {
            "scores": scores.model_dump(),
            "risks": risks,
        }
        ctx.artifacts["risk_ranking"] = payload
        ctx.dump_json("artifacts/risk_ranking.json", payload)

    @staticmethod
    def _build_scores(risks: list[dict]) -> ScoreBundle:
        p0 = sum(1 for r in risks if r["severity"] == "P0")
        p1 = sum(1 for r in risks if r["severity"] == "P1")
        p2 = sum(1 for r in risks if r["severity"] == "P2")

        novelty = max(0.0, 8.5 - 0.8 * p1 - 1.2 * p0)
        soundness = max(0.0, 8.0 - 1.0 * p1 - 1.5 * p0)
        experiment = max(0.0, 8.2 - 1.1 * p1 - 1.4 * p0)
        clarity = max(0.0, 8.8 - 0.5 * p2 - 0.6 * p1)
        overall = round((novelty + soundness + experiment + clarity) / 4.0, 2)

        return ScoreBundle(
            novelty=round(novelty, 2),
            soundness=round(soundness, 2),
            experiment=round(experiment, 2),
            clarity=round(clarity, 2),
            overall=overall,
        )
