from __future__ import annotations

from ..models import EvidenceRef, GapItem
from .base import PipelineContext, PipelineStep


class GapDetectorStep(PipelineStep):
    name = "GapDetector"

    def run(self, ctx: PipelineContext) -> None:
        raw = ctx.artifacts["paper_structured"].get("raw_text", "").lower()
        passages = ctx.artifacts["evidence_index"].get("passages", [])
        checks = {
            "missing_baseline": {
                "keywords": ["baseline", "sota", "compared with", "comparison"],
                "severity_hint": "P1",
                "description": "Strong baseline or SOTA comparison appears insufficient.",
            },
            "missing_significance": {
                "keywords": ["p-value", "confidence interval", "std", "variance", "significance"],
                "severity_hint": "P1",
                "description": "Statistical significance evidence appears missing.",
            },
            "missing_ablation": {
                "keywords": ["ablation", "remove", "w/o", "without"],
                "severity_hint": "P1",
                "description": "Ablation study does not look comprehensive.",
            },
            "missing_error_analysis": {
                "keywords": ["error analysis", "failure case", "limitations", "qualitative error"],
                "severity_hint": "P2",
                "description": "Error analysis / failure cases are under-developed.",
            },
            "missing_reproducibility": {
                "keywords": ["seed", "hyperparameter", "implementation details", "code release"],
                "severity_hint": "P1",
                "description": "Reproducibility details are likely incomplete.",
            },
        }

        gaps = []
        for code, rule in checks.items():
            if not any(k in raw for k in rule["keywords"]):
                refs = self._probe_refs(passages, rule["keywords"])
                gaps.append(
                    GapItem(
                        code=code,
                        severity_hint=rule["severity_hint"],
                        description=rule["description"],
                        evidence_refs=refs,
                    ).model_dump()
                )

        payload = {"gaps": gaps}
        ctx.artifacts["gaps"] = payload
        ctx.dump_json("artifacts/gaps.json", payload)

    @staticmethod
    def _probe_refs(passages: list[dict], keywords: list[str]) -> list[EvidenceRef]:
        refs: list[EvidenceRef] = []
        for passage in passages[:30]:
            if any(k in passage["text"].lower() for k in keywords):
                refs.append(
                    EvidenceRef(
                        section=passage["section"],
                        passage_id=passage["id"],
                        excerpt=passage["text"][:180],
                    )
                )
            if len(refs) >= 2:
                break
        return refs
