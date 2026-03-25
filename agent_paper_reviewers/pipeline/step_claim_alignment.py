from __future__ import annotations

from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency fallback
    fuzz = None

from ..models import ClaimAlignment, EvidenceRef
from .base import PipelineContext, PipelineStep


class ClaimEvidenceAlignerStep(PipelineStep):
    name = "ClaimEvidenceAligner"

    def run(self, ctx: PipelineContext) -> None:
        claims = ctx.artifacts["claims_normalized"]["claims"]
        passages = ctx.artifacts["evidence_index"]["passages"]

        matrix: list[dict] = []
        for claim in claims:
            scored = []
            for passage in passages:
                if fuzz is not None:
                    score = (
                        fuzz.token_set_ratio(claim["claim_text"], passage["text"]) / 100.0
                    )
                else:
                    score = SequenceMatcher(
                        None, claim["claim_text"], passage["text"]
                    ).ratio()
                if score > 0.1:
                    scored.append((score, passage))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:3]
            top_score = top[0][0] if top else 0.0
            strength = self._strength(top_score)

            refs = [
                EvidenceRef(
                    section=item[1]["section"],
                    passage_id=item[1]["id"],
                    excerpt=item[1]["text"][:220],
                )
                for item in top
            ]

            record = ClaimAlignment(
                claim_id=claim["claim_id"],
                claim_text=claim["claim_text"],
                strength=strength,
                score=round(top_score, 3),
                evidence_refs=refs,
            )
            matrix.append(record.model_dump())

        payload = {"alignments": matrix}
        ctx.artifacts["claim_evidence_matrix"] = payload
        ctx.dump_json("artifacts/claim_evidence_matrix.json", payload)

    @staticmethod
    def _strength(score: float) -> str:
        if score >= 0.78:
            return "Strong"
        if score >= 0.55:
            return "Medium"
        if score >= 0.35:
            return "Weak"
        return "None"
