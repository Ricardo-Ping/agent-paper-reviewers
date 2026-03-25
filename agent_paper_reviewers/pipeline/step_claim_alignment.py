from __future__ import annotations

from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency fallback
    fuzz = None

from ..models import ClaimAlignment, EvidenceRef
from ..services.embedding import cosine_similarity, encode_texts
from .base import PipelineContext, PipelineStep


class ClaimEvidenceAlignerStep(PipelineStep):
    name = "ClaimEvidenceAligner"

    def run(self, ctx: PipelineContext) -> None:
        claims = ctx.artifacts["claims_normalized"]["claims"]
        passages = ctx.artifacts["evidence_index"]["passages"]
        evidence_vectors = ctx.artifacts.get("evidence_vectors", {})

        passage_embeddings = [evidence_vectors.get(p.get("id", ""), []) for p in passages]
        if not passage_embeddings or not all(isinstance(v, list) and v for v in passage_embeddings):
            # Safety path: re-embed if index did not include vectors for some reason.
            passage_embeddings, _ = encode_texts(p.get("text", "") for p in passages)

        claim_queries = [
            self._claim_query(claim)
            for claim in claims
        ]
        claim_embeddings, _ = encode_texts(claim_queries)

        matrix: list[dict] = []
        for claim, claim_vec in zip(claims, claim_embeddings):
            scored = []
            for passage, passage_vec in zip(passages, passage_embeddings):
                semantic = cosine_similarity(claim_vec, passage_vec)
                lexical = self._lexical_score(claim["claim_text"], passage["text"])
                score = round(0.72 * semantic + 0.28 * lexical, 4)
                if score > 0.12:
                    scored.append((score, semantic, lexical, passage))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:3]
            top_score = top[0][0] if top else 0.0
            strength = self._strength(top_score)

            refs = [
                EvidenceRef(
                    section=item[3]["section"],
                    passage_id=item[3]["id"],
                    excerpt=item[3]["text"][:220],
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
            row = record.model_dump()
            if top:
                row["score_breakdown"] = {
                    "semantic": round(top[0][1], 3),
                    "lexical": round(top[0][2], 3),
                }
            matrix.append(row)

        payload = {"alignments": matrix}
        ctx.artifacts["claim_evidence_matrix"] = payload
        ctx.dump_json("artifacts/claim_evidence_matrix.json", payload)

    @staticmethod
    def _claim_query(claim: dict) -> str:
        parts = [
            str(claim.get("claim_text", "")),
            str(claim.get("verifiable_claim", "")),
            str(claim.get("success_criteria", "")),
        ]
        return "\n".join(p for p in parts if p).strip()

    @staticmethod
    def _lexical_score(a: str, b: str) -> float:
        if fuzz is not None:
            return fuzz.token_set_ratio(a, b) / 100.0
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _strength(score: float) -> str:
        if score >= 0.72:
            return "Strong"
        if score >= 0.54:
            return "Medium"
        if score >= 0.36:
            return "Weak"
        return "None"
