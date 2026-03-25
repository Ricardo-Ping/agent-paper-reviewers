from __future__ import annotations

import re

from .base import PipelineContext, PipelineStep


class ClaimNormalizerStep(PipelineStep):
    name = "ClaimNormalizer"

    def run(self, ctx: PipelineContext) -> None:
        paper = ctx.artifacts["paper_structured"]
        raw_text = paper.get("raw_text", "")

        auto_claims = self._extract_auto_claims(raw_text)
        merged = []
        seen = set()

        for claim in [*ctx.input_data.claims, *auto_claims]:
            normalized = re.sub(r"\s+", " ", claim.strip())
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            merged.append(normalized)

        normalized_claims = []
        for idx, claim in enumerate(merged, start=1):
            claim_type = self._infer_claim_type(claim)
            normalized_claims.append(
                {
                    "claim_id": f"C{idx}",
                    "claim_text": claim,
                    "claim_type": claim_type,
                }
            )

        payload = {"claims": normalized_claims}
        ctx.artifacts["claims_normalized"] = payload
        ctx.dump_json("artifacts/claims_normalized.json", payload)

    @staticmethod
    def _extract_auto_claims(raw_text: str) -> list[str]:
        candidate_sections = []
        lowered = raw_text.lower()
        for marker in ["abstract", "conclusion", "contributions"]:
            idx = lowered.find(marker)
            if idx >= 0:
                candidate_sections.append(raw_text[idx : idx + 2400])

        candidates = []
        for section in candidate_sections:
            sentences = re.split(r"(?<=[.!?])\s+", section)
            for s in sentences:
                clean = s.strip()
                if 20 <= len(clean) <= 250 and any(k in clean.lower() for k in ["we", "our", "improve", "achieve", "propose"]):
                    candidates.append(clean)
        return candidates[:8]

    @staticmethod
    def _infer_claim_type(text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["faster", "latency", "efficient", "memory"]):
            return "efficiency"
        if any(k in t for k in ["theory", "proof", "bound"]):
            return "theory"
        if any(k in t for k in ["generalize", "robust", "out-of-domain"]):
            return "generalization"
        return "performance"
