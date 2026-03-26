from __future__ import annotations

import json
import re

from ..executors.base import ExecutorAdapter
from ..models import TaskSpec
from .base import PipelineContext, PipelineStep


class ClaimNormalizerStep(PipelineStep):
    name = "ClaimNormalizer"

    _allowed_types = {
        "novelty",
        "baseline",
        "ablation",
        "statistical",
        "reproducibility",
    }

    def __init__(self, executor: ExecutorAdapter | None = None) -> None:
        self.executor = executor

    def run(self, ctx: PipelineContext) -> None:
        paper = ctx.artifacts["paper_structured"]
        raw_text = paper.get("raw_text", "")

        discovered = ctx.artifacts.get("claim_discovery", {})
        selected_claims = discovered.get("selected_claims", []) if isinstance(discovered, dict) else []
        selected_claims = [str(c) for c in selected_claims if str(c).strip()]

        if selected_claims:
            merged = self._merge_claims(selected_claims, [])
        else:
            auto_claims = self._extract_auto_claims(raw_text)
            merged = self._merge_claims(ctx.input_data.claims, auto_claims)

        if not merged:
            merged = [
                "The paper presents contributions that require direct empirical validation and reproducibility evidence."
            ]
            ctx.add_qa_issue("claim_normalizer_warning:no_claim_input_use_fallback_claim")

        normalized_claims = []
        for idx, claim in enumerate(merged, start=1):
            record = self._normalize_claim_with_executor(ctx, idx, claim)
            if record is None:
                record = self._normalize_claim_rule_based(idx, claim)
            normalized_claims.append(record)

        payload = {"claims": normalized_claims}
        ctx.artifacts["claims_normalized"] = payload
        ctx.artifacts["normalized_claims"] = normalized_claims
        ctx.dump_json("artifacts/claims_normalized.json", payload)
        ctx.dump_json("artifacts/normalized_claims.json", normalized_claims)

    def _normalize_claim_with_executor(
        self,
        ctx: PipelineContext,
        idx: int,
        claim: str,
    ) -> dict | None:
        if self.executor is None:
            return None

        spec = TaskSpec(
            task_type="claim_normalize",
            prompt=(
                "Convert a raw paper claim into a verifiable structured claim. "
                "Return JSON object only."
            ),
            context={
                "claim_id": f"C{idx}",
                "raw_claim": claim,
                "allowed_types": sorted(self._allowed_types),
                "notes": "Do not invent numbers unless explicitly present in the claim text.",
            },
            output_schema={
                "claim_id": f"C{idx}",
                "text": "original claim text",
                "type": "novelty|baseline|ablation|statistical|reproducibility",
                "verifiable_claim": "specific verifiable statement",
                "success_criteria": "how to validate this claim",
                "weakness_hint": "possible weakness of this claim",
            },
            model_profile="extract",
        )

        result = self.executor.execute(spec)
        for w in result.warnings:
            ctx.add_qa_issue(f"claim_normalizer_executor_warning:{w}")
        if not result.ok:
            ctx.add_qa_issue(f"claim_normalizer_executor_not_ok:C{idx}")
            return None

        normalized = self._coerce_executor_record(idx, claim, result.output)
        if normalized is None:
            ctx.add_qa_issue(f"claim_normalizer_executor_output_invalid:C{idx}")
        return normalized

    def _coerce_executor_record(self, idx: int, raw_claim: str, output: dict) -> dict | None:
        if not isinstance(output, dict):
            return None

        data = output
        if isinstance(output.get("response"), dict):
            data = output["response"]
        elif isinstance(output.get("response"), str):
            try:
                parsed = json.loads(output["response"])
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:  # noqa: BLE001
                pass

        claim_type = str(data.get("type") or "").strip().lower()
        if claim_type not in self._allowed_types:
            claim_type = self._infer_claim_type(raw_claim)

        text = str(data.get("text") or raw_claim).strip()
        verifiable_claim = str(data.get("verifiable_claim") or "").strip()
        success_criteria = str(data.get("success_criteria") or "").strip()
        weakness_hint = str(data.get("weakness_hint") or "").strip()

        if not verifiable_claim:
            return None
        if not success_criteria:
            success_criteria = self._default_success_criteria(claim_type)
        if not weakness_hint:
            weakness_hint = self._default_weakness_hint(claim_type)

        return {
            "claim_id": str(data.get("claim_id") or f"C{idx}"),
            "claim_text": text,
            "claim_type": claim_type,
            "verifiable_claim": verifiable_claim,
            "success_criteria": success_criteria,
            "weakness_hint": weakness_hint,
        }

    def _normalize_claim_rule_based(self, idx: int, claim: str) -> dict:
        claim_type = self._infer_claim_type(claim)
        return {
            "claim_id": f"C{idx}",
            "claim_text": claim,
            "claim_type": claim_type,
            "verifiable_claim": self._default_verifiable_claim(claim_type, claim),
            "success_criteria": self._default_success_criteria(claim_type),
            "weakness_hint": self._default_weakness_hint(claim_type),
        }

    @staticmethod
    def _merge_claims(user_claims: list[str], auto_claims: list[str]) -> list[str]:
        merged = []
        seen = set()
        for claim in [*user_claims, *auto_claims]:
            normalized = re.sub(r"\s+", " ", claim.strip())
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
        return merged

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
                if 20 <= len(clean) <= 250 and any(
                    k in clean.lower() for k in ["we", "our", "improve", "achieve", "propose"]
                ):
                    candidates.append(clean)
        return candidates[:8]

    @staticmethod
    def _infer_claim_type(text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["baseline", "compare", "compared with", "outperform"]):
            return "baseline"
        if any(k in t for k in ["ablation", "component", "remove", "without"]):
            return "ablation"
        if any(k in t for k in ["significant", "p-value", "std", "variance", "seed"]):
            return "statistical"
        if any(k in t for k in ["reproduce", "reproduc", "code", "implementation", "deterministic"]):
            return "reproducibility"
        return "novelty"

    @staticmethod
    def _default_verifiable_claim(claim_type: str, claim: str) -> str:
        if claim_type == "baseline":
            return f"Compared with strong baselines under matched settings, the paper claim '{claim}' should show consistent improvement."
        if claim_type == "ablation":
            return f"Ablation experiments should isolate key components supporting the claim '{claim}'."
        if claim_type == "statistical":
            return f"The claim '{claim}' should remain valid with multi-seed statistics and significance testing."
        if claim_type == "reproducibility":
            return f"The claim '{claim}' should be reproducible with complete implementation and configuration details."
        return f"The novelty claim '{claim}' should be supported by concrete empirical or analytical evidence."

    @staticmethod
    def _default_success_criteria(claim_type: str) -> str:
        if claim_type == "baseline":
            return "Report consistent gains against strong baselines under identical training/evaluation settings."
        if claim_type == "ablation":
            return "Show each key component contributes measurably via controlled ablation tables."
        if claim_type == "statistical":
            return "Provide mean/std over multiple seeds and significance tests against top baselines."
        if claim_type == "reproducibility":
            return "Provide enough implementation detail so independent reruns can reproduce main results."
        return "Provide direct evidence linking the contribution claim to measurable outcomes."

    @staticmethod
    def _default_weakness_hint(claim_type: str) -> str:
        if claim_type == "baseline":
            return "Weak or mismatched baseline setup can invalidate comparative conclusions."
        if claim_type == "ablation":
            return "Missing controlled ablations can make causal attribution unclear."
        if claim_type == "statistical":
            return "Single-run metrics without significance testing may be unstable."
        if claim_type == "reproducibility":
            return "Insufficient implementation details may block independent verification."
        return "Novelty claims can be rejected if evidence is indirect or positioning is unclear."
