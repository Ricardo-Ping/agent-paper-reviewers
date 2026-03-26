from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from ..executors.base import ExecutorAdapter
from ..models import TaskSpec
from .base import PipelineContext, PipelineStep


class ClaimDiscovererStep(PipelineStep):
    name = "ClaimDiscoverer"

    def __init__(self, executor: ExecutorAdapter | None = None) -> None:
        self.executor = executor

    def run(self, ctx: PipelineContext) -> None:
        structured = ctx.artifacts["paper_structured"]
        user_claims = self._clean_claims(ctx.input_data.claims)
        focus_sections = self._collect_focus_sections(structured)

        rule_candidates = self._extract_rule_candidates(focus_sections)
        llm_candidates = self._extract_llm_candidates(ctx, focus_sections, user_claims)
        suggested_candidates = self._merge_claims(rule_candidates, llm_candidates)

        additional_suggestions = [
            c for c in suggested_candidates if not self._contains_similar(user_claims, c)
        ]

        if user_claims:
            selected_claims = list(user_claims)
            confirmation_required = bool(additional_suggestions)
        else:
            selected_claims = list(suggested_candidates[:8])
            confirmation_required = bool(selected_claims)
            if selected_claims:
                ctx.add_qa_issue("claim_discovery_notice:auto_selected_claims_from_paper")

        if not selected_claims:
            selected_claims = self._fallback_claims(structured)
            confirmation_required = True
            ctx.add_qa_issue("claim_discovery_warning:no_claim_detected_use_fallback_claim")

        if additional_suggestions:
            ctx.add_qa_issue("claim_discovery_notice:suggested_additional_claims_available")

        input_guidance = self._build_input_guidance(
            user_claims=user_claims,
            selected_claims=selected_claims,
        )
        missing_categories = input_guidance.get("selected_claim_missing_categories", [])
        if isinstance(missing_categories, list) and missing_categories:
            ctx.add_qa_issue(
                "claim_discovery_guidance:missing_claim_categories:" + ",".join(str(x) for x in missing_categories)
            )

        payload = {
            "user_claims": user_claims,
            "rule_candidates": rule_candidates,
            "llm_candidates": llm_candidates,
            "suggested_candidates": suggested_candidates,
            "selected_claims": selected_claims,
            "input_guidance": input_guidance,
            "confirmation": {
                "required": confirmation_required,
                "message": (
                    "Review suggested claims in artifacts/claim_discovery.json and curate final claims "
                    "for a submission-grade rerun."
                ),
                "suggested_action": "confirm|add|remove",
            },
        }
        ctx.artifacts["claim_discovery"] = payload
        ctx.dump_json("artifacts/claim_discovery.json", payload)

    def _extract_llm_candidates(
        self,
        ctx: PipelineContext,
        focus_sections: list[dict],
        user_claims: list[str],
    ) -> list[str]:
        if self.executor is None or not focus_sections:
            return []

        spec = TaskSpec(
            task_type="claim_discover",
            prompt=(
                "Extract up to 8 concrete and verifiable paper claims from abstract/contribution/conclusion text. "
                "Return JSON only."
            ),
            context={
                "focus_sections": focus_sections,
                "existing_user_claims": user_claims,
                "must_be_verifiable": True,
            },
            output_schema={
                "claims": [
                    "A concise claim that can be validated by experiments, analysis, or reproducibility evidence."
                ]
            },
            model_profile="extract",
        )

        result = self.executor.execute(spec)
        for warning in result.warnings:
            ctx.add_qa_issue(f"claim_discoverer_executor_warning:{warning}")
        if not result.ok:
            ctx.add_qa_issue("claim_discoverer_executor_not_ok_use_rule_fallback")
            return []

        claims = self._coerce_executor_claims(result.output)
        if not claims:
            ctx.add_qa_issue("claim_discoverer_executor_output_invalid_use_rule_fallback")
        return claims

    @staticmethod
    def _coerce_executor_claims(output: dict) -> list[str]:
        data = output
        if isinstance(output.get("response"), dict):
            data = output["response"]
        elif isinstance(output.get("response"), str):
            try:
                parsed = json.loads(output["response"])
                if isinstance(parsed, dict):
                    data = parsed
                elif isinstance(parsed, list):
                    data = {"claims": parsed}
            except Exception:  # noqa: BLE001
                pass

        raw_claims = data.get("claims") if isinstance(data, dict) else None
        if not isinstance(raw_claims, list):
            return []
        cleaned = ClaimDiscovererStep._clean_claims(raw_claims)
        return cleaned[:8]

    @staticmethod
    def _collect_focus_sections(structured: dict) -> list[dict]:
        out: list[dict] = []
        sections = structured.get("sections") if isinstance(structured, dict) else []
        if isinstance(sections, list):
            for sec in sections:
                if not isinstance(sec, dict):
                    continue
                name = str(sec.get("name") or "").strip().lower()
                text = str(sec.get("text") or "").strip()
                if not text:
                    continue
                if any(k in name for k in ("abstract", "contribution", "conclusion", "summary")):
                    out.append({"name": name, "text": text[:3200]})

        if out:
            return out

        raw_text = str(structured.get("raw_text") or "")
        lowered = raw_text.lower()
        markers = ["abstract", "contribution", "contributions", "conclusion", "summary"]
        for marker in markers:
            idx = lowered.find(marker)
            if idx >= 0:
                snippet = raw_text[idx : idx + 2600]
                out.append({"name": marker, "text": snippet})
        return out

    @staticmethod
    def _extract_rule_candidates(focus_sections: list[dict]) -> list[str]:
        keywords = (
            "we propose",
            "we present",
            "we introduce",
            "we improve",
            "we achieve",
            "we outperform",
            "our method",
            "our approach",
            "significantly",
            "reduce",
            "increase",
        )
        candidates: list[str] = []
        for sec in focus_sections:
            text = str(sec.get("text") or "")
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sentence in sentences:
                clean = re.sub(r"\s+", " ", sentence).strip()
                if not (24 <= len(clean) <= 320):
                    continue
                lowered = clean.lower()
                if any(k in lowered for k in keywords):
                    candidates.append(clean)

        return ClaimDiscovererStep._clean_claims(candidates)[:12]

    @staticmethod
    def _clean_claims(claims: list[str]) -> list[str]:
        out: list[str] = []
        for raw in claims:
            text = re.sub(r"\s+", " ", str(raw or "").strip())
            if not text:
                continue
            if len(text) < 12:
                continue
            out.append(text)
        return ClaimDiscovererStep._dedupe_preserve_order(out)

    @staticmethod
    def _dedupe_preserve_order(claims: list[str]) -> list[str]:
        out: list[str] = []
        for claim in claims:
            if not ClaimDiscovererStep._contains_similar(out, claim):
                out.append(claim)
        return out

    @staticmethod
    def _merge_claims(a: list[str], b: list[str]) -> list[str]:
        return ClaimDiscovererStep._dedupe_preserve_order([*a, *b])

    @staticmethod
    def _contains_similar(existing: list[str], candidate: str, threshold: float = 0.92) -> bool:
        cand_norm = re.sub(r"\s+", " ", candidate.strip()).lower()
        if not cand_norm:
            return True
        for item in existing:
            item_norm = re.sub(r"\s+", " ", item.strip()).lower()
            if not item_norm:
                continue
            if cand_norm == item_norm:
                return True
            if SequenceMatcher(None, cand_norm, item_norm).ratio() >= threshold:
                return True
        return False

    @staticmethod
    def _claim_category(text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["baseline", "compare", "compared", "outperform", "sota", "state-of-the-art"]):
            return "baseline"
        if any(k in t for k in ["ablation", "without", "remove component", "component contribution"]):
            return "ablation"
        if any(k in t for k in ["significance", "p-value", "confidence interval", "std", "seed"]):
            return "statistical"
        if any(k in t for k in ["reproduc", "code", "implementation details", "deterministic", "rerun"]):
            return "reproducibility"
        return "novelty"

    def _coverage(self, claims: list[str]) -> dict[str, int]:
        buckets = {
            "novelty": 0,
            "baseline": 0,
            "ablation": 0,
            "statistical": 0,
            "reproducibility": 0,
        }
        for claim in claims:
            cat = self._claim_category(claim)
            buckets[cat] = buckets.get(cat, 0) + 1
        return buckets

    def _build_input_guidance(self, *, user_claims: list[str], selected_claims: list[str]) -> dict:
        user_cov = self._coverage(user_claims)
        selected_cov = self._coverage(selected_claims)
        required = ["novelty", "baseline", "statistical"]
        missing = [k for k in required if int(selected_cov.get(k, 0)) <= 0]

        questions = [
            {
                "id": "novelty_core",
                "category": "novelty",
                "question": "What are the 1-3 core contributions reviewers should remember after one read?",
                "why": "Reviewer first checks whether contribution statements are concrete and defensible.",
            },
            {
                "id": "baseline_difference",
                "category": "baseline",
                "question": "Which strong baselines did you compare with, and under what matched settings?",
                "why": "Weak or unfair baseline setup is a high-frequency rejection reason.",
            },
            {
                "id": "quant_results",
                "category": "statistical",
                "question": "Which quantitative gains are statistically validated (multi-seed, CI, p-value)?",
                "why": "Without statistical support, gains are often discounted by reviewers.",
            },
            {
                "id": "ablation_support",
                "category": "ablation",
                "question": "Which components are ablated and what does each component contribute?",
                "why": "Ablation is required to attribute gains to proposed components.",
            },
            {
                "id": "reproducibility_package",
                "category": "reproducibility",
                "question": "What reproducibility details are already ready (code, config, seeds, environment)?",
                "why": "Reproducibility gaps reduce reviewer confidence even with strong headline results.",
            },
        ]

        prompt_templates = {
            "novelty": "Claim template: We propose <new method/system> that addresses <problem> and differs from prior work by <key difference>.",
            "baseline": "Claim template: Under matched <data/compute/settings>, our method outperforms <baseline A/B> by <metric delta>.",
            "statistical": "Claim template: Gains remain significant with <N> seeds (mean/std, CI, p-value < threshold).",
            "ablation": "Claim template: Removing <component> causes <metric drop>, validating its contribution.",
            "reproducibility": "Claim template: Main results are reproducible with released <code/config/seed/environment>.",
        }

        return {
            "goal": "Help authors provide reviewer-checkable claims before normalization and alignment.",
            "questions": questions,
            "user_claim_category_coverage": user_cov,
            "selected_claim_category_coverage": selected_cov,
            "selected_claim_missing_categories": missing,
            "recommended_prompt_templates": [prompt_templates[k] for k in missing] if missing else [],
            "curation_checklist": [
                "Each key contribution should map to at least one explicit claim.",
                "Each claim should point to at least one concrete evidence anchor (table/figure/section).",
                "At least one claim should explicitly cover statistical validity.",
            ],
        }

    @staticmethod
    def _fallback_claims(structured: dict) -> list[str]:
        title = str(structured.get("title") or "the proposed method").strip()
        return [
            f"The paper claims that {title} provides measurable improvements over prior methods.",
            f"The paper claims that {title} is supported by reproducible empirical evidence.",
        ]
