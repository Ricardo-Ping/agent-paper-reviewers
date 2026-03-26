from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from ..executors.base import ExecutorAdapter
from ..models import RebuttalBundle, RunStatus, TaskSpec
from ..services.translator import Translator
from .base import PipelineContext, PipelineStep
from .step_rebuttal import RebuttalComposerStep


class PaperQAGateStep(PipelineStep):
    name = "PaperQAGate"

    def __init__(self, translator: Translator, executor: ExecutorAdapter | None = None) -> None:
        self.translator = translator
        self.executor = executor

    def run(self, ctx: PipelineContext) -> None:
        rebuttal_payload = ctx.artifacts.get("rebuttal", {})
        if not isinstance(rebuttal_payload, dict) or "en" not in rebuttal_payload:
            ctx.add_qa_issue("paper_qa_gate_warning:missing_rebuttal_payload_skip")
            ctx.artifacts["paper_qa_gate"] = {
                "accepted": True,
                "source": "skip_missing_rebuttal",
                "issues": [],
                "rewrites_applied": 0,
            }
            ctx.dump_json("artifacts/paper_qa_gate.json", ctx.artifacts["paper_qa_gate"])
            return

        en_bundle = rebuttal_payload.get("en", {}).get("bundle", {})
        if not isinstance(en_bundle, dict):
            ctx.add_qa_issue("paper_qa_gate_warning:invalid_rebuttal_bundle_skip")
            ctx.artifacts["paper_qa_gate"] = {
                "accepted": True,
                "source": "skip_invalid_rebuttal",
                "issues": [],
                "rewrites_applied": 0,
            }
            ctx.dump_json("artifacts/paper_qa_gate.json", ctx.artifacts["paper_qa_gate"])
            return

        risk_ranking = ctx.artifacts.get("risk_ranking", {})

        review = self._self_review_with_executor(ctx, en_bundle, risk_ranking)
        source = "executor"
        if review is None:
            review = self._self_review_heuristic(en_bundle)
            source = "heuristic_fallback"

        rewrites_applied = 0
        initial_accept = bool(review.get("accept", True))
        final_accept = initial_accept
        if not review.get("accept", True):
            en_bundle, rewrites_applied = self._apply_rewrites(en_bundle, review.get("rewrites", []))
            if rewrites_applied > 0:
                # Re-check after rewrite
                post = self._self_review_with_executor(ctx, en_bundle, risk_ranking)
                if post is None:
                    post = self._self_review_heuristic(en_bundle)
                review["post_recheck_accept"] = bool(post.get("accept", False))
                review["post_recheck_issues"] = list(post.get("issues", []))
                final_accept = bool(post.get("accept", False))
                if not post.get("accept", False):
                    ctx.add_qa_issue("paper_qa_gate_warning:still_not_accepted_after_rewrite")
                    ctx.status = RunStatus.PARTIAL_FAILED
            else:
                ctx.add_qa_issue("paper_qa_gate_warning:failed_but_no_rewrite_generated")
                ctx.status = RunStatus.PARTIAL_FAILED
                final_accept = False

        # Persist updated rebuttal payload (always from EN source of truth).
        rebuttal_payload["en"]["bundle"] = en_bundle
        en_model = RebuttalBundle.model_validate(en_bundle)
        rebuttal_payload["en"]["markdown"] = RebuttalComposerStep._to_markdown(en_model)

        if ctx.input_data.options.language_mode.value == "en_zh":
            zh_bundle = self._translate_bundle_to_zh(en_bundle)
            zh_model = RebuttalBundle.model_validate(zh_bundle)
            rebuttal_payload["zh"] = {
                "bundle": zh_bundle,
                "markdown": RebuttalComposerStep._to_markdown_zh(zh_model),
            }

        ctx.artifacts["rebuttal"] = rebuttal_payload

        initial_issues = list(review.get("issues", []))
        post_issues = list(review.get("post_recheck_issues", []))
        if rewrites_applied > 0:
            final_issues = [] if final_accept else post_issues
        else:
            final_issues = [] if final_accept else initial_issues

        payload = {
            "accepted": final_accept,
            "initial_accept": initial_accept,
            "source": source,
            "issues": final_issues,
            "initial_issues": initial_issues,
            "per_item": review.get("per_item", []),
            "rewrites_applied": rewrites_applied,
            "post_recheck_accept": review.get("post_recheck_accept"),
            "post_recheck_issues": post_issues,
        }
        ctx.artifacts["paper_qa_gate"] = payload
        ctx.dump_json("artifacts/paper_qa_gate.json", payload)

    def _self_review_with_executor(
        self,
        ctx: PipelineContext,
        rebuttal_bundle_en: dict,
        risk_ranking: dict,
    ) -> dict | None:
        if self.executor is None:
            return None

        spec = TaskSpec(
            task_type="paper_qa_self_review",
            prompt=(
                "Act as a strict reviewer and QA gate for rebuttal quality. "
                "Assess whether rebuttal responses are specific, non-contradictory, and non-template. "
                "If weak, provide rewrites. Return JSON only."
            ),
            context={
                "venue": ctx.input_data.venue.name,
                "year": ctx.input_data.venue.year,
                "risk_ranking": risk_ranking,
                "rebuttal_bundle_en": rebuttal_bundle_en,
                "requirements": [
                    "Check whether each rebuttal item directly answers the concern.",
                    "Check internal consistency with risk ranking.",
                    "Detect template-like repetition across reviewers.",
                    "Detect vague statements with no concrete evidence plan.",
                    "If not acceptable, provide rewrites per review_id.",
                ],
            },
            output_schema={
                "accept": True,
                "issues": ["string"],
                "per_item": [
                    {"review_id": "R1", "verdict": "pass|fail", "issues": ["string"]}
                ],
                "rewrites": [
                    {
                        "review_id": "R1",
                        "response": "string",
                        "new_evidence": ["string"],
                        "paper_change": "string",
                    }
                ],
            },
            model_profile="judge",
        )
        result = self.executor.execute(spec)
        for warning in result.warnings:
            ctx.add_qa_issue(f"paper_qa_gate_executor_warning:{warning}")
        if not result.ok:
            ctx.add_qa_issue("paper_qa_gate_executor_not_ok_use_heuristic")
            return None

        parsed = self._normalize_review_output(result.output)
        if parsed is None:
            ctx.add_qa_issue("paper_qa_gate_executor_output_invalid_use_heuristic")
        return parsed

    @staticmethod
    def _normalize_review_output(raw: object) -> dict | None:
        data = raw
        if isinstance(raw, dict):
            if isinstance(raw.get("response"), dict):
                data = raw["response"]
            elif isinstance(raw.get("response"), str):
                try:
                    parsed = json.loads(raw["response"])
                    data = parsed
                except Exception:  # noqa: BLE001
                    data = raw
        if not isinstance(data, dict):
            return None

        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        per_item = data.get("per_item", [])
        if not isinstance(per_item, list):
            per_item = []
        rewrites = data.get("rewrites", [])
        if not isinstance(rewrites, list):
            rewrites = []

        normalized_rewrites = []
        for item in rewrites:
            if not isinstance(item, dict):
                continue
            review_id = str(item.get("review_id", "")).strip()
            if not review_id:
                continue
            response = str(item.get("response", "")).strip()
            paper_change = str(item.get("paper_change", "")).strip()
            new_evidence = item.get("new_evidence", [])
            if not isinstance(new_evidence, list):
                new_evidence = []
            normalized_rewrites.append(
                {
                    "review_id": review_id,
                    "response": response,
                    "new_evidence": [str(x).strip() for x in new_evidence if str(x).strip()],
                    "paper_change": paper_change,
                }
            )

        return {
            "accept": bool(data.get("accept", True)),
            "issues": [str(x).strip() for x in issues if str(x).strip()],
            "per_item": per_item,
            "rewrites": normalized_rewrites,
        }

    @staticmethod
    def _self_review_heuristic(rebuttal_bundle_en: dict) -> dict:
        items = rebuttal_bundle_en.get("items", []) if isinstance(rebuttal_bundle_en, dict) else []
        if not isinstance(items, list):
            items = []

        issues: list[str] = []
        per_item: list[dict] = []
        rewrites: list[dict] = []

        responses = [str(x.get("response", "")) for x in items if isinstance(x, dict)]
        repetitive = False
        if len(responses) >= 3:
            similar_pairs = 0
            total_pairs = 0
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    total_pairs += 1
                    sim = SequenceMatcher(None, responses[i].lower(), responses[j].lower()).ratio()
                    if sim >= 0.88:
                        similar_pairs += 1
            repetitive = total_pairs > 0 and (similar_pairs / total_pairs) >= 0.5
            if repetitive:
                issues.append("template_repetition_detected")

        for item in items:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("review_id", "")).strip() or "R?"
            concern = str(item.get("concern", "")).strip()
            response = str(item.get("response", "")).strip()
            new_evidence = item.get("new_evidence", [])
            if not isinstance(new_evidence, list):
                new_evidence = []
            paper_change = str(item.get("paper_change", "")).strip()

            local_issues: list[str] = []
            target_blob = " ".join([response, *[str(x) for x in new_evidence], paper_change]).lower()
            concern_tokens = set(re.findall(r"[a-zA-Z]{4,}", concern.lower()))
            overlap = 0.0
            if concern_tokens:
                overlap = sum(1 for t in concern_tokens if t in target_blob) / max(1, len(concern_tokens))
            has_number = bool(re.search(r"\b\d+(\.\d+)?%?\b", target_blob))
            has_anchor = bool(re.search(r"\b(section|table|figure|fig\.|tab\.)\b", target_blob))

            if overlap < 0.2:
                local_issues.append("low_concern_overlap")
            if not has_number and not has_anchor:
                local_issues.append("no_numeric_or_anchor_evidence")
            if repetitive:
                local_issues.append("template_like_response")

            verdict = "pass" if not local_issues else "fail"
            per_item.append({"review_id": rid, "verdict": verdict, "issues": local_issues})

            if local_issues:
                issues.extend([f"{rid}:{x}" for x in local_issues])
                rewrites.append(
                    {
                        "review_id": rid,
                        "response": (
                            response
                            + " To address this directly, we will add claim-specific numeric evidence and "
                            "explicit section/table anchors in the revision."
                        ).strip(),
                        "new_evidence": [
                            *[str(x).strip() for x in new_evidence if str(x).strip()],
                            "Add one claim-to-evidence table with exact numbers and section/table anchors.",
                        ],
                        "paper_change": (
                            "Update verified sections with a point-by-point mapping from reviewer concern "
                            "to new evidence and exact anchor locations."
                        ),
                    }
                )

        accept = not issues
        return {
            "accept": accept,
            "issues": list(dict.fromkeys(issues)),
            "per_item": per_item,
            "rewrites": rewrites,
        }

    @staticmethod
    def _apply_rewrites(bundle: dict, rewrites: list[dict]) -> tuple[dict, int]:
        if not isinstance(bundle, dict):
            return bundle, 0
        rewrite_map = {}
        if isinstance(rewrites, list):
            for row in rewrites:
                if not isinstance(row, dict):
                    continue
                rid = str(row.get("review_id", "")).strip()
                if rid:
                    rewrite_map[rid] = row

        items = bundle.get("items", [])
        if not isinstance(items, list):
            return bundle, 0

        changed = 0
        new_items = []
        for item in items:
            if not isinstance(item, dict):
                new_items.append(item)
                continue
            rid = str(item.get("review_id", "")).strip()
            rewrite = rewrite_map.get(rid)
            if not rewrite:
                new_items.append(item)
                continue

            merged = dict(item)
            if str(rewrite.get("response", "")).strip():
                merged["response"] = str(rewrite.get("response", "")).strip()
            if isinstance(rewrite.get("new_evidence"), list) and rewrite.get("new_evidence"):
                merged["new_evidence"] = [str(x).strip() for x in rewrite["new_evidence"] if str(x).strip()]
            if str(rewrite.get("paper_change", "")).strip():
                merged["paper_change"] = str(rewrite.get("paper_change", "")).strip()

            merged["char_count"] = len(
                PaperQAGateStep._compose_block(
                    concern=str(merged.get("concern", "")),
                    response=str(merged.get("response", "")),
                    new_evidence=merged.get("new_evidence", []) if isinstance(merged.get("new_evidence"), list) else [],
                    paper_change=str(merged.get("paper_change", "")),
                )
            )
            char_limit = int(merged.get("char_limit", 2500) or 2500)
            if int(merged["char_count"]) > char_limit:
                overflow = int(merged["char_count"]) - char_limit
                response = str(merged.get("response", ""))
                if overflow < len(response):
                    merged["response"] = response[:-overflow].rstrip()
                else:
                    merged["response"] = response[: max(40, len(response) // 3)].rstrip()
                merged["char_count"] = len(
                    PaperQAGateStep._compose_block(
                        concern=str(merged.get("concern", "")),
                        response=str(merged.get("response", "")),
                        new_evidence=merged.get("new_evidence", []) if isinstance(merged.get("new_evidence"), list) else [],
                        paper_change=str(merged.get("paper_change", "")),
                    )
                )

            new_items.append(merged)
            changed += 1

        out = dict(bundle)
        out["items"] = new_items
        return out, changed

    @staticmethod
    def _compose_block(concern: str, response: str, new_evidence: list[str], paper_change: str) -> str:
        return (
            f"Concern: {concern}\n"
            f"Response: {response}\n"
            f"New evidence: {'; '.join(new_evidence)}\n"
            f"Paper change: {paper_change}"
        )

    def _translate_bundle_to_zh(self, en_bundle: dict) -> dict:
        bundle = dict(en_bundle)
        items = bundle.get("items", [])
        zh_items: list[dict] = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                zh_items.append(
                    {
                        "review_id": item.get("review_id", ""),
                        "concern": self.translator.to_zh(str(item.get("concern", ""))),
                        "response": self.translator.to_zh(str(item.get("response", ""))),
                        "new_evidence": [
                            self.translator.to_zh(str(x))
                            for x in (item.get("new_evidence", []) if isinstance(item.get("new_evidence", []), list) else [])
                            if str(x).strip()
                        ],
                        "paper_change": self.translator.to_zh(str(item.get("paper_change", ""))),
                        "char_count": item.get("char_count", 0),
                        "char_limit": item.get("char_limit", 2500),
                    }
                )
        bundle["items"] = zh_items
        if bundle.get("global_response"):
            bundle["global_response"] = self.translator.to_zh(str(bundle.get("global_response", "")))
        return bundle
