from __future__ import annotations

import json
import re

from ..executors.base import ExecutorAdapter
from ..models import RebuttalBundle, RebuttalItem, TaskSpec
from ..services.translator import Translator
from .base import PipelineContext, PipelineStep


class RebuttalComposerStep(PipelineStep):
    name = "RebuttalComposer"

    def __init__(
        self,
        translator: Translator,
        executor: ExecutorAdapter | None = None,
    ) -> None:
        self.translator = translator
        self.executor = executor

    def run(self, ctx: PipelineContext) -> None:
        profile = ctx.artifacts["venue_profile"]["profile"]
        policy = profile["rebuttal_policy"]
        ranking = ctx.artifacts["risk_ranking"]
        risks = ranking.get("risks", [])[:5]
        focus_risks = ranking.get("focus_risks", risks)
        remediation_tasks = ctx.artifacts.get("remediation_plan", {}).get("tasks", [])
        predicted_questions = ctx.artifacts.get("reviewer_questions", {}).get("questions", [])
        manuscript_stage = ctx.input_data.review_context.manuscript_stage.value
        reviewer_comments = ctx.input_data.review_context.reviewer_comments

        char_limit = int(policy.get("per_review_char_limit", 2500))
        mode = policy.get("mode", "per_review_only")
        global_mode = mode in {"global+per_review", "per_review_plus_global"}

        items = []
        precheck_records = []
        plan_items = []
        targets = self._build_targets(
            manuscript_stage=manuscript_stage,
            reviewer_comments=reviewer_comments,
            risks=focus_risks if isinstance(focus_risks, list) and focus_risks else risks,
        )
        for target in targets[:5]:
            target_followups = self._select_followups_for_target(
                risk=target.get("risk", {}) if isinstance(target, dict) else {},
                concern=str(target.get("concern", "") if isinstance(target, dict) else ""),
                predicted_questions=predicted_questions if isinstance(predicted_questions, list) else [],
            )
            plan_items.append(
                self._build_plan_item(
                    review_id=str(target.get("review_id", "")),
                    concern=str(target.get("concern", "")),
                    risk=target.get("risk", {}) if isinstance(target.get("risk", {}), dict) else {},
                    remediation_tasks=remediation_tasks if isinstance(remediation_tasks, list) else [],
                    predicted_followups=target_followups,
                )
            )
            item, precheck = self._compose_item(
                ctx,
                review_id=target["review_id"],
                concern=target["concern"],
                risk=target["risk"],
                remediation_tasks=remediation_tasks,
                predicted_followups=target_followups,
                char_limit=char_limit,
            )
            items.append(item)
            precheck_records.append(precheck)

        global_response = None
        if global_mode:
            global_response = self._compose_global_response(ctx, items)

        bundle_en = RebuttalBundle(
            venue=ctx.input_data.venue.name,
            year=ctx.input_data.venue.year,
            manuscript_stage=manuscript_stage,
            mode=mode,
            items=items,
            global_response=global_response,
            attachment_pdf=None,
        )

        md_en = self._to_markdown(bundle_en)

        payload = {
            "en": {
                "bundle": bundle_en.model_dump(),
                "markdown": md_en,
            }
        }

        if ctx.input_data.options.language_mode.value == "en_zh":
            bundle_zh = self._translate_bundle(bundle_en)
            payload["zh"] = {
                "bundle": bundle_zh.model_dump(),
                "markdown": self._to_markdown_zh(bundle_zh),
            }

        audit_payload = self._audit_plan_execution(
            plan_items=plan_items,
            items=items,
            precheck_records=precheck_records,
        )
        ctx.artifacts["rebuttal"] = payload
        ctx.artifacts["rebuttal_precheck"] = {"items": precheck_records}
        ctx.artifacts["rebuttal_plan"] = {
            "manuscript_stage": manuscript_stage,
            "plan_items": plan_items,
            "post_generation_audit": audit_payload.get("items", []),
            "summary": audit_payload.get("summary", {}),
        }
        ctx.dump_json("artifacts/rebuttal_bundle.en.json", payload["en"]["bundle"])
        ctx.dump_json("artifacts/rebuttal_precheck.json", ctx.artifacts["rebuttal_precheck"])
        ctx.dump_json("artifacts/rebuttal_plan.json", ctx.artifacts["rebuttal_plan"])
        if "zh" in payload:
            ctx.dump_json("artifacts/rebuttal_bundle.zh.json", payload["zh"]["bundle"])

    def _build_targets(
        self,
        *,
        manuscript_stage: str,
        reviewer_comments: list,
        risks: list[dict],
    ) -> list[dict]:
        if manuscript_stage in {"rejected_after_reviews", "meta_review_discussion"} and reviewer_comments:
            targets: list[dict] = []
            used: set[str] = set()
            for idx, comment in enumerate(reviewer_comments, start=1):
                review_id = str(getattr(comment, "review_id", "") or f"R{idx}")
                concern = str(getattr(comment, "concern", "") or "").strip()
                risk = self._best_match_risk(concern, risks)
                risk_id = str(risk.get("id", "")) if isinstance(risk, dict) else ""
                if risk_id and risk_id in used:
                    risk = self._next_unused_risk(risks, used)
                    risk_id = str(risk.get("id", "")) if isinstance(risk, dict) else ""
                if risk_id:
                    used.add(risk_id)
                targets.append(
                    {
                        "review_id": review_id,
                        "concern": concern or str(risk.get("reason") or "Reviewer concern"),
                        "risk": risk if isinstance(risk, dict) else {},
                    }
                )
            if targets:
                return targets

        targets = []
        for idx, risk in enumerate(risks[:5], start=1):
            if not isinstance(risk, dict):
                continue
            targets.append(
                {
                    "review_id": f"R{idx}",
                    "concern": str(risk.get("reason") or "").strip(),
                    "risk": risk,
                }
            )
        return targets

    @staticmethod
    def _build_plan_item(
        *,
        review_id: str,
        concern: str,
        risk: dict,
        remediation_tasks: list[dict],
        predicted_followups: list[dict],
    ) -> dict:
        risk_id = str(risk.get("id", "") if isinstance(risk, dict) else "").strip()
        linked_tasks: list[str] = []
        if isinstance(remediation_tasks, list):
            for task in remediation_tasks:
                if not isinstance(task, dict):
                    continue
                if str(task.get("risk_id", "")).strip() == risk_id:
                    tid = str(task.get("id", "")).strip()
                    if tid:
                        linked_tasks.append(tid)
        evidence_targets: list[str] = []
        if isinstance(predicted_followups, list):
            for row in predicted_followups:
                if not isinstance(row, dict):
                    continue
                for ev in row.get("evidence_to_prepare", []):
                    s = str(ev).strip()
                    if s and s not in evidence_targets:
                        evidence_targets.append(s)
        return {
            "review_id": review_id,
            "concern": concern,
            "risk_id": risk_id,
            "risk_reason": str(risk.get("reason", "") if isinstance(risk, dict) else ""),
            "linked_remediation_tasks": linked_tasks,
            "evidence_targets": evidence_targets[:4],
            "predicted_followup_questions": [
                str(row.get("question", "")).strip()
                for row in predicted_followups
                if isinstance(row, dict) and str(row.get("question", "")).strip()
            ][:3],
        }

    @staticmethod
    def _audit_plan_execution(
        *,
        plan_items: list[dict],
        items: list[RebuttalItem],
        precheck_records: list[dict],
    ) -> dict:
        item_map = {str(x.review_id): x for x in items}
        pre_map = {
            str(x.get("review_id", "")): x
            for x in precheck_records
            if isinstance(x, dict) and str(x.get("review_id", "")).strip()
        }

        audits: list[dict] = []
        status_counter = {"pass": 0, "warning": 0, "fail": 0}

        for plan in plan_items:
            if not isinstance(plan, dict):
                continue
            rid = str(plan.get("review_id", "")).strip()
            generated = item_map.get(rid)
            pre = pre_map.get(rid, {})

            if generated is None:
                audits.append(
                    {
                        "review_id": rid or "R?",
                        "status": "fail",
                        "gaps": ["missing_generated_item"],
                        "actions": ["Regenerate this rebuttal item with explicit concern mapping."],
                    }
                )
                status_counter["fail"] += 1
                continue

            blob = " ".join(
                [
                    str(generated.response or ""),
                    " ".join(str(x) for x in generated.new_evidence),
                    str(generated.paper_change or ""),
                ]
            ).lower()
            concern = str(plan.get("concern", "")).strip()
            concern_tokens = RebuttalComposerStep._audit_tokens(concern)
            concern_coverage = RebuttalComposerStep._coverage_ratio(concern_tokens, blob)

            evidence_targets = plan.get("evidence_targets", [])
            evidence_target_hit = RebuttalComposerStep._target_hit_ratio(evidence_targets, blob)
            followup_questions = plan.get("predicted_followup_questions", [])
            followup_preempted = RebuttalComposerStep._target_hit_ratio(followup_questions, blob) > 0.0
            anchor_present = bool(
                re.search(r"\b(section|table|figure|fig\.|tab\.)\b", blob)
                or re.search(r"\b\d+(\.\d+)?%?\b", blob)
            )
            char_budget_ok = int(generated.char_count) <= int(generated.char_limit)

            hallucination = pre.get("hallucination", {}) if isinstance(pre, dict) else {}
            hallucination_issues = hallucination.get("issues", []) if isinstance(hallucination, dict) else []
            if not isinstance(hallucination_issues, list):
                hallucination_issues = []
            has_hallucination = any(
                str(x).startswith("hallucinated_") or str(x).startswith("unverifiable_")
                for x in hallucination_issues
            )

            precheck_pass = bool(pre.get("pass", True)) if isinstance(pre, dict) else True
            gaps: list[str] = []
            actions: list[str] = []
            if concern_coverage < 0.25:
                gaps.append("low_concern_coverage")
                actions.append("Make response explicitly mention concern terms and direct fixes.")
            if evidence_target_hit < 0.34 and evidence_targets:
                gaps.append("weak_evidence_target_coverage")
                actions.append("Add evidence bullets that directly satisfy planned evidence targets.")
            if not followup_preempted and followup_questions:
                gaps.append("followup_not_preempted")
                actions.append("Add one sentence preempting the top predicted reviewer follow-up.")
            if not anchor_present:
                gaps.append("missing_anchor_or_number")
                actions.append("Add explicit section/table/figure anchors or concrete numbers.")
            if not char_budget_ok:
                gaps.append("char_budget_exceeded")
                actions.append("Shorten response while preserving evidence anchors.")
            if has_hallucination:
                gaps.append("hallucination_risk")
                actions.append("Replace unverifiable anchors with verified existing anchors.")
            if not precheck_pass:
                gaps.append("precheck_failed")
                actions.append("Use precheck issues to regenerate a stronger item.")

            if has_hallucination or not char_budget_ok:
                status = "fail"
            elif gaps:
                status = "warning"
            else:
                status = "pass"
            status_counter[status] += 1

            audits.append(
                {
                    "review_id": rid,
                    "risk_id": plan.get("risk_id", ""),
                    "status": status,
                    "metrics": {
                        "concern_coverage": round(concern_coverage, 3),
                        "evidence_target_hit_ratio": round(evidence_target_hit, 3),
                        "followup_preempted": bool(followup_preempted),
                        "anchor_present": bool(anchor_present),
                        "char_budget_ok": bool(char_budget_ok),
                        "precheck_pass": bool(precheck_pass),
                    },
                    "gaps": gaps,
                    "actions": list(dict.fromkeys(actions)),
                }
            )

        total = len(audits)
        return {
            "items": audits,
            "summary": {
                "total_items": total,
                "pass_count": status_counter["pass"],
                "warning_count": status_counter["warning"],
                "fail_count": status_counter["fail"],
                "manual_review_recommended": status_counter["warning"] > 0 or status_counter["fail"] > 0,
            },
        }

    @staticmethod
    def _audit_tokens(text: str) -> list[str]:
        stop = {
            "the",
            "this",
            "that",
            "with",
            "from",
            "have",
            "has",
            "will",
            "should",
            "needs",
            "need",
            "more",
            "less",
            "into",
            "about",
            "for",
            "and",
            "are",
            "was",
            "were",
        }
        out: list[str] = []
        seen: set[str] = set()
        for tok in re.findall(r"[a-zA-Z]{4,}", text.lower()):
            if tok in stop or tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
        return out[:10]

    @staticmethod
    def _coverage_ratio(tokens: list[str], blob: str) -> float:
        if not tokens:
            return 1.0
        hit = sum(1 for tok in tokens if tok in blob)
        return hit / max(1, len(tokens))

    @staticmethod
    def _target_hit_ratio(targets: object, blob: str) -> float:
        if not isinstance(targets, list) or not targets:
            return 1.0
        total = 0
        hit = 0
        for row in targets:
            text = str(row).strip()
            if not text:
                continue
            total += 1
            tokens = RebuttalComposerStep._audit_tokens(text)
            if RebuttalComposerStep._coverage_ratio(tokens, blob) >= 0.2:
                hit += 1
        if total <= 0:
            return 1.0
        return hit / total

    @staticmethod
    def _best_match_risk(concern: str, risks: list[dict]) -> dict:
        concern_tokens = set(re.findall(r"[a-zA-Z]{3,}", concern.lower()))
        best = risks[0] if risks else {}
        best_score = -1.0
        for risk in risks:
            if not isinstance(risk, dict):
                continue
            reason = str(risk.get("reason", ""))
            tokens = set(re.findall(r"[a-zA-Z]{3,}", reason.lower()))
            if not concern_tokens or not tokens:
                score = 0.0
            else:
                score = len(concern_tokens & tokens) / max(1, len(concern_tokens | tokens))
            if score > best_score:
                best_score = score
                best = risk
        return best if isinstance(best, dict) else {}

    @staticmethod
    def _next_unused_risk(risks: list[dict], used: set[str]) -> dict:
        for risk in risks:
            if not isinstance(risk, dict):
                continue
            rid = str(risk.get("id", ""))
            if rid and rid not in used:
                return risk
        return risks[0] if risks else {}

    @staticmethod
    def _select_followups_for_target(
        *,
        risk: dict,
        concern: str,
        predicted_questions: list[dict],
    ) -> list[dict]:
        if not isinstance(predicted_questions, list) or not predicted_questions:
            return []
        risk_id = str(risk.get("id", "") if isinstance(risk, dict) else "").strip()
        concern_tokens = set(re.findall(r"[a-zA-Z]{4,}", concern.lower()))
        scored: list[tuple[float, dict]] = []
        for q in predicted_questions:
            if not isinstance(q, dict):
                continue
            score = 0.0
            linked_ids = q.get("linked_risk_ids", [])
            if isinstance(linked_ids, list) and risk_id and risk_id in [str(x).strip() for x in linked_ids]:
                score += 0.9
            text_blob = " ".join(
                [
                    str(q.get("question", "")),
                    str(q.get("why_this_will_be_asked", "")),
                ]
            ).lower()
            if concern_tokens:
                q_tokens = set(re.findall(r"[a-zA-Z]{4,}", text_blob))
                score += 0.6 * (len(concern_tokens & q_tokens) / max(1, len(concern_tokens | q_tokens)))
            priority = str(q.get("priority", "medium")).lower()
            if priority == "high":
                score += 0.15
            elif priority == "medium":
                score += 0.07
            if score > 0:
                scored.append((score, q))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(item[1]) for item in scored[:2]]

    def _compose_item(
        self,
        ctx: PipelineContext,
        review_id: str,
        concern: str,
        risk: dict,
        remediation_tasks: list[dict],
        predicted_followups: list[dict],
        char_limit: int,
    ) -> tuple[RebuttalItem, dict]:
        concern = str(concern or risk.get("reason") or "").strip()
        fallback_response, fallback_evidence, fallback_change = self._template_by_risk(
            risk,
            remediation_tasks,
            concern_override=concern,
            followup_questions=predicted_followups,
        )

        response = fallback_response
        new_evidence = fallback_evidence
        paper_change = fallback_change
        source = "template"

        if self.executor is not None:
            generated = self._compose_item_with_executor(
                ctx,
                review_id=review_id,
                concern=concern,
                risk=risk,
                remediation_tasks=remediation_tasks,
                predicted_followups=predicted_followups,
                char_limit=char_limit,
            )
            if generated is not None:
                response = generated["response"]
                new_evidence = generated["new_evidence"]
                paper_change = generated["paper_change"]
                source = "executor"

        new_evidence = self._inject_followup_hint(
            new_evidence=new_evidence,
            predicted_followups=predicted_followups,
        )

        response, new_evidence, paper_change, precheck = self._precheck_and_repair(
            ctx,
            review_id=review_id,
            concern=concern,
            response=response,
            new_evidence=new_evidence,
            paper_change=paper_change,
            char_limit=char_limit,
        )
        precheck["source"] = source
        precheck["risk_id"] = str(risk.get("id") or "")

        composed = self._compose_block(concern, response, new_evidence, paper_change)
        if len(composed) > char_limit:
            response = self._shrink_response(response, len(composed) - char_limit)
            composed = self._compose_block(concern, response, new_evidence, paper_change)

        if len(composed) > char_limit:
            reserve = max(60, char_limit - len("Concern: \nResponse: \nNew evidence: \nPaper change: "))
            response = response[:reserve].rstrip()
            composed = self._compose_block(concern, response, new_evidence, paper_change)

        row = RebuttalItem(
            review_id=review_id,
            concern=concern,
            response=response,
            new_evidence=new_evidence,
            paper_change=paper_change,
            char_count=len(composed),
            char_limit=char_limit,
        )
        return row, precheck

    @staticmethod
    def _inject_followup_hint(new_evidence: list[str], predicted_followups: list[dict]) -> list[str]:
        if not isinstance(predicted_followups, list) or not predicted_followups:
            return list(new_evidence)
        out = [str(x).strip() for x in new_evidence if str(x).strip()]
        text_blob = " ".join(out).lower()
        for row in predicted_followups:
            if not isinstance(row, dict):
                continue
            q = str(row.get("question", "")).strip()
            if not q:
                continue
            key_tokens = re.findall(r"[a-zA-Z]{5,}", q.lower())
            if key_tokens and any(tok in text_blob for tok in key_tokens[:3]):
                continue
            out.append(f"Preempt likely follow-up: {q}")
            break
        return out

    def _compose_item_with_executor(
        self,
        ctx: PipelineContext,
        review_id: str,
        concern: str,
        risk: dict,
        remediation_tasks: list[dict],
        predicted_followups: list[dict],
        char_limit: int,
    ) -> dict | None:
        tasks_for_risk = [
            t
            for t in remediation_tasks
            if str(t.get("risk_id", "")).strip() == str(risk.get("id", "")).strip()
        ][:2]

        spec = TaskSpec(
            task_type="rebuttal_compose",
            prompt="Draft one reviewer response item in JSON only.",
            context={
                "review_id": review_id,
                "venue": ctx.input_data.venue.name,
                "year": ctx.input_data.venue.year,
                "concern": concern,
                "risk": risk,
                "related_tasks": tasks_for_risk,
                "predicted_followup_questions": predicted_followups,
                "evidence_refs": risk.get("evidence_refs", []),
                "char_limit": char_limit,
                "requirements": [
                    "Address the exact concern.",
                    "Include concrete new evidence and paper update location.",
                    "Prefer specific claim ids, section names, and figure/table anchors when available.",
                    "If possible, preempt one likely reviewer follow-up question.",
                    "No markdown fence; JSON only.",
                ],
            },
            output_schema={
                "response": "string",
                "new_evidence": ["string", "string"],
                "paper_change": "string",
            },
            model_profile="judge",
        )

        result = self.executor.execute(spec)
        for warning in result.warnings:
            ctx.add_qa_issue(f"rebuttal_executor_warning:{warning}")
        if not result.ok:
            return None

        parsed = self._parse_executor_payload(result.output)
        if parsed is None:
            ctx.add_qa_issue(f"rebuttal_executor_output_invalid:{review_id}")
        return parsed

    @staticmethod
    def _parse_executor_payload(output: dict) -> dict | None:
        if not isinstance(output, dict):
            return None

        data = output
        if isinstance(output.get("response"), str):
            try:
                parsed = json.loads(output["response"])
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:  # noqa: BLE001
                pass
        elif isinstance(output.get("response"), dict):
            data = output["response"]

        response = str(data.get("response") or "").strip()
        paper_change = str(data.get("paper_change") or "").strip()
        new_evidence_raw = data.get("new_evidence")
        if not response or not paper_change or not isinstance(new_evidence_raw, list):
            return None

        new_evidence = [str(x).strip() for x in new_evidence_raw if str(x).strip()]
        if not new_evidence:
            return None

        return {
            "response": response,
            "new_evidence": new_evidence[:4],
            "paper_change": paper_change,
        }

    @staticmethod
    def _template_by_risk(
        risk: dict,
        remediation_tasks: list[dict],
        concern_override: str = "",
        followup_questions: list[dict] | None = None,
    ) -> tuple[str, list[str], str]:
        risk_id = str(risk.get("id") or "RISK-001")
        reason_raw = str(concern_override or risk.get("reason") or "").strip()
        reason = reason_raw.lower()
        refs = risk.get("evidence_refs", []) if isinstance(risk.get("evidence_refs"), list) else []

        claim_ids = sorted(set(re.findall(r"\bC\d+\b", reason_raw)))
        figure_tokens = []
        for ref in refs[:3]:
            if not isinstance(ref, dict):
                continue
            excerpt = str(ref.get("excerpt", ""))
            figure_tokens.extend(re.findall(r"\b(?:Figure|Fig\.?|Table|Tab\.?)\s*\d+[A-Za-z\-\.]*\b", excerpt, flags=re.IGNORECASE))
        figure_tokens = list(dict.fromkeys(figure_tokens))[:3]

        section_hints = []
        for ref in refs[:2]:
            if not isinstance(ref, dict):
                continue
            sec = str(ref.get("section", "")).strip()
            if sec and sec not in section_hints:
                section_hints.append(sec)

        claim_hint = f" for {'/'.join(claim_ids)}" if claim_ids else ""
        section_hint = f" in sections {', '.join(section_hints)}" if section_hints else ""
        response = (
            f"Thank you for this concern on {risk_id}{claim_hint}. "
            f"We agree this point needs stronger and more directly mapped evidence{section_hint}. "
            "In the revision, we will add a dedicated validation block that directly answers this concern."
        )
        evidence = [
            f"Add a targeted experiment package linked to {risk_id} with explicit claim-to-result mapping.",
            "Report complete metrics including mean/std and significance tests where applicable.",
        ]
        paper_change = "Update Experiments and Analysis sections with explicit reviewer-concern links."

        for task in remediation_tasks:
            if str(task.get("risk_id", "")).strip() != risk_id:
                continue
            title = str(task.get("title") or "").strip()
            protocol = task.get("protocol") or []
            if title:
                evidence[0] = f"Planned experiment: {title}."
            if isinstance(protocol, list) and protocol:
                evidence[1] = f"Protocol update: {str(protocol[0]).strip()}"
            break

        if "reproduc" in reason:
            response = (
                f"Thank you for pointing out reproducibility risk ({risk_id}). "
                "We will provide complete implementation/runtime details and deterministic rerun instructions."
            )
            evidence = [
                "Release complete implementation details including preprocessing, hyperparameters, and seeds.",
                "Add reproducibility checklist with deterministic rerun instructions.",
            ]
            paper_change = "Expand Implementation Details and Appendix with reproducibility checklist and rerun scripts."
        elif "significance" in reason or "statistical" in reason:
            response = (
                f"Thank you for this statistical-validity concern ({risk_id}). "
                "We will add multi-seed statistics and paired significance tests for each primary metric."
            )
            evidence = [
                "Add multi-seed mean/std with paired significance tests.",
                "Report confidence intervals and variance-sensitive metrics.",
            ]
            paper_change = "Revise Results tables and statistical analysis subsection with test setup and p-values."
        elif "baseline" in reason or "comparison" in reason:
            response = (
                f"Thank you for this baseline-comparison concern ({risk_id}). "
                "We will strengthen fairness by adding stronger baselines and matched-budget settings."
            )
            evidence = [
                "Add stronger baseline comparisons under matched data/training budgets.",
                "Include fair compute-normalized comparison settings and discussion.",
            ]
            paper_change = "Update Baselines and Experimental Setup sections with fairness constraints and added baselines."
        elif "ablation" in reason:
            response = (
                f"Thank you for highlighting ablation completeness ({risk_id}). "
                "We will add component-level and interaction ablations to isolate each design contribution."
            )
            evidence = [
                "Add one-by-one ablation table covering each component.",
                "Add interaction ablations and discuss causal contribution.",
            ]
            paper_change = "Expand Ablation section with full component matrix and interaction analysis."
        elif "citation" in reason or "related work" in reason or "top-venue" in reason:
            response = (
                f"Thank you for the related-work positioning concern ({risk_id}). "
                "We will add recent top-venue references and a clearer novelty-positioning comparison."
            )
            evidence = [
                "Add nearest-neighbor prior methods from recent top venues (last 2-3 years).",
                "Add a positioning table: prior methods vs our method on assumptions and outcomes.",
            ]
            paper_change = "Revise Related Work and Introduction novelty-positioning paragraphs."
        elif any(k in reason for k in ["contradiction", "conflict", "opposite", "inconsistent with claim"]):
            anchor_hint = ""
            if figure_tokens:
                anchor_hint = f" We will explicitly reconcile {', '.join(figure_tokens)} with revised claim wording."
            response = (
                f"Thank you for flagging this claim-result contradiction ({risk_id}). "
                "We agree the current text may overstate conclusions in some settings. "
                "We will narrow claim scope and provide a direct contradiction-resolution analysis."
                f"{anchor_hint}"
            )
            evidence = [
                "Add a contradiction matrix: claim sentence -> conflicting result anchor -> corrected interpretation.",
                "Report setting-specific win/loss analysis and clearly state where the method does not outperform baselines.",
            ]
            paper_change = (
                "Revise Abstract/Introduction claim wording and add a contradiction-resolution subsection in Experiments."
            )

        if figure_tokens:
            evidence.append(f"Directly reference existing anchors: {', '.join(figure_tokens)}.")
        if isinstance(followup_questions, list) and followup_questions:
            q = str(followup_questions[0].get("question", "")).strip()
            if q:
                evidence.append(f"Preempt likely follow-up: {q}")

        return response, evidence, paper_change

    def _precheck_and_repair(
        self,
        ctx: PipelineContext,
        review_id: str,
        concern: str,
        response: str,
        new_evidence: list[str],
        paper_change: str,
        char_limit: int,
    ) -> tuple[str, list[str], str, dict]:
        record = {
            "review_id": review_id,
            "pass": True,
            "coverage": 1.0,
            "issues": [],
            "repair_applied": False,
        }

        # 1) executor precheck (optional)
        if self.executor is not None:
            prechecked = self._precheck_with_executor(
                ctx, review_id, concern, response, new_evidence, paper_change, char_limit
            )
            if prechecked is not None:
                response = prechecked["response"]
                new_evidence = prechecked["new_evidence"]
                paper_change = prechecked["paper_change"]
                record["pass"] = bool(prechecked.get("pass", True))
                record["issues"] = list(prechecked.get("issues", []))
                record["repair_applied"] = bool(prechecked.get("repair_applied", False))

        # 2) heuristic precheck (always)
        keywords = self._extract_keywords(concern)
        target = " ".join([response, *new_evidence, paper_change]).lower()
        covered = sum(1 for k in keywords if k in target)
        coverage = covered / max(1, len(keywords))
        record["coverage"] = round(coverage, 3)

        action_present = any(
            token in target
            for token in [
                "add",
                "update",
                "report",
                "provide",
                "run",
                "include",
                "release",
                "clarify",
                "ablation",
                "baseline",
                "significance",
                "reproduc",
            ]
        )

        if coverage < 0.25 or not action_present:
            record["pass"] = False
            record["issues"].append("low_concern_coverage_or_actionability")
            repair = self._heuristic_repair(concern, response)
            if repair != response:
                response = repair
                record["repair_applied"] = True
                ctx.add_qa_issue(f"rebuttal_precheck_warning:{review_id}:auto_repair_applied")
            else:
                ctx.add_qa_issue(f"rebuttal_precheck_warning:{review_id}:coverage_low")

        (
            response,
            new_evidence,
            paper_change,
            hallucination_meta,
        ) = self._hallucination_guard(
            ctx=ctx,
            review_id=review_id,
            response=response,
            new_evidence=new_evidence,
            paper_change=paper_change,
        )
        if hallucination_meta.get("issues"):
            record["pass"] = False
            record["issues"].extend(
                [x for x in hallucination_meta.get("issues", []) if x not in record["issues"]]
            )
        record["repair_applied"] = bool(record["repair_applied"] or hallucination_meta.get("repair_applied", False))
        record["hallucination"] = hallucination_meta

        return response, new_evidence, paper_change, record

    def _precheck_with_executor(
        self,
        ctx: PipelineContext,
        review_id: str,
        concern: str,
        response: str,
        new_evidence: list[str],
        paper_change: str,
        char_limit: int,
    ) -> dict | None:
        spec = TaskSpec(
            task_type="rebuttal_precheck",
            prompt="Check whether rebuttal addresses concern. Return JSON only.",
            context={
                "review_id": review_id,
                "concern": concern,
                "response": response,
                "new_evidence": new_evidence,
                "paper_change": paper_change,
                "char_limit": char_limit,
            },
            output_schema={
                "pass": True,
                "issues": ["string"],
                "revised_response": "string|null",
                "revised_new_evidence": ["string"],
                "revised_paper_change": "string|null",
            },
            model_profile="judge",
        )
        result = self.executor.execute(spec)
        for warning in result.warnings:
            ctx.add_qa_issue(f"rebuttal_executor_warning:{warning}")
        if not result.ok:
            return None

        data = result.output
        if isinstance(data.get("response"), str):
            try:
                parsed = json.loads(data["response"])
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:  # noqa: BLE001
                return None
        if not isinstance(data, dict):
            return None

        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = []

        revised_response = data.get("revised_response")
        revised_evidence = data.get("revised_new_evidence")
        revised_change = data.get("revised_paper_change")

        return {
            "pass": bool(data.get("pass", True)),
            "issues": [str(x) for x in issues if str(x).strip()],
            "response": str(revised_response).strip() if isinstance(revised_response, str) and revised_response.strip() else response,
            "new_evidence": [str(x).strip() for x in revised_evidence if str(x).strip()] if isinstance(revised_evidence, list) and revised_evidence else new_evidence,
            "paper_change": str(revised_change).strip() if isinstance(revised_change, str) and revised_change.strip() else paper_change,
            "repair_applied": bool(
                (isinstance(revised_response, str) and revised_response.strip())
                or (isinstance(revised_change, str) and revised_change.strip())
                or (isinstance(revised_evidence, list) and len(revised_evidence) > 0)
            ),
        }

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        stop = {
            "the",
            "this",
            "that",
            "with",
            "from",
            "have",
            "has",
            "were",
            "been",
            "into",
            "your",
            "their",
            "about",
            "which",
            "would",
            "could",
            "should",
            "there",
            "appears",
            "missing",
            "details",
        }
        tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
        uniq = []
        seen = set()
        for token in tokens:
            if token in stop or token in seen:
                continue
            seen.add(token)
            uniq.append(token)
        return uniq[:8]

    @staticmethod
    def _heuristic_repair(concern: str, response: str) -> str:
        _ = concern  # concern kept for future extension
        tail = " Specifically, we will address this concern with direct evidence, stronger analysis, and explicit paper updates in revision."
        if tail.strip() in response:
            return response
        return (response.rstrip() + tail).strip()

    def _hallucination_guard(
        self,
        *,
        ctx: PipelineContext,
        review_id: str,
        response: str,
        new_evidence: list[str],
        paper_change: str,
    ) -> tuple[str, list[str], str, dict]:
        inventory = self._paper_inventory(ctx)
        issues: list[str] = []
        repair_applied = False

        joined = " ".join([response, *new_evidence, paper_change]).strip()
        mentioned_sections = sorted(self._extract_section_mentions(joined))
        missing_sections: list[str] = []
        empty_sections: list[str] = []
        known_sections = set(inventory.get("known_sections", []))
        nonempty_sections = set(inventory.get("nonempty_sections", []))

        if known_sections:
            for sec in mentioned_sections:
                if sec not in known_sections:
                    missing_sections.append(sec)
                elif sec not in nonempty_sections:
                    empty_sections.append(sec)

        if missing_sections:
            issues.extend([f"hallucinated_section:{x}" for x in missing_sections])
        if empty_sections:
            issues.extend([f"section_without_content:{x}" for x in empty_sections])

        section_replacements = {}
        fallback_sections = self._rank_fallback_sections(nonempty_sections)
        fallback_text = ", ".join(self._title_case_section(x) for x in fallback_sections) if fallback_sections else ""
        if fallback_sections:
            for sec in [*missing_sections, *empty_sections]:
                section_replacements[sec] = fallback_sections[0]

        known_anchors = set(inventory.get("known_anchors", []))
        referenced_anchors = sorted(self._extract_anchor_mentions(joined))
        unknown_anchors: list[str] = []
        unverifiable_anchors: list[str] = []
        if referenced_anchors:
            if known_anchors:
                unknown_anchors = [a for a in referenced_anchors if a not in known_anchors]
            else:
                unverifiable_anchors = list(referenced_anchors)

        if unknown_anchors:
            issues.extend([f"hallucinated_anchor:{x}" for x in unknown_anchors])
        if unverifiable_anchors:
            issues.extend([f"unverifiable_anchor:{x}" for x in unverifiable_anchors])

        if section_replacements:
            if fallback_text:
                paper_change = (
                    "Update verified non-empty sections "
                    f"({fallback_text}) with explicit reviewer-concern links and new evidence mapping."
                )
            else:
                paper_change = (
                    "Update verified non-empty sections with explicit reviewer-concern links and new evidence mapping."
                )
            repair_applied = True

        anchor_to_strip = [*unknown_anchors, *unverifiable_anchors]
        if anchor_to_strip:
            response = self._strip_unknown_anchors(response, anchor_to_strip)
            new_evidence = [self._strip_unknown_anchors(x, anchor_to_strip) for x in new_evidence]
            paper_change = self._strip_unknown_anchors(paper_change, anchor_to_strip)
            new_evidence = [
                x if len(x.strip()) >= 12 else "Add evidence anchored to verified existing result tables/figures."
                for x in new_evidence
            ]
            if not any("verified existing result tables/figures" in x for x in new_evidence):
                new_evidence.append("Map each rebuttal point to verified existing result tables/figures and section anchors.")
            repair_applied = True

        # clean empty evidence bullets after repairs
        new_evidence = [x.strip() for x in new_evidence if x and x.strip()]
        if not new_evidence:
            new_evidence = ["Add direct evidence mapped to verified section and result anchors."]
            repair_applied = True

        for issue in issues:
            ctx.add_qa_issue(f"rebuttal_hallucination_warning:{review_id}:{issue}")

        return (
            response.strip(),
            new_evidence,
            paper_change.strip(),
            {
                "issues": issues,
                "repair_applied": repair_applied,
                "mentioned_sections": mentioned_sections,
                "missing_sections": missing_sections,
                "empty_sections": empty_sections,
                "referenced_anchors": referenced_anchors,
                "unknown_anchors": unknown_anchors,
                "unverifiable_anchors": unverifiable_anchors,
                "known_sections": sorted(known_sections),
                "known_anchors": sorted(known_anchors),
            },
        )

    @staticmethod
    def _paper_inventory(ctx: PipelineContext) -> dict:
        structured = ctx.artifacts.get("paper_structured", {})
        sections = structured.get("sections", []) if isinstance(structured, dict) else []
        known_sections: set[str] = set()
        nonempty_sections: set[str] = set()

        if isinstance(sections, list):
            for sec in sections:
                if not isinstance(sec, dict):
                    continue
                name = RebuttalComposerStep._normalize_section_name(str(sec.get("name", "")))
                if not name:
                    continue
                known_sections.add(name)
                if str(sec.get("text", "")).strip():
                    nonempty_sections.add(name)

        texts = [str(structured.get("raw_text", ""))]
        evidence_index = ctx.artifacts.get("evidence_index", {})
        passages = evidence_index.get("passages", []) if isinstance(evidence_index, dict) else []
        if isinstance(passages, list):
            for p in passages:
                if isinstance(p, dict):
                    texts.append(str(p.get("text", "")))

        known_anchors: set[str] = set()
        for text in texts:
            if not text:
                continue
            known_anchors.update(RebuttalComposerStep._extract_anchor_mentions(text))

        return {
            "known_sections": sorted(known_sections),
            "nonempty_sections": sorted(nonempty_sections),
            "known_anchors": sorted(known_anchors),
        }

    @staticmethod
    def _normalize_section_name(text: str) -> str:
        t = text.strip().lower()
        if not t:
            return ""
        aliases = {
            "experiments": ["experiment", "experiments", "evaluation", "result", "results"],
            "ablation": ["ablation", "ablations"],
            "limitations": ["limitation", "limitations"],
            "analysis": ["analysis", "analyses"],
            "method": ["method", "methods", "approach", "model"],
            "appendix": ["appendix", "supplementary"],
            "introduction": ["introduction"],
            "conclusion": ["conclusion", "conclusions"],
            "related work": ["related work", "related-work"],
            "abstract": ["abstract"],
        }
        for canonical, keys in aliases.items():
            for key in keys:
                if re.search(rf"\b{re.escape(key)}\b", t):
                    return canonical
        return t

    @staticmethod
    def _title_case_section(name: str) -> str:
        if name == "related work":
            return "Related Work"
        return " ".join(part.capitalize() for part in name.split())

    @staticmethod
    def _rank_fallback_sections(nonempty_sections: set[str]) -> list[str]:
        priority = {
            "experiments": 1,
            "ablation": 2,
            "analysis": 3,
            "method": 4,
            "limitations": 5,
            "appendix": 6,
            "related work": 7,
            "introduction": 8,
            "conclusion": 9,
            "abstract": 10,
        }
        candidates = [
            s
            for s in nonempty_sections
            if s and s not in {"preamble", "body", "citation_graph"}
        ]
        candidates.sort(key=lambda x: (priority.get(x, 99), x))
        return candidates[:3]

    @staticmethod
    def _extract_section_mentions(text: str) -> set[str]:
        blob = text.lower()
        patterns = {
            "experiments": r"\b(experiments?|evaluation|results?)\b",
            "ablation": r"\bablations?\b",
            "limitations": r"\blimitations?\b",
            "analysis": r"\banalysis\b",
            "method": r"\b(method|approach|model)\b",
            "appendix": r"\b(appendix|supplementary)\b",
            "introduction": r"\bintroduction\b",
            "conclusion": r"\bconclusions?\b",
            "related work": r"\brelated[\s\-]work\b",
            "abstract": r"\babstract\b",
        }
        out = set()
        for canonical, pattern in patterns.items():
            if re.search(pattern, blob, flags=re.IGNORECASE):
                out.add(canonical)
        return out

    @staticmethod
    def _extract_anchor_mentions(text: str) -> set[str]:
        out: set[str] = set()
        for m in re.finditer(r"\b(Figure|Fig\.?|Table|Tab\.?)\s*(\d+[A-Za-z\-\.]*)\b", text, flags=re.IGNORECASE):
            prefix = m.group(1).lower()
            number = m.group(2)
            canonical = f"Figure {number}" if prefix.startswith("fig") else f"Table {number}"
            out.add(canonical)
        return out

    @staticmethod
    def _replace_section_mentions(text: str, replacements: dict[str, str]) -> str:
        out = text
        for src, dst in replacements.items():
            if src == dst:
                continue
            src_title = RebuttalComposerStep._title_case_section(src)
            dst_title = RebuttalComposerStep._title_case_section(dst)
            out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
            out = re.sub(rf"\b{re.escape(src_title)}\b", dst_title, out, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", out).strip()

    @staticmethod
    def _strip_unknown_anchors(text: str, anchors: list[str]) -> str:
        out = text
        for anchor in anchors:
            m = re.match(r"^(Figure|Table)\s+(\d+[A-Za-z\-\.]*)$", anchor, flags=re.IGNORECASE)
            if not m:
                continue
            kind = m.group(1)
            num = m.group(2)
            if kind.lower().startswith("figure"):
                pattern = rf"\b(?:Figure|Fig\.?)\s*{re.escape(num)}\b"
            else:
                pattern = rf"\b(?:Table|Tab\.?)\s*{re.escape(num)}\b"
            out = re.sub(pattern, "verified existing result tables/figures", out, flags=re.IGNORECASE)

        out = re.sub(r"(verified existing result tables/figures)(\s*,\s*verified existing result tables/figures)+", r"\1", out)
        out = re.sub(r"\s{2,}", " ", out)
        out = re.sub(r"\s+([,.;:])", r"\1", out)
        return out.strip()

    def _compose_global_response(self, ctx: PipelineContext, items: list[RebuttalItem]) -> str:
        if self.executor is None:
            return (
                "We appreciate all reviews and will strengthen evidence quality, add statistical significance "
                "reporting, and clarify contribution boundaries in the revision."
            )

        concerns = [x.concern for x in items]
        spec = TaskSpec(
            task_type="rebuttal_global",
            prompt="Compose one concise global rebuttal summary. Return JSON only.",
            context={
                "venue": ctx.input_data.venue.name,
                "year": ctx.input_data.venue.year,
                "concerns": concerns,
                "char_limit": 1200,
            },
            output_schema={"global_response": "string"},
            model_profile="judge",
        )
        result = self.executor.execute(spec)
        for warning in result.warnings:
            ctx.add_qa_issue(f"rebuttal_executor_warning:{warning}")
        if not result.ok:
            return (
                "We appreciate all reviews and will strengthen evidence quality, add statistical significance "
                "reporting, and clarify contribution boundaries in the revision."
            )

        data = result.output
        if isinstance(data.get("response"), str):
            try:
                parsed = json.loads(data["response"])
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:  # noqa: BLE001
                pass

        value = str(data.get("global_response") or "").strip()
        if not value:
            return (
                "We appreciate all reviews and will strengthen evidence quality, add statistical significance "
                "reporting, and clarify contribution boundaries in the revision."
            )
        return value[:1200]

    @staticmethod
    def _compose_block(concern: str, response: str, new_evidence: list[str], paper_change: str) -> str:
        return (
            f"Concern: {concern}\n"
            f"Response: {response}\n"
            f"New evidence: {'; '.join(new_evidence)}\n"
            f"Paper change: {paper_change}"
        )

    @staticmethod
    def _shrink_response(response: str, overflow: int) -> str:
        if overflow <= 0:
            return response
        if overflow >= len(response):
            return response[: max(0, len(response) // 3)].rstrip()
        return response[:-overflow].rstrip()

    def _translate_bundle(self, bundle: RebuttalBundle) -> RebuttalBundle:
        translated_items = []
        for item in bundle.items:
            translated_items.append(
                RebuttalItem(
                    review_id=item.review_id,
                    concern=self.translator.to_zh(item.concern),
                    response=self.translator.to_zh(item.response),
                    new_evidence=[self.translator.to_zh(x) for x in item.new_evidence],
                    paper_change=self.translator.to_zh(item.paper_change),
                    char_count=item.char_count,
                    char_limit=item.char_limit,
                )
            )

        return RebuttalBundle(
            venue=bundle.venue,
            year=bundle.year,
            manuscript_stage=bundle.manuscript_stage,
            mode=bundle.mode,
            items=translated_items,
            global_response=self.translator.to_zh(bundle.global_response) if bundle.global_response else None,
            attachment_pdf=bundle.attachment_pdf,
        )

    @staticmethod
    def _to_markdown(bundle: RebuttalBundle) -> str:
        lines = [
            f"# Rebuttal Draft ({bundle.venue} {bundle.year})",
            "",
            f"Manuscript Stage: `{bundle.manuscript_stage}`",
            "",
        ]
        if bundle.global_response:
            lines.extend(["## Global Positioning", "", bundle.global_response, ""])

        for item in bundle.items:
            lines.extend(
                [
                    f"## Response to Reviewer {item.review_id}",
                    "",
                    "### Concern",
                    "",
                    item.concern,
                    "",
                    "### Response",
                    "",
                    item.response,
                    "",
                    "### New Evidence",
                    "",
                    *[f"- {x}" for x in item.new_evidence],
                    "",
                    "### Paper Changes",
                    "",
                    item.paper_change,
                    "",
                    f"Character Budget: {item.char_count} / {item.char_limit}",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _to_markdown_zh(bundle: RebuttalBundle) -> str:
        lines = [
            f"# Rebuttal 草稿（{bundle.venue} {bundle.year}）",
            "",
            f"稿件阶段：`{bundle.manuscript_stage}`",
            "",
        ]
        if bundle.global_response:
            lines.extend(["## 总体回应", "", bundle.global_response, ""])

        for item in bundle.items:
            lines.extend(
                [
                    f"## 对审稿人 {item.review_id} 的回复",
                    "",
                    "### 审稿意见",
                    "",
                    item.concern,
                    "",
                    "### 回应",
                    "",
                    item.response,
                    "",
                    "### 新增证据",
                    "",
                    *[f"- {x}" for x in item.new_evidence],
                    "",
                    "### 论文修改位置",
                    "",
                    item.paper_change,
                    "",
                    f"字符预算：{item.char_count} / {item.char_limit}",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

