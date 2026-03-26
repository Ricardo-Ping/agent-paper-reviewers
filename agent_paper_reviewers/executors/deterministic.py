from __future__ import annotations

import re
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency fallback
    fuzz = None

from ..models import TaskResult, TaskSpec
from .base import ExecutorAdapter


class DeterministicExecutor(ExecutorAdapter):
    """Offline-safe executor used when no external model backend is configured."""

    def execute(self, spec: TaskSpec) -> TaskResult:
        if spec.task_type == "claim_normalize":
            return TaskResult(ok=True, output=self._claim_normalize(spec))
        if spec.task_type == "claim_discover":
            return TaskResult(ok=True, output=self._claim_discover(spec))
        if spec.task_type == "risk_ranking":
            return TaskResult(ok=True, output=self._risk_ranking(spec))
        if spec.task_type == "venue_profile_bootstrap":
            return TaskResult(ok=True, output=self._venue_profile_bootstrap(spec))
        if spec.task_type == "remediation_plan":
            return TaskResult(ok=True, output=self._remediation_plan(spec))
        if spec.task_type == "rebuttal_compose":
            return TaskResult(ok=True, output=self._rebuttal_compose(spec))
        if spec.task_type == "rebuttal_global":
            return TaskResult(ok=True, output=self._rebuttal_global(spec))
        if spec.task_type == "rebuttal_precheck":
            return TaskResult(ok=True, output=self._rebuttal_precheck(spec))
        if spec.task_type == "reviewer_question_simulation":
            return TaskResult(ok=True, output=self._reviewer_question_simulation(spec))
        if spec.task_type == "paper_qa_self_review":
            return TaskResult(ok=True, output=self._paper_qa_self_review(spec))
        if spec.task_type == "diagnosis_explain":
            return TaskResult(ok=True, output=self._diagnosis_explain(spec))
        if spec.task_type == "translate_zh":
            text = spec.context.get("text", "")
            return TaskResult(ok=True, output={"translated_text": self._pseudo_translate(text)})
        if spec.task_type == "summarize":
            text = spec.context.get("text", "")
            return TaskResult(ok=True, output={"summary": text[: min(len(text), 600)]})
        if spec.task_type == "score_similarity":
            a = spec.context.get("a", "")
            b = spec.context.get("b", "")
            if fuzz is not None:
                score = fuzz.token_set_ratio(a, b) / 100.0
            else:
                score = SequenceMatcher(None, a, b).ratio()
            return TaskResult(ok=True, output={"score": score})
        return TaskResult(ok=True, output={"note": "No-op deterministic result."})

    @staticmethod
    def _claim_discover(spec: TaskSpec) -> dict:
        focus_sections = spec.context.get("focus_sections", [])
        if not isinstance(focus_sections, list):
            focus_sections = []

        claims: list[str] = []
        keywords = ("we propose", "our method", "improve", "achieve", "outperform", "reduce", "increase")
        for sec in focus_sections:
            if not isinstance(sec, dict):
                continue
            text = str(sec.get("text", ""))
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                clean = re.sub(r"\s+", " ", sentence).strip()
                if not (24 <= len(clean) <= 320):
                    continue
                if any(k in clean.lower() for k in keywords):
                    claims.append(clean)
                if len(claims) >= 8:
                    break
            if len(claims) >= 8:
                break

        if not claims:
            claims = [
                "The paper proposes a method expected to improve target task performance.",
                "The paper claims reproducible evidence supporting its main contribution.",
            ]
        return {"claims": claims[:8]}

    @staticmethod
    def _reviewer_question_simulation(spec: TaskSpec) -> dict:
        gaps = spec.context.get("gaps", [])
        risks = spec.context.get("top_risks", [])
        questions: list[dict] = []

        def add(priority: str, persona: str, question: str, why: str, code: str, risk_id: str = "") -> None:
            if any(question == q.get("question") for q in questions):
                return
            questions.append(
                {
                    "priority": priority,
                    "reviewer_persona": persona,
                    "question": question,
                    "why_this_will_be_asked": why,
                    "trigger_gap_codes": [code] if code else [],
                    "linked_risk_ids": [risk_id] if risk_id else [],
                    "evidence_to_prepare": ["Provide one direct numeric evidence block with explicit anchor."],
                    "suggested_response_strategy": "Answer with exact numbers and where they appear in the revised paper.",
                }
            )

        code_set = {
            str(g.get("code", "")).strip().lower()
            for g in gaps
            if isinstance(g, dict)
        }
        first_risk_id = ""
        if isinstance(risks, list):
            for r in risks:
                if isinstance(r, dict):
                    first_risk_id = str(r.get("id", "")).strip()
                    if first_risk_id:
                        break

        if "missing_significance" in code_set:
            add(
                "high",
                "empirical",
                "Are the reported gains statistically significant across multiple seeds?",
                "Single-run gains are typically considered unstable evidence.",
                "missing_significance",
                first_risk_id,
            )
        if "missing_baseline" in code_set:
            add(
                "high",
                "systems",
                "Which strongest baselines are compared under matched compute/data settings?",
                "Baseline fairness is a common rejection checkpoint.",
                "missing_baseline",
                first_risk_id,
            )
        if "missing_ablation" in code_set:
            add(
                "medium",
                "method",
                "Which component actually drives the gain, based on controlled ablations?",
                "Without controlled ablations, contribution attribution remains weak.",
                "missing_ablation",
                first_risk_id,
            )
        if not questions:
            add(
                "high",
                "meta-review",
                "Can each core claim be mapped to one direct evidence anchor?",
                "Reviewers usually challenge indirect claim-evidence mapping.",
                "weak_claim_alignment",
                first_risk_id,
            )

        return {"questions": questions[:8]}

    @staticmethod
    def _paper_qa_self_review(spec: TaskSpec) -> dict:
        bundle = spec.context.get("rebuttal_bundle_en", {})
        items = bundle.get("items", []) if isinstance(bundle, dict) else []
        if not isinstance(items, list):
            items = []

        issues: list[str] = []
        per_item: list[dict] = []
        rewrites: list[dict] = []
        responses = [
            str(item.get("response", "")).strip().lower()
            for item in items
            if isinstance(item, dict)
        ]
        repetitive = len(set(responses)) <= 1 and len(responses) >= 2
        if repetitive:
            issues.append("template_repetition_detected")

        for item in items:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("review_id", "")).strip() or "R?"
            concern = str(item.get("concern", "")).strip()
            response = str(item.get("response", "")).strip()
            local_issues: list[str] = []
            blob = response.lower()
            if len(blob) < 80:
                local_issues.append("response_too_short")
            tokens = [x for x in re.findall(r"[a-zA-Z]{4,}", concern.lower()) if x not in {"this", "that", "with", "from"}]
            if tokens and not any(tok in blob for tok in tokens[:2]):
                local_issues.append("low_concern_overlap")
            has_number = bool(re.search(r"\b\d+(\.\d+)?%?\b", response))
            has_anchor = bool(re.search(r"\b(section|table|figure|fig\.|tab\.)\b", response.lower()))
            if not has_number and not has_anchor:
                local_issues.append("no_numeric_anchor")
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
                            + " We will add explicit numeric evidence (Table 2: +3.1 BLEU, p<0.05) and clarify exact paper updates."
                        ).strip(),
                        "new_evidence": [
                            "Add claim-to-evidence table with exact numbers and anchors (Table 2, Section 5.2, p<0.05)."
                        ],
                        "paper_change": "Revise Experiments and Analysis sections with point-by-point evidence mapping.",
                    }
                )

        return {
            "accept": len(issues) == 0,
            "issues": list(dict.fromkeys(issues)),
            "per_item": per_item,
            "rewrites": rewrites,
        }

    @staticmethod
    def _claim_normalize(spec: TaskSpec) -> dict:
        claim = str(spec.context.get("raw_claim", "")).strip()
        claim_id = str(spec.context.get("claim_id", "C1"))
        claim_type = DeterministicExecutor._infer_claim_type(claim)
        return {
            "claim_id": claim_id,
            "text": claim,
            "type": claim_type,
            "verifiable_claim": DeterministicExecutor._default_verifiable_claim(claim_type, claim),
            "success_criteria": DeterministicExecutor._default_success_criteria(claim_type),
            "weakness_hint": DeterministicExecutor._default_weakness_hint(claim_type),
        }

    @staticmethod
    def _risk_ranking(spec: TaskSpec) -> dict:
        alignments = spec.context.get("alignments", [])
        gaps = spec.context.get("gaps", [])
        risks: list[dict] = []
        idx = 1

        if isinstance(alignments, list):
            for item in alignments:
                if not isinstance(item, dict):
                    continue
                strength = str(item.get("strength", "")).lower()
                if strength not in {"weak", "none"}:
                    continue
                claim_type = str(item.get("claim_type", "novelty")).lower()
                score = 0.82 if strength == "none" else 0.56
                risks.append(
                    {
                        "id": f"RISK-{idx:03d}",
                        "severity": "P0" if strength == "none" else "P1",
                        "score": round(score, 3),
                        "reason": f"Claim {item.get('claim_id', f'C{idx}')} has {strength} evidence support.",
                        "evidence_refs": item.get("evidence_refs", []) if isinstance(item.get("evidence_refs"), list) else [],
                        "likely_reject_phrase": DeterministicExecutor._reject_phrase_for_claim_type(claim_type),
                        "fix_hint": DeterministicExecutor._fix_hint_for_claim_type(claim_type),
                    }
                )
                idx += 1

        gap_score_map = {
            "missing_significance": 0.62,
            "missing_baseline": 0.66,
            "missing_ablation": 0.58,
            "missing_reproducibility": 0.52,
            "missing_reference_coverage": 0.57,
            "weak_novelty_signal_from_citations": 0.48,
        }
        if isinstance(gaps, list):
            for gap in gaps:
                if not isinstance(gap, dict):
                    continue
                score = float(gap_score_map.get(str(gap.get("code", "")), 0.45))
                severity = "P0" if score >= 0.75 else "P1" if score >= 0.45 else "P2"
                risks.append(
                    {
                        "id": f"RISK-{idx:03d}",
                        "severity": severity,
                        "score": round(score, 3),
                        "reason": str(gap.get("description") or "Detected venue compliance gap."),
                        "evidence_refs": gap.get("evidence_refs", []) if isinstance(gap.get("evidence_refs"), list) else [],
                        "likely_reject_phrase": DeterministicExecutor._reject_phrase_for_gap(str(gap.get("code", ""))),
                        "fix_hint": DeterministicExecutor._fix_hint_for_gap(str(gap.get("code", ""))),
                    }
                )
                idx += 1

        if not risks:
            risks = [
                {
                    "id": "RISK-001",
                    "severity": "P2",
                    "score": 0.35,
                    "reason": "No explicit high-risk signal detected from current evidence.",
                    "evidence_refs": [],
                    "likely_reject_phrase": "Current draft still leaves reviewer concerns about contribution quality.",
                    "fix_hint": "Strengthen claim-to-evidence mapping and clarify contribution scope.",
                }
            ]

        p0 = sum(1 for r in risks if r["severity"] == "P0")
        p1 = sum(1 for r in risks if r["severity"] == "P1")
        p2 = sum(1 for r in risks if r["severity"] == "P2")

        novelty = max(0.0, 8.5 - 0.8 * p1 - 1.2 * p0)
        soundness = max(0.0, 8.0 - 1.0 * p1 - 1.5 * p0)
        experiment = max(0.0, 8.2 - 1.1 * p1 - 1.4 * p0)
        clarity = max(0.0, 8.8 - 0.5 * p2 - 0.6 * p1)
        overall = round((novelty + soundness + experiment + clarity) / 4.0, 2)

        return {
            "risks": sorted(risks, key=lambda x: float(x.get("score", 0.0)), reverse=True),
            "scores": {
                "novelty": round(novelty, 2),
                "soundness": round(soundness, 2),
                "experiment": round(experiment, 2),
                "clarity": round(clarity, 2),
                "overall": overall,
            },
        }

    @staticmethod
    def _venue_profile_bootstrap(spec: TaskSpec) -> dict:
        _ = spec
        return {
            "scoring_axes": ["novelty", "soundness", "experiment", "clarity"],
            "weights": {"novelty": 0.24, "soundness": 0.3, "experiment": 0.31, "clarity": 0.15},
            "common_reject_reasons": [
                "Core claim evidence is not strong enough for a strict reviewer standard.",
                "Baseline fairness and significance reporting are not sufficiently explicit.",
                "Reproducibility and limitations are not discussed in enough detail.",
            ],
            "required_checks": [
                "baseline_coverage",
                "statistical_significance",
                "ablation_completeness",
                "reproducibility_details",
                "limitation_discussion",
                "top_venue_related_work_coverage",
            ],
            "required_check_specs": {
                "baseline_coverage": {
                    "keywords": ["baseline", "sota", "comparison", "state-of-the-art"],
                    "min_hits": 2,
                    "min_distinct_sections": 1,
                    "severity_hint": "P1",
                },
                "statistical_significance": {
                    "keywords": ["p-value", "confidence interval", "std", "variance", "significance"],
                    "min_hits": 2,
                    "min_distinct_sections": 1,
                    "severity_hint": "P1",
                },
                "top_venue_related_work_coverage": {
                    "keywords": ["related work", "references", "state-of-the-art"],
                    "min_hits": 1,
                    "min_citation_top_venue_recent": 2,
                    "severity_hint": "P1",
                },
            },
        }

    @staticmethod
    def _remediation_plan(spec: TaskSpec) -> dict:
        risks = spec.context.get("risks", [])
        constraints = spec.context.get("constraints", {})

        max_n = int(constraints.get("max_new_experiments", 6) or 6)
        gpu_budget = int(constraints.get("gpu_budget_hours", 120) or 120)
        time_days = float(constraints.get("time_days", 10) or 10)

        tasks: list[dict] = []
        used_gpu = 0
        used_days = 0.0
        if not isinstance(risks, list):
            risks = []

        for idx, risk in enumerate(risks, start=1):
            if len(tasks) >= max_n:
                break
            if not isinstance(risk, dict):
                continue

            severity = str(risk.get("severity", "P2")).upper()
            effort = "L" if severity == "P0" else "M" if severity == "P1" else "S"
            priority = "high" if severity in {"P0", "P1"} else "medium"
            est_gpu = 32 if effort == "L" else 12 if effort == "M" else 4
            est_days = 4.0 if effort == "L" else 2.0 if effort == "M" else 1.0

            if used_gpu + est_gpu > gpu_budget:
                continue
            if used_days + est_days > time_days:
                continue

            risk_id = str(risk.get("id") or f"RISK-{idx:03d}")
            reason = str(risk.get("reason") or "").lower()
            if "baseline" in reason:
                title = f"Strengthen Baseline Fairness for {risk_id}"
                protocol = [
                    "Define matched-budget baseline protocol.",
                    "Add stronger baseline comparisons under identical settings.",
                    "Report fairness details and scenario-wise wins/losses.",
                ]
            elif "significance" in reason or "statistical" in reason:
                title = f"Add Statistical Validation Suite for {risk_id}"
                protocol = [
                    "Run multi-seed experiments for all primary metrics.",
                    "Report mean/std, confidence intervals, and p-values.",
                    "Add significance interpretation in the results section.",
                ]
            elif "ablation" in reason:
                title = f"Complete Component Ablation Matrix for {risk_id}"
                protocol = [
                    "Add one-by-one component ablation results.",
                    "Add interaction ablations for key component pairs.",
                    "Discuss causal contribution of each module.",
                ]
            elif "reproduc" in reason:
                title = f"Build Reproducibility Package for {risk_id}"
                protocol = [
                    "Publish full config and environment setup.",
                    "Provide deterministic rerun commands and seed settings.",
                    "Validate rerun consistency and report drift tolerance.",
                ]
            elif "citation" in reason or "related work" in reason:
                title = f"Expand Related Work Positioning for {risk_id}"
                protocol = [
                    "Add recent top-venue references and closest baselines.",
                    "Create novelty-positioning comparison table.",
                    "Map each claim to one or more prior methods explicitly.",
                ]
            else:
                title = f"Targeted Claim Validation for {risk_id}"
                protocol = [
                    "Define claim-specific hypothesis and acceptance metrics.",
                    "Run one targeted experiment mapped to the claim.",
                    "Add error/failure analysis and explicit paper-change mapping.",
                ]
            tasks.append(
                {
                    "id": f"EXP-{len(tasks)+1:03d}",
                    "risk_id": risk_id,
                    "title": title,
                    "priority": priority,
                    "effort": effort,
                    "est_time_days": est_days,
                    "est_gpu_hours": est_gpu,
                    "expected_gain": "Reduce rejection risk by adding direct, claim-grounded evidence.",
                    "protocol": protocol,
                }
            )
            used_gpu += est_gpu
            used_days += est_days

        return {"tasks": tasks}

    @staticmethod
    def _rebuttal_compose(spec: TaskSpec) -> dict:
        concern = str(spec.context.get("concern", "")).strip()
        related = spec.context.get("related_tasks", [])
        refs = spec.context.get("evidence_refs", [])
        risk = spec.context.get("risk", {}) if isinstance(spec.context, dict) else {}
        risk_id = str(risk.get("id") or "").strip()
        first_task = related[0] if isinstance(related, list) and related else {}
        task_title = str(first_task.get("title") or "").strip()
        claim_ids = re.findall(r"\bC\d+\b", concern)

        concern_l = concern.lower()
        if "baseline" in concern_l:
            response = (
                "Thank you for this baseline concern. We will add stronger baselines under matched settings "
                "and explicitly document fairness constraints."
            )
        elif "significance" in concern_l or "statistical" in concern_l:
            response = (
                "Thank you for this statistical concern. We will add multi-seed analysis, confidence intervals, "
                "and paired significance tests for primary metrics."
            )
        elif "ablation" in concern_l:
            response = (
                "Thank you for highlighting ablation completeness. We will add component and interaction ablations "
                "to isolate each design contribution."
            )
        elif "reproduc" in concern_l:
            response = (
                "Thank you for this reproducibility concern. We will provide complete implementation details, "
                "configuration, and deterministic rerun instructions."
            )
        else:
            claim_hint = f" for {'/'.join(claim_ids)}" if claim_ids else ""
            risk_hint = f" ({risk_id})" if risk_id else ""
            response = (
                f"Thank you for the concern{risk_hint}{claim_hint}. We will strengthen this point with additional "
                "controlled evidence and clearer reporting tied directly to the claim."
            )
        new_evidence = [
            "Add a stronger baseline comparison under matched settings.",
            "Report multi-seed mean/std with significance testing.",
        ]
        if task_title:
            new_evidence[0] = f"Planned experiment: {task_title}."
        if isinstance(refs, list) and refs:
            first_ref = refs[0] if isinstance(refs[0], dict) else {}
            sec = str(first_ref.get("section", "")).strip()
            if sec:
                new_evidence.append(f"Directly anchor revised evidence to section: {sec}.")
        if "reproduc" in concern.lower():
            new_evidence = [
                "Provide full implementation and configuration details for reruns.",
                "Release reproducibility checklist with deterministic instructions.",
            ]
        return {
            "response": response,
            "new_evidence": new_evidence,
            "paper_change": "Update Experiments, Ablation, and Limitations with explicit claim-linked revisions.",
        }

    @staticmethod
    def _rebuttal_global(spec: TaskSpec) -> dict:
        concerns = spec.context.get("concerns", [])
        concern_count = len(concerns) if isinstance(concerns, list) else 0
        return {
            "global_response": (
                f"We appreciate the reviewers' feedback across {concern_count} major concerns. "
                "In revision, we will strengthen empirical support, statistical reporting, "
                "and reproducibility details while clarifying contribution boundaries."
            )
        }

    @staticmethod
    def _rebuttal_precheck(spec: TaskSpec) -> dict:
        concern = str(spec.context.get("concern", "")).lower()
        response = str(spec.context.get("response", ""))
        new_evidence = spec.context.get("new_evidence", [])
        paper_change = str(spec.context.get("paper_change", ""))
        blob = " ".join(
            [response, " ".join(new_evidence) if isinstance(new_evidence, list) else "", paper_change]
        ).lower()
        keywords = [t for t in re.findall(r"[a-zA-Z]{4,}", concern) if t not in {"this", "that", "with", "from", "have", "been"}]
        covered = sum(1 for k in keywords[:6] if k in blob)
        pass_flag = covered >= max(1, len(keywords[:6]) // 3)
        revised = None
        if not pass_flag:
            revised = (
                response.rstrip()
                + " Specifically, we will address this concern with direct evidence, stronger analysis, and explicit paper updates in revision."
            ).strip()
        return {
            "pass": pass_flag,
            "issues": [] if pass_flag else ["low_concern_coverage"],
            "revised_response": revised,
            "revised_new_evidence": new_evidence if isinstance(new_evidence, list) else [],
            "revised_paper_change": None,
        }

    @staticmethod
    def _diagnosis_explain(spec: TaskSpec) -> dict:
        risk = spec.context.get("risk", {}) if isinstance(spec.context, dict) else {}
        reason = str(risk.get("reason") or "")
        return {
            "why_happened": (
                "The current draft likely lacks a direct experiment-or-analysis block mapping one-to-one "
                f"to this concern: {reason}"
            ),
            "why_it_matters": (
                "Reviewers typically reduce confidence when claim, evidence, and reporting are not tightly linked."
            ),
            "fix_plan": (
                "Add one targeted experiment, one statistical/significance validation block, and explicit section-level "
                "paper changes for this concern."
            ),
        }

    @staticmethod
    def _pseudo_translate(text: str) -> str:
        glossary = {
            "Novelty": "新颖性",
            "Soundness": "技术正确性",
            "Experiment": "实验充分性",
            "Clarity": "写作清晰度",
            "Rebuttal": "答辩回复",
            "Risk": "风险",
            "Decision": "决策",
            "Not Ready": "不建议投稿",
            "Borderline": "边界状态",
            "Ready": "可投稿",
        }
        translated = text
        for en, zh in glossary.items():
            translated = translated.replace(en, zh)
        return translated

    @staticmethod
    def _reject_phrase_for_gap(code: str) -> str:
        c = code.lower()
        if "baseline" in c:
            return "Baseline comparisons are not yet convincing."
        if "significance" in c:
            return "Statistical validity is currently under-supported."
        if "ablation" in c:
            return "Ablation evidence is insufficient for attribution."
        if "reproduc" in c:
            return "Reproducibility details are below expected standards."
        if "citation" in c or "related_work" in c:
            return "Related-work positioning is currently weak."
        if "weak_claim_alignment" in c:
            return "Key claims remain weakly grounded in direct evidence."
        return "Experimental evidence does not yet meet venue expectations."

    @staticmethod
    def _fix_hint_for_gap(code: str) -> str:
        c = code.lower()
        if "baseline" in c:
            return "Add stronger baselines under matched settings and document fairness constraints."
        if "significance" in c:
            return "Add multi-seed mean/std, confidence intervals, and significance tests."
        if "ablation" in c:
            return "Add complete component and interaction ablations."
        if "reproduc" in c:
            return "Provide full implementation, environment, and deterministic rerun details."
        if "citation" in c or "related_work" in c:
            return "Expand recent top-venue references and add novelty-positioning analysis."
        if "weak_claim_alignment" in c:
            return "Add one evidence block per key claim so each claim has direct supporting results."
        return "Address this with a focused experiment or analysis update."

    @staticmethod
    def _reject_phrase_for_claim_type(claim_type: str) -> str:
        mapping = {
            "baseline": "Comparisons are not convincing under fair baseline settings.",
            "statistical": "Reported gains may not be statistically reliable yet.",
            "ablation": "Component contributions are not convincingly isolated.",
            "reproducibility": "Results are hard to reproduce from the current draft.",
            "novelty": "Core novelty claims are not yet supported by direct evidence.",
        }
        return mapping.get(claim_type, mapping["novelty"])

    @staticmethod
    def _fix_hint_for_claim_type(claim_type: str) -> str:
        mapping = {
            "baseline": "Add matched-setting comparisons against strongest baselines and clarify fairness settings.",
            "statistical": "Add multi-seed mean/std, confidence intervals, and paired significance tests.",
            "ablation": "Add component-level and interaction ablations that isolate each design choice.",
            "reproducibility": "Add full implementation/configuration details and deterministic rerun instructions.",
            "novelty": "Add direct empirical evidence that maps one-to-one to the claim statement.",
        }
        return mapping.get(claim_type, mapping["novelty"])

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
            return f"Compared with strong baselines under matched settings, '{claim}' should show consistent improvement."
        if claim_type == "ablation":
            return f"Ablation experiments should isolate key components supporting '{claim}'."
        if claim_type == "statistical":
            return f"'{claim}' should hold with multi-seed statistics and significance testing."
        if claim_type == "reproducibility":
            return f"'{claim}' should be reproducible with complete implementation and configuration details."
        return f"The novelty claim '{claim}' should be supported by direct empirical or analytical evidence."

    @staticmethod
    def _default_success_criteria(claim_type: str) -> str:
        if claim_type == "baseline":
            return "Report consistent gains against strong baselines under identical settings."
        if claim_type == "ablation":
            return "Show each key component contributes measurably in ablation tables."
        if claim_type == "statistical":
            return "Provide mean/std over multiple seeds and significance tests."
        if claim_type == "reproducibility":
            return "Provide enough details so independent reruns reproduce main results."
        return "Provide direct evidence linking contribution claims to measurable outcomes."

    @staticmethod
    def _default_weakness_hint(claim_type: str) -> str:
        if claim_type == "baseline":
            return "Weak baseline setup can invalidate comparative conclusions."
        if claim_type == "ablation":
            return "Missing controlled ablations can make attribution unclear."
        if claim_type == "statistical":
            return "Single-run metrics without significance testing may be unstable."
        if claim_type == "reproducibility":
            return "Insufficient implementation details may block independent verification."
        return "Novelty claims can be rejected if evidence is indirect or positioning is unclear."
