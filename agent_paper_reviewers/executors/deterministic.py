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
        if spec.task_type == "gap_detection_agent":
            return TaskResult(ok=True, output=self._gap_detection_agent(spec))
        if spec.task_type == "venue_recommend":
            return TaskResult(ok=True, output=self._venue_recommend(spec))
        if spec.task_type == "venue_recommend_refine":
            return TaskResult(ok=True, output=self._venue_recommend_refine(spec))
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
        if spec.task_type == "diagnosis_deep_dive":
            return TaskResult(ok=True, output=self._diagnosis_deep_dive(spec))
        if spec.task_type == "diagnosis_explain":
            return TaskResult(ok=True, output=self._diagnosis_explain(spec))
        if spec.task_type == "student_pack_generate":
            return TaskResult(ok=True, output=self._student_pack_generate(spec))
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
    def _gap_detection_agent(spec: TaskSpec) -> dict:
        context = spec.context if isinstance(spec.context, dict) else {}
        rule_gaps = context.get("rule_gaps", [])
        if not isinstance(rule_gaps, list):
            rule_gaps = []
        gaps: list[dict] = []
        for row in rule_gaps[:5]:
            if not isinstance(row, dict):
                continue
            code = str(row.get("code", "")).strip().lower() or "missing_claim_support"
            severity = str(row.get("severity_hint", "P1")).strip().upper()
            if severity not in {"P0", "P1", "P2"}:
                severity = "P1"
            desc = str(row.get("description", "")).strip() or "Evidence support is insufficient for at least one core claim."
            refs = row.get("evidence_refs", [])
            pids: list[str] = []
            if isinstance(refs, list):
                for ref in refs[:3]:
                    if not isinstance(ref, dict):
                        continue
                    pid = str(ref.get("passage_id", "")).strip()
                    if pid:
                        pids.append(pid)
            fix = "Add one direct evidence block with explicit section/table anchor for this gap."
            if "significance" in code:
                fix = "Report mean/std across seeds and add paired significance tests for primary metrics."
            elif "baseline" in code:
                fix = "Add strongest matched-setting baselines and clarify fairness constraints."
            elif "ablation" in code:
                fix = "Add component-level ablation matrix and interaction ablations."
            elif "reproduc" in code:
                fix = "Add deterministic rerun details, seeds, environment, and hyperparameters."
            gaps.append(
                {
                    "code": code,
                    "severity_hint": severity,
                    "specific_description": desc,
                    "evidence_passage_ids": pids,
                    "fix_action": fix,
                }
            )
        if not gaps:
            gaps = [
                {
                    "code": "missing_claim_support",
                    "severity_hint": "P1",
                    "specific_description": "At least one core claim still lacks direct evidence mapping.",
                    "evidence_passage_ids": [],
                    "fix_action": "Create claim -> evidence -> paper-change table and fill one direct evidence per core claim.",
                }
            ]
        return {"gaps": gaps}

    @staticmethod
    def _venue_recommend(spec: TaskSpec) -> dict:
        context = spec.context if isinstance(spec.context, dict) else {}
        candidates = context.get("candidate_venues", [])
        if not isinstance(candidates, list):
            candidates = []

        rows: list[dict] = []
        for row in candidates[:5]:
            if not isinstance(row, dict):
                continue
            venue = str(row.get("venue", "")).strip().lower()
            year = int(row.get("year", 0) or 0)
            try:
                score = float(row.get("match_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            score = max(0.0, min(1.0, score))
            failed = row.get("failed_checks", [])
            if not isinstance(failed, list):
                failed = []
            gap_text = ", ".join(str(x) for x in failed[:2]) if failed else "claim-evidence alignment"
            reasons = list(row.get("reasons", [])) if isinstance(row.get("reasons", []), list) else []
            reasons.extend(
                [
                    "Ranking combines topic fit and venue-required check readiness.",
                    f"Main pre-submit weakness for this venue: {gap_text}.",
                ]
            )
            rows.append(
                {
                    "venue": venue,
                    "year": year,
                    "match_score": round(score, 3),
                    "reasons": list(dict.fromkeys([str(x).strip() for x in reasons if str(x).strip()]))[:8],
                    "fit_summary": "Current fit is viable after direct closure of top reject drivers.",
                    "specific_gap_summary": f"Prioritize fixing {gap_text}.",
                    "required_check_passed_count": int(row.get("required_check_passed_count", 0) or 0),
                    "required_check_total": int(row.get("required_check_total", 0) or 0),
                    "passed_checks": row.get("passed_checks", []) if isinstance(row.get("passed_checks", []), list) else [],
                    "failed_checks": failed,
                }
            )
        return {
            "recommended_venues": rows,
            "method_note": "deterministic recommendation derived from rule candidates with semantic-style reasons.",
        }

    @staticmethod
    def _venue_recommend_refine(spec: TaskSpec) -> dict:
        context = spec.context if isinstance(spec.context, dict) else {}
        rows = context.get("recommended_venues", [])
        if not isinstance(rows, list):
            rows = []
        overrides: list[dict] = []
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            venue = str(row.get("venue", "")).strip().lower()
            year = int(row.get("year", 0) or 0)
            failed = row.get("failed_checks", [])
            if not isinstance(failed, list):
                failed = []
            gap_text = ", ".join(str(x) for x in failed[:2]) if failed else "claim-evidence alignment"
            overrides.append(
                {
                    "venue": venue,
                    "year": year,
                    "match_score_adjust": 0.0,
                    "reasons": [
                        f"Most critical venue-specific gap to close first: {gap_text}.",
                        "Recommendation is based on claim-topic overlap plus required-check readiness, not only keyword match.",
                    ],
                    "specific_gap_summary": f"Close {gap_text} before submission to improve acceptance odds.",
                    "fit_summary": "Current fit is feasible if top reject drivers are fixed with direct evidence.",
                }
            )
        return {"reason_overrides": overrides}
    @staticmethod
    def _student_pack_generate(spec: TaskSpec) -> dict:
        context = spec.context if isinstance(spec.context, dict) else {}
        decision = context.get("decision_json", {}) if isinstance(context.get("decision_json", {}), dict) else {}
        diagnosis = context.get("diagnosis_json", {}) if isinstance(context.get("diagnosis_json", {}), dict) else {}
        rebuttal = context.get("rebuttal_bundle_en", {}) if isinstance(context.get("rebuttal_bundle_en", {}), dict) else {}

        decision_text = str(decision.get("decision", "Not Ready"))
        meaning = str(decision.get("decision_interpretation", "Please fix top blockers before submission."))
        top_issue = "No top issue found."
        items = diagnosis.get("items", [])
        if isinstance(items, list) and items and isinstance(items[0], dict):
            first = items[0]
            top_issue = str(first.get("problem_statement", first.get("issue", ""))).strip() or top_issue

        en_001 = (
            "# 001 Submission Decision\n\n"
            + f"- Decision: **{decision_text}**\n"
            + f"- Meaning: {meaning}\n\n"
            + "## Top blocker\n"
            + f"- {top_issue}\n"
        )
        en_002 = (
            "# 002 Action Items\n\n"
            "1. Fix statistical/baseline evidence for top claims.\n"
            "2. Add direct evidence anchors (section/table/figure) per risk.\n"
            "3. Update rebuttal with concrete numbers and exact paper changes.\n"
        )
        rebuttal_items = rebuttal.get("items", []) if isinstance(rebuttal.get("items", []), list) else []
        first_resp = ""
        if rebuttal_items and isinstance(rebuttal_items[0], dict):
            first_resp = str(rebuttal_items[0].get("response", "")).strip()
        en_003 = (
            "# 003 Rebuttal Draft\n\n"
            "## R1\n"
            + (first_resp if first_resp else "Thank you for the concern. We will add direct evidence and explicit paper updates.")
            + "\n"
        )

        zh_001 = (
            "# 001 \u6295\u7a3f\u51b3\u7b56\n\n"
            + f"- \u7ed3\u8bba: **{decision_text}**\n"
            + f"- \u89e3\u91ca: {meaning}\n\n"
            + "## \u5f53\u524d\u6700\u5173\u952e\u95ee\u9898\n"
            + f"- {top_issue}\n"
        )
        zh_002 = (
            "# 002 \u884c\u52a8\u6e05\u5355\n\n"
            "1. \u5148\u4fee\u590d\u6700\u9ad8\u4f18\u5148\u7ea7\u98ce\u9669\u5bf9\u5e94\u7684\u7edf\u8ba1/\u57fa\u7ebf\u8bc1\u636e\u3002\n"
            "2. \u6bcf\u6761\u98ce\u9669\u8865\u9f50\u53ef\u5b9a\u4f4d\u951a\u70b9\uff08section/table/figure\uff09\u3002\n"
            "3. rebuttal \u5fc5\u987b\u5199\u6e05\u5177\u4f53\u6570\u5b57\u548c\u8bba\u6587\u4fee\u6539\u4f4d\u7f6e\u3002\n"
        )
        zh_003 = (
            "# 003 Rebuttal \u8349\u7a3f\n\n"
            "## R1\n"
            + (
                first_resp
                if first_resp
                else "\u611f\u8c22\u5ba1\u7a3f\u610f\u89c1\u3002\u6211\u4eec\u4f1a\u8865\u5145\u76f4\u63a5\u8bc1\u636e\uff0c\u5e76\u660e\u786e\u8bba\u6587\u4e2d\u5bf9\u5e94\u7684\u4fee\u6539\u4f4d\u7f6e\u3002"
            )
            + "\n"
        )

        return {
            "en": {"001": en_001, "002": en_002, "003": en_003},
            "zh": {"001": zh_001, "002": zh_002, "003": zh_003},
        }


    @staticmethod
    def _reviewer_question_simulation(spec: TaskSpec) -> dict:
        gaps = spec.context.get("gaps", [])
        risks = spec.context.get("top_risks", [])
        manuscript_stage = str(spec.context.get("manuscript_stage", "initial_submission")).strip().lower()
        reviewer_comments = spec.context.get("reviewer_comments", [])
        questions: list[dict] = []

        def add(
            priority: str,
            persona: str,
            question: str,
            why: str,
            code: str,
            risk_id: str = "",
            question_type: str = "evidence_closure",
        ) -> None:
            if any(question == q.get("question") for q in questions):
                return
            questions.append(
                {
                    "priority": priority,
                    "question_type": question_type,
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
                "stats_rigor",
            )
        if "missing_baseline" in code_set:
            add(
                "high",
                "systems",
                "Which strongest baselines are compared under matched compute/data settings?",
                "Baseline fairness is a common rejection checkpoint.",
                "missing_baseline",
                first_risk_id,
                "baseline_fairness",
            )
        if "missing_ablation" in code_set:
            add(
                "medium",
                "method",
                "Which component actually drives the gain, based on controlled ablations?",
                "Without controlled ablations, contribution attribution remains weak.",
                "missing_ablation",
                first_risk_id,
                "ablation_attribution",
            )

        if manuscript_stage in {"meta_review_discussion", "rejected_after_reviews"} and isinstance(reviewer_comments, list):
            for row in reviewer_comments[:3]:
                if not isinstance(row, dict):
                    continue
                review_id = str(row.get("review_id", "")).strip() or "R?"
                concern = str(row.get("concern", "")).strip()
                if not concern:
                    continue
                add(
                    "high",
                    "meta-review",
                    f"Reviewer {review_id} concern '{concern}': what exact new evidence and paper edits directly resolve this point?",
                    "Discussion-stage outcomes depend on direct closure of reviewer concerns.",
                    "weak_claim_alignment",
                    first_risk_id,
                    "stage_followup",
                )
            add(
                "medium",
                "critical",
                "Which rebuttal claims still rely on promises rather than completed evidence?",
                "Meta reviewers discount rebuttals that avoid concrete completed evidence.",
                "weak_claim_alignment",
                first_risk_id,
                "stage_followup",
            )
        elif manuscript_stage == "initial_submission":
            add(
                "medium",
                "related-work",
                "Where is the novelty boundary against nearest prior work stated with concrete pairwise comparisons?",
                "Initial-submission reviewers often challenge broad novelty wording.",
                "missing_reference_coverage",
                first_risk_id,
                "novelty_boundary",
            )
            add(
                "medium",
                "reproducibility",
                "What exact reproducibility assets (code/scripts/seeds/configs) will be released at submission time?",
                "Reproducibility details are a frequent decision pivot.",
                "weak_claim_alignment",
                first_risk_id,
                "reproducibility",
            )
        if not questions:
            add(
                "high",
                "meta-review",
                "Can each core claim be mapped to one direct evidence anchor?",
                "Reviewers usually challenge indirect claim-evidence mapping.",
                "weak_claim_alignment",
                first_risk_id,
                "evidence_closure",
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
            "missing_baseline_fairness": 0.61,
            "missing_ablation": 0.58,
            "missing_reproducibility": 0.52,
            "missing_error_analysis": 0.41,
            "missing_robustness": 0.56,
            "missing_contribution_alignment": 0.60,
            "missing_limitations": 0.40,
            "missing_ethics_limitations": 0.38,
            "missing_practical_impact": 0.44,
            "missing_qualitative_analysis": 0.43,
            "terminology_inconsistency": 0.47,
            "missing_reference_coverage": 0.57,
            "missing_top_venue_related_work_coverage": 0.54,
            "missing_workload_diversity": 0.57,
            "missing_scalability_evaluation": 0.59,
            "missing_efficiency_tradeoff": 0.56,
            "missing_system_setting_reproducibility": 0.57,
            "claim_evidence_contradiction": 0.83,
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
        context = spec.context if isinstance(spec.context, dict) else {}
        baseline = context.get("baseline_profile", {})
        if not isinstance(baseline, dict):
            baseline = {}

        fallback_axes = ["novelty", "soundness", "experiment", "clarity"]
        fallback_weights = {"novelty": 0.24, "soundness": 0.3, "experiment": 0.31, "clarity": 0.15}
        fallback_reasons = [
            "Core claim evidence is not strong enough for a strict reviewer standard.",
            "Baseline fairness and significance reporting are not sufficiently explicit.",
            "Reproducibility and limitations are not discussed in enough detail.",
        ]
        fallback_checks = [
            "baseline_coverage",
            "statistical_significance",
            "ablation_completeness",
            "reproducibility_details",
            "limitation_discussion",
            "top_venue_related_work_coverage",
        ]

        axes_raw = baseline.get("scoring_axes", fallback_axes)
        axes = [str(x).strip().lower() for x in axes_raw if str(x).strip()] if isinstance(axes_raw, list) else []
        axes = list(dict.fromkeys(axes)) or fallback_axes

        weights = dict(fallback_weights)
        raw_weights = baseline.get("weights")
        if isinstance(raw_weights, dict):
            for key, value in raw_weights.items():
                k = str(key).strip().lower()
                try:
                    weights[k] = float(value)
                except Exception:  # noqa: BLE001
                    continue

        reasons = list(fallback_reasons)
        raw_reasons = baseline.get("common_reject_reasons")
        if isinstance(raw_reasons, list):
            cleaned = [str(x).strip() for x in raw_reasons if str(x).strip()]
            if cleaned:
                reasons = list(dict.fromkeys(cleaned + reasons))
        reasons = list(
            dict.fromkeys(
                reasons
                + [
                    "Recent related-work coverage against top venues remains underdeveloped.",
                    "Claim-level evidence is still indirect for one or more high-impact contributions.",
                ]
            )
        )[:12]

        checks = list(fallback_checks)
        raw_checks = baseline.get("required_checks")
        if isinstance(raw_checks, list):
            cleaned = [str(x).strip() for x in raw_checks if str(x).strip()]
            if cleaned:
                checks = list(dict.fromkeys(cleaned + checks))
        checks = list(
            dict.fromkeys(
                checks
                + [
                    "contribution_alignment",
                    "error_analysis",
                    "robustness_checks",
                    "practical_impact",
                ]
            )
        )

        specs = {}
        raw_specs = baseline.get("required_check_specs")
        if isinstance(raw_specs, dict):
            for key, value in raw_specs.items():
                k = str(key).strip()
                if k and isinstance(value, dict):
                    specs[k] = dict(value)
        specs.setdefault(
            "error_analysis",
            {
                "check_name": "error_analysis",
                "gap_code": "missing_error_analysis",
                "description": "Failure-case analysis is too limited to support reviewer trust.",
                "severity_hint": "P2",
                "keywords": ["error analysis", "failure case", "qualitative", "where it fails"],
                "min_hits": 1,
                "min_distinct_sections": 1,
            },
        )
        specs.setdefault(
            "robustness_checks",
            {
                "check_name": "robustness_checks",
                "gap_code": "missing_robustness",
                "description": "Robustness checks are missing or underdeveloped.",
                "severity_hint": "P1",
                "keywords": ["robustness", "ood", "noise", "stress test", "perturbation"],
                "min_hits": 1,
                "min_distinct_sections": 1,
            },
        )
        specs.setdefault(
            "contribution_alignment",
            {
                "check_name": "contribution_alignment",
                "gap_code": "missing_contribution_alignment",
                "description": "Contribution statements are not concretely aligned with evidence.",
                "severity_hint": "P1",
                "keywords": ["contribution", "we propose", "novelty", "prior work"],
                "min_hits": 2,
                "min_distinct_sections": 1,
            },
        )
        specs.setdefault(
            "practical_impact",
            {
                "check_name": "practical_impact",
                "gap_code": "missing_practical_impact",
                "description": "Practical impact/deployment value is not convincingly demonstrated.",
                "severity_hint": "P2",
                "keywords": ["deployment", "practical", "real-world", "latency", "cost"],
                "min_hits": 1,
                "min_distinct_sections": 1,
            },
        )
        checks = list(dict.fromkeys(checks + list(specs.keys())))

        return {
            "scoring_axes": axes,
            "weights": weights,
            "common_reject_reasons": reasons,
            "required_checks": checks,
            "required_check_specs": specs,
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
        issues: list[str] = []
        if not pass_flag:
            issues.append("low_concern_coverage")

        statistical_concern = DeterministicExecutor._is_statistical_concern(concern)
        stat_meta = DeterministicExecutor._statistical_fix_meta(blob) if statistical_concern else {}
        if statistical_concern and not stat_meta.get("strict_pass", True):
            pass_flag = False
            for miss in stat_meta.get("critical_missing", []):
                issues.append(f"statistical_missing:{miss}")

        revised = None
        revised_new_evidence = new_evidence if isinstance(new_evidence, list) else []
        revised_paper_change = None
        if not pass_flag:
            revised = (
                response.rstrip()
                + " Specifically, we will address this concern with direct evidence, stronger analysis, and explicit paper updates in revision."
            ).strip()
            if statistical_concern:
                revised = (
                    revised
                    + " We will run at least 5 seeds and report mean/std plus 95% confidence intervals for all primary metrics, "
                    "with paired significance tests and explicit p-value thresholds (p<0.05)."
                ).strip()
                if not isinstance(revised_new_evidence, list):
                    revised_new_evidence = []
                revised_new_evidence = [
                    str(x).strip() for x in revised_new_evidence if str(x).strip()
                ]
                if not any(">=5 seeds" in x or "at least 5 seeds" in x.lower() for x in revised_new_evidence):
                    revised_new_evidence.append(
                        "Add statistical validation table with >=5 seeds, mean/std, 95% confidence intervals, and paired tests (p<0.05) for all primary metrics."
                    )
                if not any("metric-level p-values" in x.lower() or "p-values" in x.lower() for x in revised_new_evidence):
                    revised_new_evidence.append("Report metric-level p-values and indicate which gains remain significant.")
                revised_paper_change = (
                    "Update Results and add a dedicated Statistical Analysis subsection with seed settings, test protocol, and thresholds."
                )
        return {
            "pass": pass_flag,
            "issues": [] if pass_flag else list(dict.fromkeys(issues)),
            "revised_response": revised,
            "revised_new_evidence": revised_new_evidence,
            "revised_paper_change": revised_paper_change,
            "statistical_check": stat_meta if statistical_concern else {},
        }

    @staticmethod
    def _is_statistical_concern(text: str) -> bool:
        blob = str(text or "").lower()
        return any(
            token in blob
            for token in [
                "statistical",
                "significance",
                "p-value",
                "p value",
                "confidence interval",
                "seed",
                "variance",
                "std",
            ]
        )

    @staticmethod
    def _statistical_fix_meta(text: str) -> dict:
        blob = str(text or "").lower()
        has_seed_plan = bool(re.search(r"\b\d+\s*seeds?\b|\bmulti[-\s]?seed\b|\bmultiple\s+seeds\b", blob))
        has_test_name = bool(
            re.search(
                r"\b(significance\s+tests?|paired\s+tests?|paired\s+t-?tests?|t-?tests?|wilcoxon|mann[-\s]?whitney|anova|bootstrap)\b",
                blob,
            )
        )
        has_pvalue = bool(re.search(r"\bp\s*(?:<|<=|=)\s*0?\.\d+|\bp-?value\b", blob))
        has_uncertainty = bool(
            re.search(
                r"\b(mean\s*/\s*std|mean\s*\+/-\s*std|mean and std|std\b|standard deviation|variance|confidence interval|95%\s*ci|\bci\b)\b",
                blob,
            )
        )
        has_metric_scope = bool(re.search(r"\b(all|each|every)\s+(?:primary|main|headline)?\s*metrics?\b|\bprimary\s+metrics?\b", blob))
        has_numeric_threshold = bool(re.search(r"\bp\s*(?:<|<=|=)\s*0?\.\d+|\b95%\s*ci\b|\b99%\s*ci\b", blob))
        has_test_or_pvalue = bool(has_test_name or has_pvalue)

        critical_missing: list[str] = []
        if not has_seed_plan:
            critical_missing.append("seed_plan")
        if not has_test_or_pvalue:
            critical_missing.append("significance_test_or_pvalue")
        if not has_uncertainty:
            critical_missing.append("uncertainty_reporting")
        quality_missing: list[str] = []
        if not has_metric_scope:
            quality_missing.append("metric_scope")
        if not has_numeric_threshold:
            quality_missing.append("numeric_threshold")
        strength_score = sum(
            [
                bool(has_seed_plan),
                bool(has_test_or_pvalue),
                bool(has_uncertainty),
                bool(has_metric_scope),
                bool(has_numeric_threshold),
            ]
        )
        return {
            "strict_pass": len(critical_missing) == 0,
            "strength_score": int(strength_score),
            "critical_missing": critical_missing,
            "quality_missing": quality_missing,
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
    def _diagnosis_deep_dive(spec: TaskSpec) -> dict:
        context = spec.context if isinstance(spec.context, dict) else {}
        risk = context.get("risk", {}) if isinstance(context.get("risk", {}), dict) else {}
        related_claims = context.get("related_claims", [])
        evidence_anchors = context.get("evidence_anchors", [])
        followups = context.get("reviewer_followups", [])
        fallback = context.get("fallback", {}) if isinstance(context.get("fallback", {}), dict) else {}

        reason = str(risk.get("reason", "")).strip() or "Core claim-evidence support is not yet convincing."
        severity = str(risk.get("severity", "P1")).upper().strip()

        claim_hint = ""
        if isinstance(related_claims, list):
            claim_ids = [
                str(row.get("claim_id", "")).strip()
                for row in related_claims
                if isinstance(row, dict) and str(row.get("claim_id", "")).strip()
            ]
            if claim_ids:
                claim_hint = f" The affected claims include {', '.join(claim_ids[:3])}."

        anchor_hint = ""
        if isinstance(evidence_anchors, list) and evidence_anchors:
            first = evidence_anchors[0] if isinstance(evidence_anchors[0], dict) else {}
            section = str(first.get("section", "")).strip() or "unknown"
            passage_id = str(first.get("passage_id", "")).strip() or "unknown"
            anchor_hint = f" Current evidence anchor is {section}/{passage_id}, which should be replaced or strengthened with cleaner numeric evidence."

        root_cause = (
            str(fallback.get("root_cause_analysis", "")).strip()
            or f"The draft does not provide a direct, reviewer-verifiable evidence block for this risk: {reason}.{claim_hint}{anchor_hint}"
        )
        if len(root_cause) < 40:
            root_cause = (
                f"The current evidence chain remains indirect for this concern ({reason})."
                f"{claim_hint}{anchor_hint}"
            )

        impact = (
            str(fallback.get("impact_analysis", "")).strip()
            or (
                "If reviewers cannot map this claim to concrete numbers and controlled settings, they will discount confidence "
                "even when headline results look strong."
            )
        )
        if severity == "P0":
            impact = (
                "This is a central blocker: unresolved evidence ambiguity can independently drive rejection at meta-review."
            )

        fix_summary = (
            str(fallback.get("fix_summary", "")).strip()
            or "Convert this into a claim-to-evidence table with one row per claim and explicit statistical support."
        )
        if "statistical" in reason.lower() or "significance" in reason.lower():
            fix_summary = (
                "Add multi-seed statistics (mean/std), significance tests, and confidence intervals for all headline gains."
            )

        actions: list[dict] = []
        raw_actions = fallback.get("fix_actions", [])
        if isinstance(raw_actions, list):
            for row in raw_actions:
                if isinstance(row, dict):
                    actions.append(
                        {
                            "action": str(row.get("action", "")).strip(),
                            "why": str(row.get("why", "")).strip()
                            or "Directly reduces reviewer uncertainty.",
                            "expected_gain": str(row.get("expected_gain", "")).strip()
                            or "Improve confidence in claim validity.",
                            "section_target": str(row.get("section_target", "")).strip() or "Experiments",
                            "effort": str(row.get("effort", "M")).strip().upper() or "M",
                        }
                    )
        if not actions:
            actions = [
                {
                    "action": "Add one targeted experiment directly tied to the criticized claim.",
                    "why": "Turns indirect evidence into direct evidence.",
                    "expected_gain": "Reduce immediate reject risk.",
                    "section_target": "Experiments",
                    "effort": "M",
                }
            ]

        if not isinstance(followups, list):
            followups = []
        followup_list = [str(x).strip() for x in followups if str(x).strip()][:4]
        if not followup_list:
            if "statistical" in reason.lower() or "significance" in reason.lower():
                followup_list = [
                    "How many seeds were used, and are the gains statistically significant on every primary metric?"
                ]
            else:
                followup_list = [
                    "Can you point to the exact figure/table and numeric delta that validates this claim?"
                ]

        expected_impact = (
            str(fallback.get("expected_impact", "")).strip()
            or "After these fixes, this issue should move from a reject trigger to a manageable discussion point."
        )

        return {
            "problem_statement": reason,
            "root_cause_analysis": root_cause,
            "impact_analysis": impact,
            "fix_summary": fix_summary,
            "fix_actions": actions[:4],
            "expected_impact": expected_impact,
            "reviewer_followups": followup_list,
            "confidence": 0.74,
            "generator": "deterministic_executor",
        }
    @staticmethod
    def _pseudo_translate(text: str) -> str:
        glossary = {
            "Novelty": "\u65b0\u9896\u6027",
            "Soundness": "\u6280\u672f\u6b63\u786e\u6027",
            "Experiment": "\u5b9e\u9a8c\u5145\u5206\u6027",
            "Clarity": "\u5199\u4f5c\u6e05\u6670\u5ea6",
            "Rebuttal": "\u7b54\u8fa9\u56de\u590d",
            "Risk": "\u98ce\u9669",
            "Decision": "\u51b3\u7b56",
            "Not Ready": "\u4e0d\u5efa\u8bae\u6295\u7a3f",
            "Borderline": "\u8fb9\u754c\u72b6\u6001",
            "Ready": "\u53ef\u6295\u7a3f",
        }
        translated = text
        for en, zh in glossary.items():
            translated = translated.replace(en, zh)
        return translated


    @staticmethod
    def _reject_phrase_for_gap(code: str) -> str:
        c = code.lower()
        if "claim_evidence_contradiction" in c:
            return "Some reported results appear inconsistent with the stated claim direction."
        if "baseline" in c:
            return "Baseline comparisons are not yet convincing."
        if "significance" in c:
            return "Statistical validity is currently under-supported."
        if "ablation" in c:
            return "Ablation evidence is insufficient for attribution."
        if "reproduc" in c:
            return "Reproducibility details are below expected standards."
        if "top_venue_related_work_coverage" in c:
            return "Recent top-venue positioning is not yet convincing."
        if "citation" in c or "related_work" in c or "reference_coverage" in c:
            return "Related-work positioning is currently weak."
        if "contribution_alignment" in c:
            return "Contribution boundaries and novelty positioning remain under-specified."
        if "robustness" in c:
            return "Robustness evidence is insufficient for the claimed scope."
        if "error_analysis" in c:
            return "Failure-case analysis is too shallow to support reviewer confidence."
        if "limitations" in c:
            return "Limitations and scope boundaries are not explicitly discussed."
        if "terminology_inconsistency" in c:
            return "Terminology and acronym usage are inconsistent across sections."
        if "practical_impact" in c:
            return "Practical deployment impact is not sufficiently demonstrated."
        if "workload_diversity" in c:
            return "Workload diversity is insufficient to justify generalization claims."
        if "scalability_evaluation" in c:
            return "Scalability evidence is incomplete for system-level claims."
        if "efficiency_tradeoff" in c:
            return "Efficiency trade-offs are not convincingly characterized."
        if "weak_claim_alignment" in c:
            return "Key claims remain weakly grounded in direct evidence."
        return "Experimental evidence does not yet meet venue expectations."

    @staticmethod
    def _fix_hint_for_gap(code: str) -> str:
        c = code.lower()
        if "claim_evidence_contradiction" in c:
            return "Identify conflicting anchors, narrow claim scope, and add reconciled result analysis."
        if "baseline" in c:
            return "Add stronger baselines under matched settings and document fairness constraints."
        if "significance" in c:
            return "Add multi-seed mean/std, confidence intervals, and significance tests."
        if "ablation" in c:
            return "Add complete component and interaction ablations."
        if "reproduc" in c:
            return "Provide full implementation, environment, and deterministic rerun details."
        if "top_venue_related_work_coverage" in c:
            return "Add recent top-venue papers (last 2-3 years) and explicit novelty-positioning differences."
        if "citation" in c or "related_work" in c or "reference_coverage" in c:
            return "Expand recent top-venue references and add novelty-positioning analysis."
        if "contribution_alignment" in c:
            return "Add a contribution-positioning table against nearest prior work and tighten claim scope."
        if "robustness" in c:
            return "Add OOD/noise/stress robustness evaluations and discuss degradation behavior."
        if "error_analysis" in c:
            return "Add failure taxonomy with representative error cases and root-cause analysis."
        if "limitations" in c:
            return "Add explicit limitations and scope-boundary discussion with concrete failure conditions."
        if "terminology_inconsistency" in c:
            return (
                "Create a canonical terminology table (full name + acronym), "
                "then enforce consistent term usage and notation across all sections."
            )
        if "practical_impact" in c:
            return "Add deployment-oriented impact analysis, including latency/cost/operational constraints."
        if "workload_diversity" in c:
            return "Add heterogeneous workloads/benchmarks and explain why they are representative."
        if "scalability_evaluation" in c:
            return "Add scale-up/scale-out experiments across data size and resource scale."
        if "efficiency_tradeoff" in c:
            return "Report throughput-latency-resource trade-off curves, not only single-point metrics."
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

