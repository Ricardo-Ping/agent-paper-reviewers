from __future__ import annotations

import re

from ..executors.base import ExecutorAdapter
from ..models import TaskSpec
from ..services.translator import Translator
from .base import PipelineContext, PipelineStep


class ReportBuilderStep(PipelineStep):
    name = "ReportBuilder"

    def __init__(self, translator: Translator, executor: ExecutorAdapter | None = None) -> None:
        self.translator = translator
        self.executor = executor

    def run(self, ctx: PipelineContext) -> None:
        ranking = ctx.artifacts["risk_ranking"]
        risks = ranking["risks"]
        scores = ranking["scores"]
        score_explanations = self._ensure_score_explanations(ranking)
        gaps_payload = ctx.artifacts.get("gaps", {})
        gaps = gaps_payload.get("gaps", []) if isinstance(gaps_payload, dict) else []
        remediation = ctx.artifacts["remediation_plan"]["tasks"]
        alignments = ctx.artifacts["claim_evidence_matrix"]["alignments"]
        reviewer_questions = ctx.artifacts.get("reviewer_questions", {})
        top_reviewer_questions = (
            reviewer_questions.get("questions", [])[:5]
            if isinstance(reviewer_questions, dict)
            else []
        )
        historical_profile = ctx.artifacts.get("historical_profile_prior", {})
        venue_recommendations = ctx.artifacts.get("venue_recommendations", {})
        paper_qa_gate = ctx.artifacts.get("paper_qa_gate", {})
        venue_name = str(ctx.input_data.venue.name or "")
        venue_profile = ctx.artifacts.get("venue_profile", {}).get("profile", {})
        manuscript_stage = ctx.input_data.review_context.manuscript_stage.value
        reviewer_comments_count = len(ctx.input_data.review_context.reviewer_comments)
        stage_strategy = self._stage_strategy(manuscript_stage)
        stage_focus = ranking.get("focus_risks", risks)
        if not isinstance(stage_focus, list) or not stage_focus:
            stage_focus = risks
        submission_readiness = self._build_submission_readiness(
            venue_name=venue_name,
            venue_recommendations=venue_recommendations,
            gaps=gaps if isinstance(gaps, list) else [],
            risks=risks if isinstance(risks, list) else [],
            paper_qa_gate=paper_qa_gate if isinstance(paper_qa_gate, dict) else {},
            qa_issues=ctx.qa_issues,
            manuscript_stage=manuscript_stage,
        )

        decision, decision_policy, decision_interpretation = self._decision(
            risks=risks,
            scores=scores,
            venue_name=venue_name,
            venue_profile=venue_profile,
            manuscript_stage=manuscript_stage,
        )
        top_risks = stage_focus[:5]
        top_tasks = remediation[:5]

        decision_json = {
            "decision": decision,
            "manuscript_stage": manuscript_stage,
            "reviewer_comments_count": reviewer_comments_count,
            "stage_strategy": stage_strategy,
            "decision_interpretation": decision_interpretation,
            "scores": scores,
            "score_leverage_analysis": self._score_leverage_analysis(
                scores=scores,
                decision_policy=decision_policy,
                venue_profile=venue_profile,
            ),
            "score_explanations": score_explanations,
            "historical_profile": historical_profile,
            "venue_recommendations": venue_recommendations,
            "paper_qa_gate": paper_qa_gate,
            "submission_readiness": submission_readiness,
            "decision_policy_used": decision_policy,
            "stage_strategy_runtime": ranking.get("stage_strategy", {}),
            "top_risks": top_risks,
            "top_remediation_tasks": top_tasks,
            "predicted_reviewer_questions": top_reviewer_questions[:3],
        }
        full_json = {
            "decision": decision,
            "manuscript_stage": manuscript_stage,
            "reviewer_comments_count": reviewer_comments_count,
            "stage_strategy": stage_strategy,
            "decision_interpretation": decision_interpretation,
            "scores": scores,
            "score_leverage_analysis": self._score_leverage_analysis(
                scores=scores,
                decision_policy=decision_policy,
                venue_profile=venue_profile,
            ),
            "score_explanations": score_explanations,
            "historical_profile": historical_profile,
            "venue_recommendations": venue_recommendations,
            "paper_qa_gate": paper_qa_gate,
            "submission_readiness": submission_readiness,
            "decision_policy_used": decision_policy,
            "stage_strategy_runtime": ranking.get("stage_strategy", {}),
            "all_risks": risks,
            "focus_risks": stage_focus,
            "claim_evidence_matrix": alignments,
            "reviewer_question_simulation": reviewer_questions,
            "remediation_tasks": remediation,
            "rebuttal_plan": ctx.artifacts.get("rebuttal_plan", {}),
            "rebuttal": ctx.artifacts["rebuttal"]["en"]["bundle"],
        }
        diagnosis_json = self._diagnosis_json(ctx, full_json)

        decision_md = self._decision_md(decision_json)
        full_md = self._full_md(full_json)
        diagnosis_md = self._diagnosis_md(diagnosis_json)

        payload = {
            "en": {
                "decision_json": decision_json,
                "decision_md": decision_md,
                "full_json": full_json,
                "full_md": full_md,
                "diagnosis_json": diagnosis_json,
                "diagnosis_md": diagnosis_md,
            }
        }

        if ctx.input_data.options.language_mode.value == "en_zh":
            zh_decision_json = self._decision_json_zh(decision_json)
            zh_full_json = self._full_json_zh(full_json, ctx)
            zh_diagnosis_json = self._diagnosis_json_zh(diagnosis_json)
            payload["zh"] = {
                "decision_json": zh_decision_json,
                "decision_md": self._decision_md_zh(zh_decision_json),
                "full_json": zh_full_json,
                "full_md": self._full_md_zh(zh_full_json),
                "diagnosis_json": zh_diagnosis_json,
                "diagnosis_md": self._diagnosis_md_zh(zh_diagnosis_json),
            }

        ctx.artifacts["reports"] = payload
        ctx.dump_json("artifacts/report.decision.en.json", decision_json)
        ctx.dump_json("artifacts/report.full.en.json", full_json)
        ctx.dump_json("artifacts/report.diagnosis.en.json", diagnosis_json)
        if "zh" in payload:
            ctx.dump_json("artifacts/report.decision.zh.json", payload["zh"]["decision_json"])
            ctx.dump_json("artifacts/report.full.zh.json", payload["zh"]["full_json"])
            ctx.dump_json("artifacts/report.diagnosis.zh.json", payload["zh"]["diagnosis_json"])

    @staticmethod
    def _decision(
        *,
        risks: list[dict],
        scores: dict,
        venue_name: str,
        venue_profile: dict | None,
        manuscript_stage: str = "initial_submission",
    ) -> tuple[str, dict, str]:
        policy = ReportBuilderStep._resolve_decision_policy(venue_name, venue_profile)
        has_p0 = any(r["severity"] == "P0" for r in risks)
        p1_count = sum(1 for r in risks if r["severity"] == "P1")
        overall = float(scores.get("overall", 0.0) or 0.0)

        if manuscript_stage == "rejected_after_reviews":
            if has_p0 or p1_count >= 3 or overall < 6.0:
                return "Major Revision Required", policy, (
                    "This draft is in post-reject mode: priority is salvageability for resubmission, "
                    "not immediate re-submit."
                )
            if p1_count >= 1 or overall < 7.0:
                return "Resubmission Candidate", policy, (
                    "Salvage is possible if top reviewer-critical issues are fixed with direct evidence."
                )
            return "Ready for Resubmission", policy, (
                "Main reject drivers are reduced; this draft is close to a viable resubmission package."
            )

        if manuscript_stage == "meta_review_discussion":
            if has_p0 or p1_count >= 3 or overall < 6.0:
                return "Weak Discussion Position", policy, (
                    "Discussion-stage priority is point-by-point rebuttal rescue; current position is weak."
                )
            if p1_count >= 1 or overall < 7.0:
                return "Recoverable in Discussion", policy, (
                    "Directly answering reviewer concerns can still recover the decision trajectory."
                )
            return "Strong Discussion Position", policy, (
                "The draft has enough support to defend key concerns during reviewer discussion."
            )

        if has_p0 and bool(policy.get("p0_not_ready", True)):
            return "Not Ready", policy, (
                "Initial-submission gate: high reject risk, hold submission until key blockers are fixed."
            )
        if p1_count >= int(policy.get("p1_not_ready_threshold", 99) or 99):
            return "Not Ready", policy, (
                "Initial-submission gate: too many P1 risks for this venue tier."
            )
        if overall < float(policy.get("min_overall_borderline", 5.2) or 5.2):
            return "Not Ready", policy, (
                "Initial-submission gate: overall quality is below the practical threshold."
            )
        if p1_count >= int(policy.get("p1_borderline_threshold", 3) or 3):
            return "Borderline", policy, (
                "Submission is possible but reject risk remains material unless top risks are addressed."
            )
        if overall < float(policy.get("min_overall_ready", 6.0) or 6.0):
            return "Borderline", policy, (
                "Core quality is close, but not yet robust for a strong submit recommendation."
            )
        return "Ready", policy, "Initial-submission gate passed with manageable residual risk."

    @staticmethod
    def _stage_strategy(manuscript_stage: str) -> dict:
        mapping = {
            "initial_submission": {
                "title": "Initial Submission",
                "audience": "Author before first submission",
                "focus": "Should submit now or hold and revise first",
                "risk_heading_en": "Top Rejection Risks Before Submission",
                "risk_heading_zh": "投稿前高优先级拒稿风险",
                "task_heading_en": "Must-Do Pre-Submission Experiments",
                "task_heading_zh": "投稿前必补实验",
            },
            "rejected_after_reviews": {
                "title": "Post-Reject Revision",
                "audience": "Author after reject decision",
                "focus": "Salvageability for next round/resubmission",
                "risk_heading_en": "Top Salvage-Critical Risks From Prior Reject",
                "risk_heading_zh": "拒稿后最关键可挽救风险",
                "task_heading_en": "Must-Do Revision Actions",
                "task_heading_zh": "复投前必做修复动作",
            },
            "meta_review_discussion": {
                "title": "Meta-Review Discussion",
                "audience": "Author in reviewer discussion/rebuttal window",
                "focus": "Which reviewer concerns can be recovered now",
                "risk_heading_en": "Top Reviewer Concerns To Address Now",
                "risk_heading_zh": "当前讨论期最需要回应的审稿问题",
                "task_heading_en": "High-Leverage Discussion/Rebuttal Actions",
                "task_heading_zh": "讨论期高杠杆回应动作",
            },
        }
        return mapping.get(manuscript_stage, mapping["initial_submission"])

    @staticmethod
    def _resolve_decision_policy(venue_name: str, venue_profile: dict | None) -> dict:
        profile_policy = {}
        if isinstance(venue_profile, dict):
            maybe = venue_profile.get("decision_policy")
            if isinstance(maybe, dict):
                profile_policy = dict(maybe)

        venue_slug = venue_name.strip().lower().replace("_", "-").replace(" ", "-")
        high_competition = {
            "neurips",
            "iclr",
            "icml",
            "cvpr",
            "eccv",
            "acl-arr",
        }
        medium_competition = {
            "kdd",
            "aaai",
            "emnlp",
            "sigmod",
            "vldb",
            "icde",
        }
        if venue_slug in high_competition:
            inferred = {
                "strictness_tier": "high_competition",
                "p0_not_ready": True,
                "p1_not_ready_threshold": 2,
                "p1_borderline_threshold": 1,
                "min_overall_ready": 7.0,
                "min_overall_borderline": 6.0,
                "notes": "High-competition venue: Borderline is near reject risk.",
            }
        elif venue_slug in medium_competition:
            inferred = {
                "strictness_tier": "medium_competition",
                "p0_not_ready": True,
                "p1_not_ready_threshold": 4,
                "p1_borderline_threshold": 2,
                "min_overall_ready": 6.5,
                "min_overall_borderline": 5.8,
                "notes": "Medium-competition venue with balanced gate.",
            }
        else:
            inferred = {
                "strictness_tier": "default",
                "p0_not_ready": True,
                "p1_not_ready_threshold": 5,
                "p1_borderline_threshold": 3,
                "min_overall_ready": 6.0,
                "min_overall_borderline": 5.2,
                "notes": "Default venue gate.",
            }

        merged = dict(inferred)
        for key in (
            "strictness_tier",
            "p0_not_ready",
            "p1_not_ready_threshold",
            "p1_borderline_threshold",
            "min_overall_ready",
            "min_overall_borderline",
            "notes",
        ):
            if key in profile_policy:
                merged[key] = profile_policy[key]
        return merged

    @staticmethod
    def _ensure_score_explanations(ranking: dict) -> dict:
        scores = ranking.get("scores", {})
        raw = ranking.get("score_explanations", {})
        out: dict[str, dict] = {}
        for axis in ("novelty", "soundness", "experiment", "clarity"):
            axis_score = float(scores.get(axis, 0.0) or 0.0)
            item = raw.get(axis, {}) if isinstance(raw, dict) else {}
            reasoning = ""
            signals: list[str] = []
            if isinstance(item, dict):
                reasoning = str(item.get("reasoning", "")).strip()
                if isinstance(item.get("signals"), list):
                    signals = [str(x).strip() for x in item.get("signals", []) if str(x).strip()]
            if not reasoning:
                reasoning = "Score generated from detected claim-evidence strength, venue checks, and risk distribution."
            out[axis] = {
                "score": round(axis_score, 2),
                "reasoning": reasoning,
                "signals": signals,
            }
        return out

    @staticmethod
    def _decision_md(payload: dict) -> str:
        stage = payload.get("stage_strategy", {}) if isinstance(payload.get("stage_strategy"), dict) else {}
        risk_heading = str(stage.get("risk_heading_en") or "Top Rejection Risks")
        task_heading = str(stage.get("task_heading_en") or "Must-Do Experiments")
        leverage = payload.get("score_leverage_analysis", {})
        lines = [
            "# Submission Decision Brief",
            "",
            f"Decision: **{payload['decision']}**",
            "",
            f"Manuscript Stage: `{payload.get('manuscript_stage', 'initial_submission')}`",
            f"Stage Focus: {stage.get('focus', 'Submission readiness')}",
            f"Decision Meaning: {payload.get('decision_interpretation', '')}",
            "",
            "## Scores",
            f"- Novelty: {payload['scores']['novelty']}",
            f"- Soundness: {payload['scores']['soundness']}",
            f"- Experiment: {payload['scores']['experiment']}",
            f"- Clarity: {payload['scores']['clarity']}",
            f"- Overall: {payload['scores']['overall']}",
            "",
            "## Score Rationales",
        ]
        for axis, label in (
            ("novelty", "Novelty"),
            ("soundness", "Soundness"),
            ("experiment", "Experiment"),
            ("clarity", "Clarity"),
        ):
            detail = payload.get("score_explanations", {}).get(axis, {})
            reasoning = str(detail.get("reasoning", "")).strip()
            lines.append(f"- {label}: {payload['scores'][axis]} | Reasoning: {reasoning}")

        gate = payload.get("paper_qa_gate", {})
        if isinstance(gate, dict) and gate:
            lines.extend(
                [
                    "",
                    "## Rebuttal Self-Review Gate",
                    f"- Accepted: {gate.get('accepted', True)}",
                    f"- Source: {gate.get('source', 'n/a')}",
                    f"- Rewrites applied: {gate.get('rewrites_applied', 0)}",
                ]
            )
            gate_issues = gate.get("issues", [])
            if isinstance(gate_issues, list) and gate_issues:
                lines.append(f"- Issues: {', '.join(str(x) for x in gate_issues[:4])}")

        lines.extend(ReportBuilderStep._submission_readiness_md_lines(payload.get("submission_readiness", {})))
        lines.extend(ReportBuilderStep._score_leverage_md_lines(leverage))
        lines.extend(ReportBuilderStep._historical_md_lines(payload.get("historical_profile", {})))
        lines.extend(ReportBuilderStep._venue_reco_md_lines(payload.get("venue_recommendations", {})))
        lines.extend(
            [
                "",
            "## Decision Policy",
            f"- Tier: {payload['decision_policy_used'].get('strictness_tier', 'default')}",
            f"- P1 -> Not Ready threshold: {payload['decision_policy_used'].get('p1_not_ready_threshold')}",
            f"- P1 -> Borderline threshold: {payload['decision_policy_used'].get('p1_borderline_threshold')}",
            f"- Min overall for Ready: {payload['decision_policy_used'].get('min_overall_ready')}",
            "",
            f"## {risk_heading}",
            ]
        )
        for risk in payload["top_risks"]:
            lines.append(f"- [{risk['severity']}] {risk['id']} ({risk['score']}): {risk['reason']}")

        lines.append(f"\n## {task_heading}")
        for task in payload["top_remediation_tasks"]:
            lines.append(
                f"- {task['id']} ({task['priority']}, effort={task['effort']}): {task['title']}"
            )

        followups = payload.get("predicted_reviewer_questions", [])
        if isinstance(followups, list) and followups:
            lines.append("\n## Likely Reviewer Follow-ups")
            for row in followups[:3]:
                if not isinstance(row, dict):
                    continue
                lines.append(f"- [{row.get('priority', 'medium')}] {row.get('question', '')}")
                why = str(row.get("why_this_will_be_asked", "")).strip()
                if why:
                    lines.append(f"  - Why: {why}")

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _full_md(payload: dict) -> str:
        stage = payload.get("stage_strategy", {}) if isinstance(payload.get("stage_strategy"), dict) else {}
        risk_heading = str(stage.get("risk_heading_en") or "Detailed Risks")
        leverage = payload.get("score_leverage_analysis", {})
        lines = [
            "# Full Review Report",
            "",
            f"Decision: **{payload['decision']}**",
            "",
            f"Manuscript Stage: `{payload.get('manuscript_stage', 'initial_submission')}`",
            f"Stage Focus: {stage.get('focus', 'Submission readiness')}",
            f"Decision Meaning: {payload.get('decision_interpretation', '')}",
            "",
            "## Score Rationales",
        ]
        for axis, label in (
            ("novelty", "Novelty"),
            ("soundness", "Soundness"),
            ("experiment", "Experiment"),
            ("clarity", "Clarity"),
        ):
            detail = payload.get("score_explanations", {}).get(axis, {})
            reasoning = str(detail.get("reasoning", "")).strip()
            lines.append(f"- {label}: {payload['scores'][axis]} | Reasoning: {reasoning}")

        gate = payload.get("paper_qa_gate", {})
        if isinstance(gate, dict) and gate:
            lines.extend(
                [
                    "",
                    "## Rebuttal Self-Review Gate",
                    f"- Accepted: {gate.get('accepted', True)}",
                    f"- Source: {gate.get('source', 'n/a')}",
                    f"- Rewrites applied: {gate.get('rewrites_applied', 0)}",
                ]
            )
            gate_issues = gate.get("issues", [])
            if isinstance(gate_issues, list) and gate_issues:
                lines.append(f"- Issues: {', '.join(str(x) for x in gate_issues[:6])}")

        lines.extend(ReportBuilderStep._submission_readiness_md_lines(payload.get("submission_readiness", {})))
        lines.extend(ReportBuilderStep._score_leverage_md_lines(leverage))
        lines.extend(ReportBuilderStep._historical_md_lines(payload.get("historical_profile", {})))
        lines.extend(ReportBuilderStep._venue_reco_md_lines(payload.get("venue_recommendations", {})))
        lines.extend(
            [
                "",
            f"## {risk_heading}",
            ]
        )
        focus = payload.get("focus_risks", payload.get("all_risks", []))
        if not isinstance(focus, list):
            focus = payload.get("all_risks", [])
        for risk in focus:
            lines.extend(
                [
                    f"### {risk['id']} [{risk['severity']}]",
                    f"- Score: {risk['score']}",
                    f"- Reason: {risk['reason']}",
                    f"- Likely Reject Phrase: {risk['likely_reject_phrase']}",
                    f"- Suggested Fix: {risk['fix_hint']}",
                    "",
                ]
            )

        lines.append("## Claim-Evidence Alignment")
        lines.append("- Traceback tip: use `artifacts/evidence_index.json -> passage_locator[passage_id]` to find section/page origin.")
        for row in payload["claim_evidence_matrix"]:
            diagnostics = row.get("diagnostics", {}) if isinstance(row, dict) else {}
            sections = diagnostics.get("selected_sections", []) if isinstance(diagnostics, dict) else []
            avg_quality = diagnostics.get("avg_quality", 0.0) if isinstance(diagnostics, dict) else 0.0
            section_text = ", ".join(str(x) for x in sections[:3]) if isinstance(sections, list) else ""
            contradiction = float(row.get("contradiction_score", 0.0) or 0.0)
            contradiction_flag = "yes" if bool(row.get("contradiction_detected")) else "no"
            contradiction_refs = row.get("contradictory_evidence_refs", [])
            contradiction_anchor = ""
            if isinstance(contradiction_refs, list) and contradiction_refs:
                first = contradiction_refs[0]
                if isinstance(first, dict):
                    contradiction_anchor = f"{first.get('section', '')}/{first.get('passage_id', '')}".strip("/")
            lines.append(
                f"- {row['claim_id']} [{row['strength']}] score={row['score']} -> {len(row['evidence_refs'])} evidence refs; "
                f"anchors={section_text or 'n/a'}; avg_quality={avg_quality}; "
                f"contradiction={contradiction_flag} ({contradiction}); "
                f"contradiction_anchor={contradiction_anchor or 'n/a'}"
            )

        plan = payload.get("rebuttal_plan", {})
        plan_items = []
        plan_audit = []
        plan_summary = {}
        if isinstance(plan, dict):
            raw_items = plan.get("plan_items", plan.get("items", []))
            if isinstance(raw_items, list):
                plan_items = raw_items
            if isinstance(plan.get("post_generation_audit", []), list):
                plan_audit = plan.get("post_generation_audit", [])
            if isinstance(plan.get("summary", {}), dict):
                plan_summary = plan.get("summary", {})
        if isinstance(plan_items, list) and plan_items:
            lines.append("\n## Rebuttal Plan (Before Drafting)")
            for row in plan_items[:5]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"- {row.get('review_id', 'R?')} -> risk {row.get('risk_id', 'n/a')}: "
                    f"{row.get('concern', '')}"
                )
                evid = row.get("evidence_targets", [])
                if isinstance(evid, list) and evid:
                    lines.append(f"  - Evidence targets: {evid[0]}")
        if isinstance(plan_summary, dict) and plan_summary:
            lines.append("\n## Rebuttal Plan Audit")
            lines.append(
                "- Summary: "
                + f"pass={plan_summary.get('pass_count', 0)}, "
                + f"warning={plan_summary.get('warning_count', 0)}, "
                + f"fail={plan_summary.get('fail_count', 0)}, "
                + f"manual_review_recommended={plan_summary.get('manual_review_recommended', False)}"
            )
            for row in plan_audit[:5]:
                if not isinstance(row, dict):
                    continue
                gaps = [str(x) for x in row.get("gaps", []) if str(x).strip()]
                gap_text = ", ".join(gaps[:2]) if gaps else "no actionable gaps"
                lines.append(
                    f"- {row.get('review_id', 'R?')} [{row.get('status', 'warning')}] {gap_text}"
                )

        lines.append("\n## Rebuttal Skeleton Included")
        lines.append("See `rebuttal.*` artifacts for per-review responses.")

        sim = payload.get("reviewer_question_simulation", {})
        rows = sim.get("questions", []) if isinstance(sim, dict) else []
        if isinstance(rows, list) and rows:
            lines.append("\n## Predicted Reviewer Questions")
            for row in rows[:8]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"- [{row.get('priority', 'medium')}] {row.get('question', '')} "
                    f"(persona={row.get('reviewer_persona', 'empirical')})"
                )
                why = str(row.get("why_this_will_be_asked", "")).strip()
                if why:
                    lines.append(f"  - Why: {why}")
                evid = row.get("evidence_to_prepare", [])
                if isinstance(evid, list) and evid:
                    lines.append(f"  - Evidence to prepare: {evid[0]}")

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _historical_md_lines(profile: dict) -> list[str]:
        if not isinstance(profile, dict) or not profile:
            return []

        lines = ["", "## Historical Weakness Profile"]
        author_profile = profile.get("author_profile", {})
        venue_profile = profile.get("venue_year_profile", {})

        if isinstance(author_profile, dict) and author_profile:
            lines.append(f"- Author history runs: {author_profile.get('runs', 0)}")
            top = author_profile.get("top_weaknesses", [])
            if isinstance(top, list) and top:
                top_text = ", ".join(
                    f"{w.get('name')}({w.get('count')})"
                    for w in top[:3]
                    if isinstance(w, dict)
                )
                if top_text:
                    lines.append(f"- Author recurring weaknesses: {top_text}")

        if isinstance(venue_profile, dict) and venue_profile:
            lines.append(f"- Venue/year history runs: {venue_profile.get('runs', 0)}")
            top = venue_profile.get("top_weaknesses", [])
            if isinstance(top, list) and top:
                top_text = ", ".join(
                    f"{w.get('name')}({w.get('count')})"
                    for w in top[:3]
                    if isinstance(w, dict)
                )
                if top_text:
                    lines.append(f"- Venue/year common weaknesses: {top_text}")

        if len(lines) == 2:
            lines.append("- No historical profile found yet. This run will start accumulating profile data.")
        return lines

    @staticmethod
    def _venue_reco_md_lines(payload: dict) -> list[str]:
        if not isinstance(payload, dict):
            return []
        rows = payload.get("recommended_venues", [])
        if not isinstance(rows, list) or not rows:
            return []

        lines = ["", "## Recommended Venues (If You Are Unsure)"]
        for row in rows[:3]:
            if not isinstance(row, dict):
                continue
            venue = str(row.get("venue", "")).upper()
            year = int(row.get("year", 0) or 0)
            score = row.get("match_score", 0)
            lines.append(f"- {venue} {year}: match={score}")
            reasons = row.get("reasons", [])
            if isinstance(reasons, list):
                for reason in reasons[:2]:
                    lines.append(f"  - {reason}")
        return lines

    @staticmethod
    def _score_leverage_md_lines(payload: dict) -> list[str]:
        if not isinstance(payload, dict) or not payload:
            return []

        lines = ["", "## Score Leverage Analysis (What Lifts Overall Fastest)"]
        fastest = payload.get("fastest_axis")
        rationale = payload.get("rationale")
        target = payload.get("target_axis_score")
        if fastest and fastest != "none":
            lines.append(
                f"- Fastest axis to improve first: **{str(fastest).capitalize()}** "
                f"(target score: {target})."
            )
        elif bool(payload.get("no_urgent_axis", False)):
            lines.append("- All major axes already meet the target threshold; prioritize closing concrete risk items.")
        if rationale:
            lines.append(f"- Why: {rationale}")

        rows = payload.get("axes", [])
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                axis = str(row.get("axis", "")).capitalize()
                lines.append(
                    "- "
                    + f"{axis}: score={row.get('score')}, weight={row.get('weight')}, "
                    + f"weighted_contribution={row.get('weighted_contribution')}, "
                    + f"weighted_gap_to_target={row.get('weighted_gap_to_target')}, "
                    + f"priority_index={row.get('priority_index')}"
                )
        return lines

    @staticmethod
    def _score_leverage_md_lines_zh(payload: dict) -> list[str]:
        if not isinstance(payload, dict) or not payload:
            return []

        lines = ["", "## 评分杠杆分析（最快拉升总分的维度）"]
        fastest = payload.get("fastest_axis")
        rationale = payload.get("rationale")
        target = payload.get("target_axis_score")
        if bool(payload.get("no_urgent_axis", False)) or str(fastest).strip() in {"none", "无"}:
            lines.append("- 当前主要维度已达到目标阈值，优先闭环具体风险项而非继续拉分。")
        elif fastest:
            lines.append(f"- 优先提升维度：**{fastest}**（目标分：{target}）。")
        if rationale:
            lines.append(f"- 原因：{rationale}")

        rows = payload.get("axes", [])
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                axis = str(row.get("axis", ""))
                lines.append(
                    "- "
                    + f"{axis}：当前分={row.get('score')}，权重={row.get('weight')}，"
                    + f"当前加权贡献={row.get('weighted_contribution')}，"
                    + f"距目标加权缺口={row.get('weighted_gap_to_target')}，"
                    + f"优先指数={row.get('priority_index')}"
                )
        return lines

    @staticmethod
    def _score_leverage_analysis(
        *,
        scores: dict,
        decision_policy: dict,
        venue_profile: dict,
    ) -> dict:
        axes = ["novelty", "soundness", "experiment", "clarity"]
        raw_weights = venue_profile.get("weights", {}) if isinstance(venue_profile, dict) else {}
        weights: dict[str, float] = {}
        for axis in axes:
            try:
                weights[axis] = max(0.0, float(raw_weights.get(axis, 0.25)))
            except (TypeError, ValueError):
                weights[axis] = 0.25
        total = sum(weights.values())
        if total <= 0:
            weights = {axis: 0.25 for axis in axes}
            total = 1.0
        weights = {axis: round(v / total, 4) for axis, v in weights.items()}

        try:
            target_axis = float(decision_policy.get("min_overall_ready", 7.0))
        except (TypeError, ValueError):
            target_axis = 7.0
        target_axis = max(6.0, min(8.5, target_axis))

        rows: list[dict] = []
        for axis in axes:
            try:
                score = float(scores.get(axis, 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            weight = float(weights.get(axis, 0.25))
            contribution = score * weight
            gap_to_target = max(0.0, target_axis - score)
            weighted_gap = gap_to_target * weight
            # Priority index: how much overall gain this axis can deliver against target gap.
            priority_index = weighted_gap
            rows.append(
                {
                    "axis": axis,
                    "score": round(score, 2),
                    "weight": round(weight, 4),
                    "weighted_contribution": round(contribution, 4),
                    "gap_to_target": round(gap_to_target, 2),
                    "weighted_gap_to_target": round(weighted_gap, 4),
                    "marginal_gain_per_plus1_axis_point": round(weight, 4),
                    "points_needed_for_plus_0_5_overall": round(0.5 / max(weight, 1e-6), 2),
                    "priority_index": round(priority_index, 4),
                }
            )
        rows.sort(key=lambda x: float(x.get("priority_index", 0.0)), reverse=True)
        total_weighted_gap = round(sum(float(x.get("weighted_gap_to_target", 0.0)) for x in rows), 4)
        no_urgent_axis = total_weighted_gap <= 1e-6
        if no_urgent_axis:
            fastest = "none"
            rationale = (
                "All axis scores already meet/exceed the target threshold. "
                "Prioritize closing concrete rejection risks instead of score maximization."
            )
        else:
            fastest = rows[0]["axis"] if rows else "soundness"
            rationale = (
                "Priority is ranked by weighted gap to target: (target_axis_score - current_score) * axis_weight. "
                "This directly estimates which axis can raise overall score fastest."
            )
        return {
            "target_axis_score": round(target_axis, 2),
            "fastest_axis": fastest,
            "rationale": rationale,
            "no_urgent_axis": no_urgent_axis,
            "axes": rows,
        }

    @staticmethod
    def _historical_md_lines_zh(profile: dict) -> list[str]:
        if not isinstance(profile, dict) or not profile:
            return []

        lines = ["", "## 历史弱项画像"]
        author_profile = profile.get("author_profile", {})
        venue_profile = profile.get("venue_year_profile", {})

        if isinstance(author_profile, dict) and author_profile:
            lines.append(f"- 作者历史运行次数：{author_profile.get('runs', 0)}")
            top = author_profile.get("top_weaknesses", [])
            if isinstance(top, list) and top:
                top_text = ", ".join(
                    f"{w.get('name')}({w.get('count')})"
                    for w in top[:3]
                    if isinstance(w, dict)
                )
                if top_text:
                    lines.append(f"- 作者高频弱项：{top_text}")

        if isinstance(venue_profile, dict) and venue_profile:
            lines.append(f"- 会议/年份历史运行次数：{venue_profile.get('runs', 0)}")
            top = venue_profile.get("top_weaknesses", [])
            if isinstance(top, list) and top:
                top_text = ", ".join(
                    f"{w.get('name')}({w.get('count')})"
                    for w in top[:3]
                    if isinstance(w, dict)
                )
                if top_text:
                    lines.append(f"- 会议/年份高频弱项：{top_text}")

        if len(lines) == 2:
            lines.append("- 暂无历史画像，本次运行后将开始累计。")
        return lines

    @staticmethod
    def _venue_reco_md_lines_zh(payload: dict) -> list[str]:
        if not isinstance(payload, dict):
            return []
        rows = payload.get("recommended_venues", [])
        if not isinstance(rows, list) or not rows:
            return []

        lines = ["", "## 会议推荐（当你不确定投哪里）"]
        for row in rows[:3]:
            if not isinstance(row, dict):
                continue
            venue = str(row.get("venue", "")).upper()
            year = int(row.get("year", 0) or 0)
            score = row.get("match_score", 0)
            lines.append(f"- {venue} {year}：匹配度={score}")
            reasons = row.get("reasons", [])
            if isinstance(reasons, list):
                for reason in reasons[:2]:
                    lines.append(f"  - {reason}")
        return lines

    @staticmethod
    def _submission_readiness_md_lines(payload: dict) -> list[str]:
        if not isinstance(payload, dict) or not payload:
            return []
        checks = payload.get("checks", [])
        if not isinstance(checks, list):
            checks = []

        lines = ["", "## Submission Readiness Checklist"]
        lines.append(
            f"- Overall: **{payload.get('overall_status', 'warning').upper()}** "
            f"(pass={payload.get('pass_count', 0)}, warning={payload.get('warning_count', 0)}, critical={payload.get('critical_count', 0)})"
        )
        if bool(payload.get("human_review_recommended", False)):
            reasons = payload.get("human_review_reasons", [])
            reason_text = ", ".join(str(x) for x in reasons[:4]) if isinstance(reasons, list) else "quality_gate"
            lines.append(f"- Human review recommended: yes ({reason_text})")
        next_action = str(payload.get("recommended_next_action", "")).strip()
        if next_action:
            lines.append(f"- Recommended action: {next_action}")

        blockers = [
            x
            for x in checks
            if isinstance(x, dict)
            and str(x.get("status", "")).strip().lower() not in {"pass", "通过"}
            and str(x.get("status", "")).strip() not in {"pass", "通过"}
        ]
        for row in blockers[:6]:
            lines.append(
                f"- [{str(row.get('status', 'warning')).upper()}] {row.get('id', '')}: {row.get('title', '')} "
                f"-> {row.get('why', '')}"
            )
            action = str(row.get("action", "")).strip()
            if action:
                lines.append(f"  - Fix: {action}")
        return lines

    @staticmethod
    def _submission_readiness_md_lines_zh(payload: dict) -> list[str]:
        if not isinstance(payload, dict) or not payload:
            return []
        checks = payload.get("checks", [])
        if not isinstance(checks, list):
            checks = []

        lines = ["", "## 投稿就绪清单"]
        lines.append(
            f"- 总体状态：**{payload.get('overall_status', 'warning')}** "
            f"(通过={payload.get('pass_count', 0)}，预警={payload.get('warning_count', 0)}，阻断={payload.get('critical_count', 0)})"
        )
        if bool(payload.get("human_review_recommended", False)):
            reasons = payload.get("human_review_reasons", [])
            reason_text = "、".join(str(x) for x in reasons[:4]) if isinstance(reasons, list) else "质量门控"
            lines.append(f"- 建议人工复核：是（{reason_text}）")
        next_action = str(payload.get("recommended_next_action", "")).strip()
        if next_action:
            lines.append(f"- 建议动作：{next_action}")

        blockers = [x for x in checks if isinstance(x, dict) and str(x.get("status", "")).lower() != "pass"]
        for row in blockers[:6]:
            lines.append(
                f"- [{row.get('status', '预警')}] {row.get('id', '')}: {row.get('title', '')} "
                f"-> {row.get('why', '')}"
            )
            action = str(row.get("action", "")).strip()
            if action:
                lines.append(f"  - 修复：{action}")
        return lines

    @staticmethod
    def _build_submission_readiness(
        *,
        venue_name: str,
        venue_recommendations: dict,
        gaps: list[dict],
        risks: list[dict],
        paper_qa_gate: dict,
        qa_issues: list[str],
        manuscript_stage: str,
    ) -> dict:
        gap_codes = {
            str(g.get("code", "")).strip().lower()
            for g in gaps
            if isinstance(g, dict)
        }
        risk_rows = [r for r in risks if isinstance(r, dict)]
        qa_rows = [str(x) for x in qa_issues if str(x).strip()]

        def _mk(
            check_id: str,
            category: str,
            title: str,
            status: str,
            why: str,
            action: str,
            evidence: str = "",
        ) -> dict:
            return {
                "id": check_id,
                "category": category,
                "title": title,
                "status": status,
                "why": why,
                "action": action,
                "evidence": evidence,
            }

        checks: list[dict] = []

        # 1) Venue scope fit
        scope_status = "warning"
        scope_why = "Current venue does not appear in top recommended venues."
        scope_evidence = ""
        rows = venue_recommendations.get("recommended_venues", []) if isinstance(venue_recommendations, dict) else []
        if isinstance(rows, list):
            for row in rows[:6]:
                if not isinstance(row, dict):
                    continue
                if str(row.get("venue", "")).strip().lower() == venue_name.strip().lower():
                    score = float(row.get("match_score", 0.0) or 0.0)
                    scope_status = "pass" if score >= 0.30 else "warning"
                    scope_why = (
                        f"Venue match score is {score:.3f}."
                        if scope_status == "pass"
                        else f"Venue match score is {score:.3f}, which is relatively weak."
                    )
                    scope_evidence = f"match_score={score:.3f}"
                    break
        checks.append(
            _mk(
                "CHK-001",
                "venue_fit",
                "Venue Scope Fit",
                scope_status,
                scope_why,
                "Re-evaluate venue choice or strengthen venue-specific positioning paragraph.",
                scope_evidence,
            )
        )

        # 2) Claim-evidence support
        ce_critical = any(
            str(r.get("severity", "")).upper() == "P0"
            and ("claim" in str(r.get("reason", "")).lower() or "evidence" in str(r.get("reason", "")).lower())
            for r in risk_rows
        )
        ce_warning = any(
            str(r.get("severity", "")).upper() == "P1"
            and ("claim" in str(r.get("reason", "")).lower() or "evidence" in str(r.get("reason", "")).lower())
            for r in risk_rows
        )
        if ce_critical:
            checks.append(
                _mk(
                    "CHK-002",
                    "evidence_alignment",
                    "Claim-Evidence Alignment",
                    "critical",
                    "At least one P0 risk indicates core claim evidence is not sufficiently grounded.",
                    "Build claim-to-evidence table with exact section/table anchors and numeric support.",
                )
            )
        elif ce_warning:
            checks.append(
                _mk(
                    "CHK-002",
                    "evidence_alignment",
                    "Claim-Evidence Alignment",
                    "warning",
                    "Detected P1 claim-evidence weakness that can reduce reviewer confidence.",
                    "Add direct evidence blocks for weak claims and tighten claim wording scope.",
                )
            )
        else:
            checks.append(
                _mk(
                    "CHK-002",
                    "evidence_alignment",
                    "Claim-Evidence Alignment",
                    "pass",
                    "No major claim-evidence blocker detected.",
                    "Keep claim-to-evidence mapping explicit in final draft.",
                )
            )

        # 3-7) Common technical checks
        def gap_check(check_id: str, code_keys: set[str], title: str, action: str) -> dict:
            hit = any(code in gap_codes for code in code_keys)
            if hit:
                return _mk(
                    check_id,
                    "technical_checks",
                    title,
                    "warning",
                    f"Detected gap(s): {', '.join(sorted(code_keys & gap_codes))}",
                    action,
                )
            return _mk(
                check_id,
                "technical_checks",
                title,
                "pass",
                "No explicit gap detected.",
                action,
            )

        checks.append(
            gap_check(
                "CHK-003",
                {"missing_baseline"},
                "Baseline Coverage",
                "Add matched-setting comparison against strongest baselines.",
            )
        )
        checks.append(
            gap_check(
                "CHK-004",
                {"missing_significance", "significance_reporting"},
                "Statistical Significance",
                "Report multi-seed mean/std, confidence intervals, and paired tests.",
            )
        )
        checks.append(
            gap_check(
                "CHK-005",
                {"missing_ablation", "ablation_completeness"},
                "Ablation Completeness",
                "Add component and interaction ablations for key modules.",
            )
        )
        checks.append(
            gap_check(
                "CHK-006",
                {"missing_reproducibility", "reproducibility_details"},
                "Reproducibility Details",
                "Add full environment/hyperparameter/seed/rerun instructions.",
            )
        )
        checks.append(
            gap_check(
                "CHK-007",
                {"missing_reference_coverage", "missing_top_venue_related_work_coverage"},
                "Related Work Coverage",
                "Add nearest-neighbor comparison table and top-venue references.",
            )
        )

        # 8) Parser quality
        parser_warn = any(("paper_parser_warning" in x or "parser_quality" in x) for x in qa_rows)
        checks.append(
            _mk(
                "CHK-008",
                "input_quality",
                "Paper Parsing Quality",
                "warning" if parser_warn else "pass",
                "Parser quality warnings detected." if parser_warn else "No parser quality warning detected.",
                "Provide a machine-readable PDF or markdown source if parser quality is low.",
            )
        )

        # 9) Rebuttal quality gate
        gate_accept = bool(paper_qa_gate.get("accepted", True)) if isinstance(paper_qa_gate, dict) else True
        gate_rewrites = int(paper_qa_gate.get("rewrites_applied", 0) or 0) if isinstance(paper_qa_gate, dict) else 0
        if not gate_accept:
            gate_status = "critical"
            gate_why = "Self-review gate did not pass after rewrite."
        elif gate_rewrites > 0:
            gate_status = "warning"
            gate_why = "Rebuttal needed rewrite to pass self-review."
        else:
            gate_status = "pass"
            gate_why = "Rebuttal quality gate passed on first attempt."
        checks.append(
            _mk(
                "CHK-009",
                "rebuttal_quality",
                "Rebuttal Readiness",
                gate_status,
                gate_why,
                "Ensure each response contains concrete numbers and anchor locations.",
            )
        )

        status_rank = {"pass": 0, "warning": 1, "critical": 2}
        checks.sort(key=lambda x: (-status_rank.get(str(x.get("status", "pass")), 0), x.get("id", "")))
        critical_count = sum(1 for x in checks if x.get("status") == "critical")
        warning_count = sum(1 for x in checks if x.get("status") == "warning")
        pass_count = sum(1 for x in checks if x.get("status") == "pass")

        if critical_count > 0:
            overall = "critical"
            next_action = "Hold submission and close all CRITICAL checks first."
        elif warning_count > 0:
            overall = "warning"
            next_action = "Submission is possible, but close WARNING checks to reduce rejection risk."
        else:
            overall = "pass"
            next_action = "Checklist passed. Final polish and consistency review before submission."

        # discussion stages: checklist semantics change from submit/no-submit to rescue quality.
        if manuscript_stage in {"rejected_after_reviews", "meta_review_discussion"}:
            if critical_count > 0:
                next_action = "Prioritize reviewer-critical fixes and rebuttal evidence before next discussion round."
            elif warning_count > 0:
                next_action = "Focus on high-impact reviewer concerns and close remaining warning checks."
            else:
                next_action = "Use this as a stable rebuttal baseline and iterate with reviewer feedback."

        human_review_recommended = False
        human_review_reasons: list[str] = []
        if critical_count > 0:
            human_review_recommended = True
            human_review_reasons.append("critical_checks_present")
        if warning_count >= 3:
            human_review_recommended = True
            human_review_reasons.append("multiple_warning_checks")
        if any("paper_parser_warning" in x for x in qa_rows):
            human_review_recommended = True
            human_review_reasons.append("input_parser_quality_warning")

        return {
            "overall_status": overall,
            "pass_count": pass_count,
            "warning_count": warning_count,
            "critical_count": critical_count,
            "recommended_next_action": next_action,
            "human_review_recommended": human_review_recommended,
            "human_review_reasons": human_review_reasons,
            "checks": checks,
        }

    def _diagnosis_json(self, ctx: PipelineContext, full_json: dict) -> dict:
        risks = full_json.get("all_risks", [])
        remediation = full_json.get("remediation_tasks", [])
        rebuttal_items = full_json.get("rebuttal", {}).get("items", [])

        tasks_by_risk: dict[str, list[dict]] = {}
        for task in remediation:
            if not isinstance(task, dict):
                continue
            risk_id = str(task.get("risk_id") or "").strip()
            if not risk_id:
                continue
            tasks_by_risk.setdefault(risk_id, []).append(task)

        diagnosis_items: list[dict] = []
        for idx, risk in enumerate(risks, start=1):
            if not isinstance(risk, dict):
                continue
            risk_id = str(risk.get("id") or f"RISK-{idx:03d}")
            reason = str(risk.get("reason") or "").strip()
            severity = str(risk.get("severity") or "P2")
            linked_tasks = tasks_by_risk.get(risk_id, [])
            linked_rebuttal = rebuttal_items[idx - 1] if idx - 1 < len(rebuttal_items) else {}

            why_happened = self._default_why_happened(reason, severity)
            why_it_matters = self._default_why_it_matters(reason, severity)
            fix_plan = self._default_fix_plan(linked_tasks, risk)

            enriched = self._diagnosis_explain_with_executor(
                ctx,
                risk=risk,
                why_happened=why_happened,
                why_it_matters=why_it_matters,
                fix_plan=fix_plan,
            )
            if enriched is not None:
                why_happened = str(enriched.get("why_happened") or why_happened).strip()
                why_it_matters = str(enriched.get("why_it_matters") or why_it_matters).strip()
                fix_plan = str(enriched.get("fix_plan") or fix_plan).strip()

            evidence_refs = risk.get("evidence_refs", [])
            top_anchor = ""
            if isinstance(evidence_refs, list) and evidence_refs:
                top = evidence_refs[0] if isinstance(evidence_refs[0], dict) else {}
                section = str(top.get("section") or "unknown")
                passage_id = str(top.get("passage_id") or "unknown")
                excerpt = str(top.get("excerpt") or "").strip()
                top_anchor = f"{section}/{passage_id}: {excerpt}"[:260]

            diagnosis_items.append(
                {
                    "issue_id": risk_id,
                    "severity": severity,
                    "issue": reason,
                    "why_happened": why_happened,
                    "why_it_matters": why_it_matters,
                    "suggested_fix": fix_plan,
                    "linked_rebuttal_review_id": str(linked_rebuttal.get("review_id") or f"R{idx}"),
                    "linked_rebuttal_concern": str(linked_rebuttal.get("concern") or reason),
                    "evidence_anchor": top_anchor,
                }
            )

        return {
            "decision": full_json.get("decision"),
            "summary": {
                "risk_count": len(diagnosis_items),
                "p0_count": sum(1 for x in diagnosis_items if x.get("severity") == "P0"),
                "p1_count": sum(1 for x in diagnosis_items if x.get("severity") == "P1"),
            },
            "items": diagnosis_items,
        }

    def _diagnosis_explain_with_executor(
        self,
        ctx: PipelineContext,
        *,
        risk: dict,
        why_happened: str,
        why_it_matters: str,
        fix_plan: str,
    ) -> dict | None:
        if self.executor is None:
            return None

        spec = TaskSpec(
            task_type="diagnosis_explain",
            prompt=(
                "Rewrite the diagnosis for graduate students. Keep it concrete, plain, and actionable. "
                "Return JSON only."
            ),
            context={
                "risk": risk,
                "why_happened": why_happened,
                "why_it_matters": why_it_matters,
                "suggested_fix": fix_plan,
            },
            output_schema={
                "why_happened": "string",
                "why_it_matters": "string",
                "fix_plan": "string",
            },
            model_profile="judge",
        )
        result = self.executor.execute(spec)
        for warning in result.warnings:
            ctx.add_qa_issue(f"diagnosis_executor_warning:{warning}")
        if not result.ok:
            return None

        payload = result.output
        if isinstance(payload.get("response"), dict):
            payload = payload["response"]
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _default_why_happened(reason: str, severity: str) -> str:
        text = reason.lower()
        if "baseline" in text:
            return "Baseline design or matching settings are not explicit enough, so reviewers cannot judge fairness."
        if "significance" in text or "statistical" in text:
            return "Results are likely reported as single numbers without enough multi-seed uncertainty evidence."
        if "ablation" in text:
            return "Key components are not isolated with controlled ablations, so contribution attribution remains weak."
        if "reproduc" in text:
            return "Implementation and environment details are incomplete, making independent reruns hard."
        if "citation" in text or "related work" in text:
            return "Related work positioning is likely thin, especially on recent top-venue papers."
        if severity == "P0":
            return "This issue directly affects whether core claims are believable."
        return "Current draft evidence does not fully match the strict check expected by reviewers."

    @staticmethod
    def _default_why_it_matters(reason: str, severity: str) -> str:
        if severity == "P0":
            return "A reviewer can reject mainly because this issue alone undermines the paper's central claim."
        if severity == "P1":
            return "This is a common rejection reason and usually triggers weak confidence even if novelty is good."
        text = reason.lower()
        if "clarity" in text:
            return "Poor clarity lowers reviewer confidence and makes strengths harder to credit."
        return "Even as a secondary issue, it can reduce score consistency across reviewers."

    @staticmethod
    def _default_fix_plan(linked_tasks: list[dict], risk: dict) -> str:
        if linked_tasks:
            first = linked_tasks[0]
            return (
                f"Execute `{first.get('id', 'EXP-001')}` first: {first.get('title', 'targeted experiment')}; "
                f"then report protocol and significance details in the paper revision."
            )
        return str(risk.get("fix_hint") or "Add targeted evidence and update the related paper section explicitly.")

    @staticmethod
    def _diagnosis_md(payload: dict) -> str:
        lines = [
            "# Detailed Diagnosis Report",
            "",
            f"Decision Snapshot: **{payload.get('decision', 'N/A')}**",
            "",
            "## Summary",
            f"- Total issues: {payload.get('summary', {}).get('risk_count', 0)}",
            f"- P0 issues: {payload.get('summary', {}).get('p0_count', 0)}",
            f"- P1 issues: {payload.get('summary', {}).get('p1_count', 0)}",
            "",
            "## Issue-by-Issue Diagnosis",
        ]
        for item in payload.get("items", []):
            lines.extend(
                [
                    f"### {item.get('issue_id', 'RISK')} [{item.get('severity', 'P2')}]",
                    f"- Issue: {item.get('issue', '')}",
                    f"- Why It Happened: {item.get('why_happened', '')}",
                    f"- Why It Matters: {item.get('why_it_matters', '')}",
                    f"- Suggested Fix: {item.get('suggested_fix', '')}",
                    f"- Evidence Anchor: {item.get('evidence_anchor', '')}",
                    (
                        f"- Rebuttal Link: Reviewer {item.get('linked_rebuttal_review_id', '')} "
                        f"-> {item.get('linked_rebuttal_concern', '')}"
                    ),
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def _decision_json_zh(self, payload: dict) -> dict:
        stage_strategy = payload.get("stage_strategy", {})
        stage_strategy_zh = dict(stage_strategy) if isinstance(stage_strategy, dict) else {}
        if stage_strategy_zh:
            stage_strategy_zh["focus"] = self._phrase_zh(str(stage_strategy_zh.get("focus", "")))
            stage_strategy_zh["risk_heading_zh"] = stage_strategy_zh.get("risk_heading_zh", stage_strategy_zh.get("risk_heading_en"))
            stage_strategy_zh["task_heading_zh"] = stage_strategy_zh.get("task_heading_zh", stage_strategy_zh.get("task_heading_en"))
        return {
            "decision": self._decision_zh(payload["decision"]),
            "manuscript_stage": self._stage_zh(str(payload.get("manuscript_stage", "initial_submission"))),
            "reviewer_comments_count": payload.get("reviewer_comments_count", 0),
            "stage_strategy": stage_strategy_zh,
            "decision_interpretation": self._phrase_zh(str(payload.get("decision_interpretation", ""))),
            "scores": payload["scores"],
            "score_leverage_analysis": self._score_leverage_analysis_zh(payload.get("score_leverage_analysis", {})),
            "score_explanations": self._score_explanations_zh(payload.get("score_explanations", {})),
            "historical_profile": self._historical_profile_zh(payload.get("historical_profile", {})),
            "venue_recommendations": self._venue_recommendations_zh(payload.get("venue_recommendations", {})),
            "paper_qa_gate": self._paper_qa_gate_zh(payload.get("paper_qa_gate", {})),
            "submission_readiness": self._submission_readiness_zh(payload.get("submission_readiness", {})),
            "decision_policy_used": payload.get("decision_policy_used", {}),
            "stage_strategy_runtime": payload.get("stage_strategy_runtime", {}),
            "top_risks": [self._risk_zh(r) for r in payload["top_risks"]],
            "top_remediation_tasks": [self._task_zh(t) for t in payload["top_remediation_tasks"]],
            "predicted_reviewer_questions": self._reviewer_questions_zh(
                payload.get("predicted_reviewer_questions", [])
            ),
        }

    def _diagnosis_json_zh(self, payload: dict) -> dict:
        items_zh = []
        for item in payload.get("items", []):
            if not isinstance(item, dict):
                continue
            items_zh.append(
                {
                    "issue_id": item.get("issue_id", ""),
                    "severity": item.get("severity", ""),
                    "issue": self._phrase_zh(str(item.get("issue", ""))),
                    "why_happened": self._phrase_zh(str(item.get("why_happened", ""))),
                    "why_it_matters": self._phrase_zh(str(item.get("why_it_matters", ""))),
                    "suggested_fix": self._phrase_zh(str(item.get("suggested_fix", ""))),
                    "linked_rebuttal_review_id": item.get("linked_rebuttal_review_id", ""),
                    "linked_rebuttal_concern": self._phrase_zh(str(item.get("linked_rebuttal_concern", ""))),
                    "evidence_anchor": self._phrase_zh(str(item.get("evidence_anchor", ""))),
                }
            )
        return {
            "decision": self._decision_zh(str(payload.get("decision", ""))),
            "summary": payload.get("summary", {}),
            "items": items_zh,
        }

    def _full_json_zh(self, payload: dict, ctx: PipelineContext) -> dict:
        stage_strategy = payload.get("stage_strategy", {})
        stage_strategy_zh = dict(stage_strategy) if isinstance(stage_strategy, dict) else {}
        if stage_strategy_zh:
            stage_strategy_zh["focus"] = self._phrase_zh(str(stage_strategy_zh.get("focus", "")))
            stage_strategy_zh["risk_heading_zh"] = stage_strategy_zh.get("risk_heading_zh", stage_strategy_zh.get("risk_heading_en"))
            stage_strategy_zh["task_heading_zh"] = stage_strategy_zh.get("task_heading_zh", stage_strategy_zh.get("task_heading_en"))
        return {
            "decision": self._decision_zh(payload["decision"]),
            "manuscript_stage": self._stage_zh(str(payload.get("manuscript_stage", "initial_submission"))),
            "reviewer_comments_count": payload.get("reviewer_comments_count", 0),
            "stage_strategy": stage_strategy_zh,
            "decision_interpretation": self._phrase_zh(str(payload.get("decision_interpretation", ""))),
            "scores": payload["scores"],
            "score_leverage_analysis": self._score_leverage_analysis_zh(payload.get("score_leverage_analysis", {})),
            "score_explanations": self._score_explanations_zh(payload.get("score_explanations", {})),
            "historical_profile": self._historical_profile_zh(payload.get("historical_profile", {})),
            "venue_recommendations": self._venue_recommendations_zh(payload.get("venue_recommendations", {})),
            "paper_qa_gate": self._paper_qa_gate_zh(payload.get("paper_qa_gate", {})),
            "submission_readiness": self._submission_readiness_zh(payload.get("submission_readiness", {})),
            "decision_policy_used": payload.get("decision_policy_used", {}),
            "stage_strategy_runtime": payload.get("stage_strategy_runtime", {}),
            "all_risks": [self._risk_zh(r) for r in payload["all_risks"]],
            "focus_risks": [self._risk_zh(r) for r in payload.get("focus_risks", [])],
            "claim_evidence_matrix": [self._alignment_zh(r) for r in payload["claim_evidence_matrix"]],
            "reviewer_question_simulation": {
                **(payload.get("reviewer_question_simulation", {}) if isinstance(payload.get("reviewer_question_simulation", {}), dict) else {}),
                "questions": self._reviewer_questions_zh(
                    (payload.get("reviewer_question_simulation", {}) if isinstance(payload.get("reviewer_question_simulation", {}), dict) else {}).get("questions", [])
                ),
            },
            "remediation_tasks": [self._task_zh(t) for t in payload["remediation_tasks"]],
            "rebuttal_plan": self._rebuttal_plan_zh(payload.get("rebuttal_plan", {})),
            "rebuttal": ctx.artifacts["rebuttal"]["zh"]["bundle"],
        }

    def _risk_zh(self, risk: dict) -> dict:
        row = dict(risk)
        row["reason"] = self._phrase_zh(row["reason"])
        row["likely_reject_phrase"] = self._phrase_zh(row["likely_reject_phrase"])
        row["fix_hint"] = self._phrase_zh(row["fix_hint"])
        return row

    def _task_zh(self, task: dict) -> dict:
        row = dict(task)
        row["title"] = self._task_title_zh(row["title"])
        row["expected_gain"] = self._phrase_zh(row.get("expected_gain", ""))
        row["protocol"] = [self._phrase_zh(x) for x in row.get("protocol", [])]
        if row.get("priority") == "high":
            row["priority"] = "高"
        elif row.get("priority") == "medium":
            row["priority"] = "中"
        return row

    def _alignment_zh(self, row: dict) -> dict:
        out = dict(row)
        strength_map = {
            "Strong": "强",
            "Medium": "中",
            "Weak": "弱",
            "None": "无",
        }
        out["strength"] = strength_map.get(out.get("strength", ""), out.get("strength", ""))
        return out

    def _reviewer_questions_zh(self, rows: object) -> list[dict]:
        out: list[dict] = []
        if not isinstance(rows, list):
            return out
        persona_map = {
            "systems": "系统审稿人",
            "theory": "理论审稿人",
            "empirical": "实验审稿人",
            "reproducibility": "可复现性审稿人",
            "meta-review": "Meta 审稿视角",
            "critical": "严苛审稿视角",
            "db-systems": "数据库系统审稿人",
            "related-work": "相关工作审稿人",
            "method": "方法审稿人",
        }
        for row in rows:
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "id": row.get("id"),
                    "priority": row.get("priority"),
                    "reviewer_persona": persona_map.get(str(row.get("reviewer_persona", "")), row.get("reviewer_persona")),
                    "question": self._phrase_zh(str(row.get("question", ""))),
                    "why_this_will_be_asked": self._phrase_zh(str(row.get("why_this_will_be_asked", ""))),
                    "trigger_gap_codes": row.get("trigger_gap_codes", []),
                    "linked_risk_ids": row.get("linked_risk_ids", []),
                    "evidence_to_prepare": [
                        self._phrase_zh(str(x))
                        for x in row.get("evidence_to_prepare", [])
                        if str(x).strip()
                    ],
                    "suggested_response_strategy": self._phrase_zh(str(row.get("suggested_response_strategy", ""))),
                }
            )
        return out

    def _score_explanations_zh(self, payload: dict) -> dict:
        out: dict[str, dict] = {}
        if not isinstance(payload, dict):
            return out
        for axis in ("novelty", "soundness", "experiment", "clarity"):
            item = payload.get(axis, {})
            if not isinstance(item, dict):
                continue
            out[axis] = {
                "score": item.get("score", 0.0),
                "reasoning": self._phrase_zh(str(item.get("reasoning", ""))),
                "signals": item.get("signals", []),
            }
        return out

    def _score_leverage_analysis_zh(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {}
        rows = payload.get("axes", [])
        rows_zh: list[dict] = []
        axis_map = {
            "novelty": "新颖性",
            "soundness": "技术正确性",
            "experiment": "实验充分性",
            "clarity": "写作清晰度",
        }
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                axis = str(row.get("axis", ""))
                row_zh = dict(row)
                row_zh["axis"] = axis_map.get(axis, axis)
                rows_zh.append(row_zh)
        fastest_axis = str(payload.get("fastest_axis", ""))
        return {
            "target_axis_score": payload.get("target_axis_score"),
            "fastest_axis": axis_map.get(fastest_axis, "无" if fastest_axis == "none" else fastest_axis),
            "rationale": self._phrase_zh(str(payload.get("rationale", ""))),
            "no_urgent_axis": bool(payload.get("no_urgent_axis", False)),
            "axes": rows_zh,
        }

    def _historical_profile_zh(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {}

        def _convert_top(items: object) -> list[dict]:
            out_items: list[dict] = []
            if not isinstance(items, list):
                return out_items
            for row in items:
                if not isinstance(row, dict):
                    continue
                out_items.append(
                    {
                        "name": self._phrase_zh(str(row.get("name", ""))),
                        "count": int(row.get("count", 0) or 0),
                    }
                )
            return out_items

        author = payload.get("author_profile", {}) if isinstance(payload.get("author_profile"), dict) else {}
        venue = payload.get("venue_year_profile", {}) if isinstance(payload.get("venue_year_profile"), dict) else {}
        return {
            "available": bool(payload.get("available", False)),
            "author_hash": payload.get("author_hash"),
            "author_profile": {
                "runs": int(author.get("runs", 0) or 0),
                "top_weaknesses": _convert_top(author.get("top_weaknesses", [])),
            }
            if author
            else None,
            "venue_year_profile": {
                "runs": int(venue.get("runs", 0) or 0),
                "top_weaknesses": _convert_top(venue.get("top_weaknesses", [])),
            }
            if venue
            else None,
        }

    def _venue_recommendations_zh(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {}
        rows = payload.get("recommended_venues", [])
        out_rows: list[dict] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                reasons = row.get("reasons", [])
                out_rows.append(
                    {
                        "venue": row.get("venue"),
                        "year": row.get("year"),
                        "match_score": row.get("match_score"),
                        "readiness_score": row.get("readiness_score"),
                        "topic_overlap_score": row.get("topic_overlap_score"),
                        "system_bias_score": row.get("system_bias_score"),
                        "reasons": [
                            self._phrase_zh(str(r)) for r in reasons[:6]
                        ]
                        if isinstance(reasons, list)
                        else [],
                        "passed_checks": row.get("passed_checks", []),
                        "failed_checks": row.get("failed_checks", []),
                        "profile_source": row.get("profile_source"),
                    }
                )
        return {
            "method": payload.get("method"),
            "target_year": payload.get("target_year"),
            "candidate_count": payload.get("candidate_count"),
            "recommended_venues": out_rows,
        }

    def _paper_qa_gate_zh(self, payload: dict) -> dict:
        if not isinstance(payload, dict) or not payload:
            return {}

        source_map = {
            "executor": "agent自审",
            "heuristic_fallback": "启发式回退",
            "skip_missing_rebuttal": "跳过（缺少rebuttal）",
            "skip_invalid_rebuttal": "跳过（rebuttal格式无效）",
        }
        verdict_map = {"pass": "通过", "fail": "未通过"}

        out = {
            "accepted": bool(payload.get("accepted", True)),
            "initial_accept": bool(payload.get("initial_accept", payload.get("accepted", True))),
            "source": source_map.get(str(payload.get("source", "")), payload.get("source")),
            "issues": [
                self._phrase_zh(str(x))
                for x in payload.get("issues", [])
                if str(x).strip()
            ]
            if isinstance(payload.get("issues", []), list)
            else [],
            "rewrites_applied": int(payload.get("rewrites_applied", 0) or 0),
            "post_recheck_accept": payload.get("post_recheck_accept"),
            "post_recheck_issues": [
                self._phrase_zh(str(x))
                for x in payload.get("post_recheck_issues", [])
                if str(x).strip()
            ]
            if isinstance(payload.get("post_recheck_issues", []), list)
            else [],
            "initial_issues": [
                self._phrase_zh(str(x))
                for x in payload.get("initial_issues", [])
                if str(x).strip()
            ]
            if isinstance(payload.get("initial_issues", []), list)
            else [],
            "per_item": [],
        }

        per_item = payload.get("per_item", [])
        if isinstance(per_item, list):
            for item in per_item:
                if not isinstance(item, dict):
                    continue
                out["per_item"].append(
                    {
                        "review_id": item.get("review_id", ""),
                        "verdict": verdict_map.get(str(item.get("verdict", "")), item.get("verdict")),
                        "issues": [
                            self._phrase_zh(str(x))
                            for x in item.get("issues", [])
                            if str(x).strip()
                        ]
                        if isinstance(item.get("issues", []), list)
                        else [],
                    }
                )
        return out

    def _submission_readiness_zh(self, payload: dict) -> dict:
        if not isinstance(payload, dict) or not payload:
            return {}
        status_map = {"pass": "通过", "warning": "预警", "critical": "阻断"}
        category_map = {
            "venue_fit": "会议匹配",
            "evidence_alignment": "证据对齐",
            "technical_checks": "技术检查",
            "input_quality": "输入质量",
            "rebuttal_quality": "rebuttal质量",
        }
        title_map = {
            "Venue Scope Fit": "会议范围匹配",
            "Claim-Evidence Alignment": "主张与证据一致性",
            "Baseline Coverage": "基线覆盖度",
            "Statistical Significance": "统计显著性",
            "Ablation Completeness": "消融完整性",
            "Reproducibility Details": "可复现细节",
            "Related Work Coverage": "相关工作覆盖",
            "Paper Parsing Quality": "论文解析质量",
            "Rebuttal Readiness": "rebuttal就绪度",
        }
        reason_map = {
            "critical_checks_present": "存在阻断项",
            "multiple_warning_checks": "存在多项预警检查",
            "input_parser_quality_warning": "输入解析质量存在预警",
        }
        phrase_map = {
            "Detected P1 claim-evidence weakness that can reduce reviewer confidence.": "检测到 P1 级主张-证据薄弱点，可能降低审稿人信心。",
            "At least one P0 risk indicates core claim evidence is not sufficiently grounded.": "至少一个 P0 风险表明核心主张证据不足。",
            "No major claim-evidence blocker detected.": "未检测到主张-证据阻断项。",
            "Detected gap(s): missing_significance": "检测到缺口：missing_significance（统计显著性不足）。",
            "No explicit gap detected.": "未检测到显式缺口。",
            "Parser quality warnings detected.": "检测到解析质量预警。",
            "No parser quality warning detected.": "未检测到解析质量预警。",
            "Rebuttal needed rewrite to pass self-review.": "rebuttal 需重写后才通过自审。",
            "Rebuttal quality gate passed on first attempt.": "rebuttal 质量门一次通过。",
            "Self-review gate did not pass after rewrite.": "重写后仍未通过自审门。",
            "Submission is possible, but close WARNING checks to reduce rejection risk.": "可以提交，但建议先关闭 WARNING 项以降低拒稿风险。",
            "Hold submission and close all CRITICAL checks first.": "建议暂缓提交，先关闭所有 CRITICAL 阻断项。",
            "Checklist passed. Final polish and consistency review before submission.": "清单通过，提交前做最后一致性检查即可。",
            "Prioritize reviewer-critical fixes and rebuttal evidence before next discussion round.": "下一轮讨论前优先修复审稿人关键问题并补齐 rebuttal 证据。",
            "Focus on high-impact reviewer concerns and close remaining warning checks.": "聚焦高影响审稿意见并关闭剩余 warning 检查项。",
            "Use this as a stable rebuttal baseline and iterate with reviewer feedback.": "以该版本作为稳定 rebuttal 基线，结合审稿反馈迭代。",
            "Add direct evidence blocks for weak claims and tighten claim wording scope.": "为薄弱主张补充直接证据块，并收紧主张措辞范围。",
            "Report multi-seed mean/std, confidence intervals, and paired tests.": "补充多种子 mean/std、置信区间与配对检验。",
            "Ensure each response contains concrete numbers and anchor locations.": "确保每条回应都包含具体数字与锚点位置。",
            "Re-evaluate venue choice or strengthen venue-specific positioning paragraph.": "重新评估投稿会议，或强化会议定制化定位段落。",
            "Add matched-setting comparison against strongest baselines.": "补充与最强基线的同设置对比。",
            "Add component and interaction ablations for key modules.": "补充关键模块的组件级与交互级消融。",
            "Add full environment/hyperparameter/seed/rerun instructions.": "补充完整环境、超参、随机种子与复现实验指令。",
            "Add nearest-neighbor comparison table and top-venue references.": "补充近邻工作对比表与顶会参考文献。",
            "Provide a machine-readable PDF or markdown source if parser quality is low.": "若解析质量偏低，请提供可机器读取的 PDF 或 Markdown 源稿。",
        }

        def z(text: str) -> str:
            clean = str(text or "").strip()
            if not clean:
                return ""
            if clean in phrase_map:
                return phrase_map[clean]
            translated = self._phrase_zh(clean)
            if ReportBuilderStep._is_mojibake(translated):
                return clean
            return translated

        out_checks: list[dict] = []
        checks = payload.get("checks", [])
        if isinstance(checks, list):
            for row in checks:
                if not isinstance(row, dict):
                    continue
                out_checks.append(
                    {
                        "id": row.get("id"),
                        "category": category_map.get(str(row.get("category", "")), z(str(row.get("category", "")))),
                        "title": title_map.get(str(row.get("title", "")), z(str(row.get("title", "")))),
                        "status": status_map.get(str(row.get("status", "")), row.get("status")),
                        "why": z(str(row.get("why", ""))),
                        "action": z(str(row.get("action", ""))),
                        "evidence": z(str(row.get("evidence", ""))),
                    }
                )
        return {
            "overall_status": status_map.get(str(payload.get("overall_status", "")), payload.get("overall_status")),
            "pass_count": int(payload.get("pass_count", 0) or 0),
            "warning_count": int(payload.get("warning_count", 0) or 0),
            "critical_count": int(payload.get("critical_count", 0) or 0),
            "recommended_next_action": z(str(payload.get("recommended_next_action", ""))),
            "human_review_recommended": bool(payload.get("human_review_recommended", False)),
            "human_review_reasons": [
                reason_map.get(str(x), z(str(x)))
                for x in payload.get("human_review_reasons", [])
                if str(x).strip()
            ]
            if isinstance(payload.get("human_review_reasons", []), list)
            else [],
            "checks": out_checks,
        }

    @staticmethod
    def _is_mojibake(text: str) -> bool:
        if not text:
            return False
        if "�" in text or "锟" in text:
            return True
        pua = sum(1 for ch in text if 0xE000 <= ord(ch) <= 0xF8FF)
        return pua >= 1

    def _rebuttal_plan_zh(self, payload: dict) -> dict:
        if not isinstance(payload, dict) or not payload:
            return {}
        out_items: list[dict] = []
        items = payload.get("plan_items", payload.get("items", []))
        if isinstance(items, list):
            for row in items:
                if not isinstance(row, dict):
                    continue
                out_items.append(
                    {
                        "review_id": row.get("review_id"),
                        "concern": self._phrase_zh(str(row.get("concern", ""))),
                        "risk_id": row.get("risk_id"),
                        "risk_reason": self._phrase_zh(str(row.get("risk_reason", ""))),
                        "linked_remediation_tasks": row.get("linked_remediation_tasks", []),
                        "evidence_targets": [
                            self._phrase_zh(str(x))
                            for x in row.get("evidence_targets", [])
                            if str(x).strip()
                        ]
                        if isinstance(row.get("evidence_targets", []), list)
                        else [],
                        "predicted_followup_questions": [
                            self._phrase_zh(str(x))
                            for x in row.get("predicted_followup_questions", [])
                            if str(x).strip()
                        ]
                        if isinstance(row.get("predicted_followup_questions", []), list)
                        else [],
                    }
                )
        status_map = {"pass": "通过", "warning": "预警", "fail": "失败"}
        audit_rows: list[dict] = []
        raw_audit = payload.get("post_generation_audit", [])
        if isinstance(raw_audit, list):
            for row in raw_audit:
                if not isinstance(row, dict):
                    continue
                audit_rows.append(
                    {
                        "review_id": row.get("review_id"),
                        "risk_id": row.get("risk_id"),
                        "status": status_map.get(str(row.get("status", "")), row.get("status")),
                        "metrics": row.get("metrics", {}),
                        "gaps": [self._phrase_zh(str(x)) for x in row.get("gaps", []) if str(x).strip()]
                        if isinstance(row.get("gaps", []), list)
                        else [],
                        "actions": [self._phrase_zh(str(x)) for x in row.get("actions", []) if str(x).strip()]
                        if isinstance(row.get("actions", []), list)
                        else [],
                    }
                )
        summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {}
        return {
            "manuscript_stage": self._stage_zh(str(payload.get("manuscript_stage", ""))),
            "plan_items": out_items,
            "post_generation_audit": audit_rows,
            "summary": {
                "total_items": int(summary.get("total_items", 0) or 0),
                "pass_count": int(summary.get("pass_count", 0) or 0),
                "warning_count": int(summary.get("warning_count", 0) or 0),
                "fail_count": int(summary.get("fail_count", 0) or 0),
                "manual_review_recommended": bool(summary.get("manual_review_recommended", False)),
            },
        }

    @staticmethod
    def _decision_zh(value: str) -> str:
        mapping = {
            "Not Ready": "不建议投稿",
            "Borderline": "边界状态",
            "Ready": "可投稿",
            "Major Revision Required": "需要大修后再考虑复投",
            "Resubmission Candidate": "具备复投潜力",
            "Ready for Resubmission": "可进入复投",
            "Weak Discussion Position": "讨论期处于弱势",
            "Recoverable in Discussion": "讨论期可挽回",
            "Strong Discussion Position": "讨论期处于有利位置",
        }
        return mapping.get(value, value)

    @staticmethod
    def _stage_zh(value: str) -> str:
        mapping = {
            "initial_submission": "初次投稿前",
            "rejected_after_reviews": "已被拒稿，准备复投",
            "meta_review_discussion": "审稿讨论/Meta Review 阶段",
        }
        return mapping.get(value, value)

    def _phrase_zh(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        mapping = {
            "Statistical significance evidence appears missing.": "统计显著性证据可能缺失。",
            "Reproducibility details are likely incomplete.": "可复现性细节可能不完整。",
            "Experimental evidence does not yet meet venue expectations.": "实验性证据尚未达到目标会议预期。",
            "Core claims are not sufficiently supported by rigorous evidence.": "核心主张尚未得到严格证据的充分支撑。",
            "Address this with a focused experiment or analysis update.": "通过补充针对性实验或分析更新来解决该问题。",
            "Add direct experiments and statistical validation tied to this claim.": "补充与该主张直接对应的实验和统计验证。",
            "Reduce rejection likelihood by strengthening claim-evidence linkage.": "通过强化主张与证据链条，降低拒稿风险。",
            "Define exact hypothesis and target claim.": "明确待验证假设和目标主张。",
            "Run comparison against strong baselines with identical settings.": "在相同设置下与强基线进行对比实验。",
            "Report mean/std over multiple seeds and significance tests.": "报告多随机种子的均值/方差及显著性检验结果。",
            "Add analysis of failures and limitations.": "补充失败案例分析与局限性讨论。",
            "Citation coverage appears shallow; references may be insufficient to position novelty and baselines.": "引用覆盖度偏浅，参考文献可能不足以支撑新颖性与基线定位。",
            "Related work appears to under-cover recent top-venue papers relevant to this topic.": "相关工作对近期顶会论文覆盖不足，可能影响定位说服力。",
            "One or more key claims have weak evidence alignment and need direct support.": "一项或多项关键主张的证据对齐偏弱，需要直接支撑。",
            "Reviewers typically reduce confidence when claim, evidence, and reporting are not tightly linked.": "当主张、证据与报告链条不够紧密时，审稿人通常会下调置信度。",
            "Add one targeted experiment, one statistical/significance validation block, and explicit section-level paper changes for this concern.": "针对该问题补充一组定向实验、一组统计显著性验证，并在论文中明确标注对应修改位置。",
            "Mitigate": "缓解",
        }
        if text in mapping:
            return mapping[text]
        no_period = text.rstrip(".")
        if no_period in mapping:
            return mapping[no_period]

        # Structured risk/task templates.
        m = re.match(r"^Claim (C\d+) has Weak evidence support\.$", text)
        if m:
            return f"主张 {m.group(1)} 的证据支持较弱。"
        m2 = re.match(r"^Claim (C\d+) has (Strong|Medium|Weak|None) evidence support\.?$", text, flags=re.IGNORECASE)
        if m2:
            level = m2.group(2).lower()
            level_map = {"strong": "强", "medium": "中", "weak": "弱", "none": "无"}
            return f"主张 {m2.group(1)} 的证据支持等级为{level_map.get(level, level)}。"
        m3 = re.match(r"^Mitigate (RISK-\d+) - (P\d) risk$", text)
        if m3:
            return f"缓解 {m3.group(1)} - {m3.group(2)} 风险"

        m4 = re.match(
            r"^The current draft likely lacks a direct experiment-or-analysis block mapping one-to-one to this concern:\s*(.+)$",
            text,
        )
        if m4:
            return f"当前稿件很可能缺少与该问题一一对应的实验或分析支撑：{self._phrase_zh(m4.group(1).strip())}"

        # Stable translator fallback (already includes glossary and robust degradation).
        return self.translator.to_zh(text)

    def _task_title_zh(self, title: str) -> str:
        m = re.match(r"^Mitigate (RISK-\d+) - (P\d) risk$", title)
        if m:
            return f"缓解 {m.group(1)} - {m.group(2)} 风险"
        return self._phrase_zh(title)

    @staticmethod
    def _decision_md_zh(payload: dict) -> str:
        stage = payload.get("stage_strategy", {}) if isinstance(payload.get("stage_strategy"), dict) else {}
        risk_heading = str(stage.get("risk_heading_zh") or "拒稿风险 Top")
        task_heading = str(stage.get("task_heading_zh") or "必补实验")
        leverage = payload.get("score_leverage_analysis", {})
        lines = [
            "# 投稿决策简报",
            "",
            f"决策：**{payload['decision']}**",
            "",
            f"稿件阶段：`{payload.get('manuscript_stage', '初次投稿前')}`",
            f"阶段焦点：{stage.get('focus', '')}",
            f"决策含义：{payload.get('decision_interpretation', '')}",
            "",
            "## 评分",
            f"- 新颖性：{payload['scores']['novelty']}",
            f"- 技术正确性：{payload['scores']['soundness']}",
            f"- 实验充分性：{payload['scores']['experiment']}",
            f"- 写作清晰度：{payload['scores']['clarity']}",
            f"- 总分：{payload['scores']['overall']}",
            "",
            "## 评分解释",
        ]
        for axis, label in (
            ("novelty", "新颖性"),
            ("soundness", "技术正确性"),
            ("experiment", "实验充分性"),
            ("clarity", "写作清晰度"),
        ):
            detail = payload.get("score_explanations", {}).get(axis, {})
            reasoning = str(detail.get("reasoning", "")).strip()
            lines.append(f"- {label}：{payload['scores'][axis]} | 解释：{reasoning}")

        gate = payload.get("paper_qa_gate", {})
        if isinstance(gate, dict) and gate:
            lines.extend(
                [
                    "",
                    "## Rebuttal 自审门控",
                    f"- 是否通过：{gate.get('accepted', True)}",
                    f"- 来源：{gate.get('source', 'n/a')}",
                    f"- 自动重写次数：{gate.get('rewrites_applied', 0)}",
                ]
            )
            gate_issues = gate.get("issues", [])
            if isinstance(gate_issues, list) and gate_issues:
                lines.append(f"- 发现问题：{', '.join(str(x) for x in gate_issues[:4])}")

        lines.extend(ReportBuilderStep._submission_readiness_md_lines_zh(payload.get("submission_readiness", {})))
        lines.extend(ReportBuilderStep._score_leverage_md_lines_zh(leverage))
        lines.extend(ReportBuilderStep._historical_md_lines_zh(payload.get("historical_profile", {})))
        lines.extend(ReportBuilderStep._venue_reco_md_lines_zh(payload.get("venue_recommendations", {})))
        lines.extend(
            [
                "",
            "## 决策阈值策略",
            f"- 策略层级：{payload['decision_policy_used'].get('strictness_tier', 'default')}",
            f"- 进入不建议投稿的 P1 数阈值：{payload['decision_policy_used'].get('p1_not_ready_threshold')}",
            f"- 进入边界状态的 P1 数阈值：{payload['decision_policy_used'].get('p1_borderline_threshold')}",
            f"- 可投稿最低总分：{payload['decision_policy_used'].get('min_overall_ready')}",
            "",
            f"## {risk_heading}",
            ]
        )
        for risk in payload["top_risks"]:
            lines.append(f"- [{risk['severity']}] {risk['id']} ({risk['score']}): {risk['reason']}")

        lines.append(f"\n## {task_heading}")
        for task in payload["top_remediation_tasks"]:
            lines.append(
                f"- {task['id']} ({task['priority']}, effort={task['effort']}): {task['title']}"
            )

        followups = payload.get("predicted_reviewer_questions", [])
        if isinstance(followups, list) and followups:
            lines.append("\n## 可能的审稿人追问")
            for row in followups[:3]:
                if not isinstance(row, dict):
                    continue
                lines.append(f"- [{row.get('priority', 'medium')}] {row.get('question', '')}")
                why = str(row.get("why_this_will_be_asked", "")).strip()
                if why:
                    lines.append(f"  - 原因：{why}")

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _diagnosis_md_zh(payload: dict) -> str:
        lines = [
            "# 详细诊断报告",
            "",
            f"决策快照：**{payload.get('decision', 'N/A')}**",
            "",
            "## 汇总",
            f"- 问题总数：{payload.get('summary', {}).get('risk_count', 0)}",
            f"- P0 问题数：{payload.get('summary', {}).get('p0_count', 0)}",
            f"- P1 问题数：{payload.get('summary', {}).get('p1_count', 0)}",
            "",
            "## 逐项问题诊断",
        ]
        for item in payload.get("items", []):
            lines.extend(
                [
                    f"### {item.get('issue_id', 'RISK')} [{item.get('severity', 'P2')}]",
                    f"- 存在问题：{item.get('issue', '')}",
                    f"- 出现原因：{item.get('why_happened', '')}",
                    f"- 影响解释：{item.get('why_it_matters', '')}",
                    f"- 修复建议：{item.get('suggested_fix', '')}",
                    f"- 证据锚点：{item.get('evidence_anchor', '')}",
                    (
                        f"- Rebuttal 对应：Reviewer {item.get('linked_rebuttal_review_id', '')} "
                        f"-> {item.get('linked_rebuttal_concern', '')}"
                    ),
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _full_md_zh(payload: dict) -> str:
        stage = payload.get("stage_strategy", {}) if isinstance(payload.get("stage_strategy"), dict) else {}
        risk_heading = str(stage.get("risk_heading_zh") or "详细风险")
        leverage = payload.get("score_leverage_analysis", {})
        lines = [
            "# 完整评审报告",
            "",
            f"决策：**{payload['decision']}**",
            "",
            f"稿件阶段：`{payload.get('manuscript_stage', '初次投稿前')}`",
            f"阶段焦点：{stage.get('focus', '')}",
            f"决策含义：{payload.get('decision_interpretation', '')}",
            "",
            "## 评分解释",
        ]
        for axis, label in (
            ("novelty", "新颖性"),
            ("soundness", "技术正确性"),
            ("experiment", "实验充分性"),
            ("clarity", "写作清晰度"),
        ):
            detail = payload.get("score_explanations", {}).get(axis, {})
            reasoning = str(detail.get("reasoning", "")).strip()
            lines.append(f"- {label}：{payload['scores'][axis]} | 解释：{reasoning}")

        gate = payload.get("paper_qa_gate", {})
        if isinstance(gate, dict) and gate:
            lines.extend(
                [
                    "",
                    "## Rebuttal 自审门控",
                    f"- 是否通过：{gate.get('accepted', True)}",
                    f"- 来源：{gate.get('source', 'n/a')}",
                    f"- 自动重写次数：{gate.get('rewrites_applied', 0)}",
                ]
            )
            gate_issues = gate.get("issues", [])
            if isinstance(gate_issues, list) and gate_issues:
                lines.append(f"- 发现问题：{', '.join(str(x) for x in gate_issues[:6])}")

        lines.extend(ReportBuilderStep._submission_readiness_md_lines_zh(payload.get("submission_readiness", {})))
        lines.extend(ReportBuilderStep._score_leverage_md_lines_zh(leverage))
        lines.extend(ReportBuilderStep._historical_md_lines_zh(payload.get("historical_profile", {})))
        lines.extend(ReportBuilderStep._venue_reco_md_lines_zh(payload.get("venue_recommendations", {})))
        lines.extend(
            [
                "",
            f"## {risk_heading}",
            ]
        )
        focus = payload.get("focus_risks", payload.get("all_risks", []))
        if not isinstance(focus, list):
            focus = payload.get("all_risks", [])
        for risk in focus:
            lines.extend(
                [
                    f"### {risk['id']} [{risk['severity']}]",
                    f"- 分数：{risk['score']}",
                    f"- 原因：{risk['reason']}",
                    f"- 可能拒稿话术：{risk['likely_reject_phrase']}",
                    f"- 建议修复：{risk['fix_hint']}",
                    "",
                ]
            )

        lines.append("## 主张-证据对齐")
        lines.append("- 回溯提示：使用 `artifacts/evidence_index.json -> passage_locator[passage_id]` 查看章节与页码来源。")
        for row in payload["claim_evidence_matrix"]:
            diagnostics = row.get("diagnostics", {}) if isinstance(row, dict) else {}
            sections = diagnostics.get("selected_sections", []) if isinstance(diagnostics, dict) else []
            avg_quality = diagnostics.get("avg_quality", 0.0) if isinstance(diagnostics, dict) else 0.0
            section_text = ", ".join(str(x) for x in sections[:3]) if isinstance(sections, list) else ""
            contradiction = float(row.get("contradiction_score", 0.0) or 0.0)
            contradiction_flag = "是" if bool(row.get("contradiction_detected")) else "否"
            contradiction_refs = row.get("contradictory_evidence_refs", [])
            contradiction_anchor = ""
            if isinstance(contradiction_refs, list) and contradiction_refs:
                first = contradiction_refs[0]
                if isinstance(first, dict):
                    contradiction_anchor = f"{first.get('section', '')}/{first.get('passage_id', '')}".strip("/")
            lines.append(
                f"- {row['claim_id']} [{row['strength']}] score={row['score']} -> {len(row['evidence_refs'])} 条证据；"
                f"锚点章节={section_text or 'n/a'}；平均质量={avg_quality}；"
                f"反证={contradiction_flag}（{contradiction}）；"
                f"反证锚点={contradiction_anchor or 'n/a'}"
            )

        plan = payload.get("rebuttal_plan", {})
        plan_items = []
        plan_audit = []
        plan_summary = {}
        if isinstance(plan, dict):
            raw_items = plan.get("plan_items", plan.get("items", []))
            if isinstance(raw_items, list):
                plan_items = raw_items
            if isinstance(plan.get("post_generation_audit", []), list):
                plan_audit = plan.get("post_generation_audit", [])
            if isinstance(plan.get("summary", {}), dict):
                plan_summary = plan.get("summary", {})
        if isinstance(plan_items, list) and plan_items:
            lines.append("\n## Rebuttal 计划层（起草前）")
            for row in plan_items[:5]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"- {row.get('review_id', 'R?')} -> 风险 {row.get('risk_id', 'n/a')}："
                    f"{row.get('concern', '')}"
                )
                evid = row.get("evidence_targets", [])
                if isinstance(evid, list) and evid:
                    lines.append(f"  - 证据目标：{evid[0]}")
        if isinstance(plan_summary, dict) and plan_summary:
            lines.append("\n## Rebuttal 计划审计")
            lines.append(
                "- 汇总："
                + f"通过={plan_summary.get('pass_count', 0)}，"
                + f"预警={plan_summary.get('warning_count', 0)}，"
                + f"失败={plan_summary.get('fail_count', 0)}，"
                + f"建议人工复核={plan_summary.get('manual_review_recommended', False)}"
            )
            for row in plan_audit[:5]:
                if not isinstance(row, dict):
                    continue
                gaps = [str(x) for x in row.get("gaps", []) if str(x).strip()]
                gap_text = "，".join(gaps[:2]) if gaps else "无待处理缺口"
                lines.append(
                    f"- {row.get('review_id', 'R?')} [{row.get('status', 'warning')}] {gap_text}"
                )

        lines.append("\n## Rebuttal 草稿")
        lines.append("请查看 `rebuttal.*` 产物。")

        sim = payload.get("reviewer_question_simulation", {})
        rows = sim.get("questions", []) if isinstance(sim, dict) else []
        if isinstance(rows, list) and rows:
            lines.append("\n## 预测性审稿追问")
            for row in rows[:8]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"- [{row.get('priority', 'medium')}] {row.get('question', '')} "
                    f"(视角={row.get('reviewer_persona', 'empirical')})"
                )
                why = str(row.get("why_this_will_be_asked", "")).strip()
                if why:
                    lines.append(f"  - 原因：{why}")
                evid = row.get("evidence_to_prepare", [])
                if isinstance(evid, list) and evid:
                    lines.append(f"  - 建议准备证据：{evid[0]}")

        return "\n".join(lines).strip() + "\n"
