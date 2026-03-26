from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import RunStatus
from ..services.feedback_store import build_feedback_template
from ..services.pdf_export import export_markdown_to_pdf
from .base import PipelineContext, PipelineStep


class ExporterAndQAGateStep(PipelineStep):
    name = "ExporterAndQAGate"
    _PDF_EXPORT_PREFIXES = (
        "START_HERE",
        "student_pack/",
        "decision_brief.",
        "full_review.",
        "diagnosis_report.",
        "rebuttal.",
    )

    def run(self, ctx: PipelineContext) -> None:
        deliverables = self._collect_deliverables(ctx)
        errors = []

        for relpath, content in deliverables.items():
            path = ctx.run_dir / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            if relpath.endswith(".json"):
                ctx.dump_json(relpath, content)
            else:
                md_encoding = "utf-8-sig" if relpath.endswith(".md") else "utf-8"
                path.write_text(str(content), encoding=md_encoding)

        if ctx.input_data.options.always_export_pdf:
            for md_path in self._pdf_export_md_paths(ctx.run_dir, deliverables):
                pdf_path = md_path.with_suffix(".pdf")
                result = export_markdown_to_pdf(md_path, pdf_path)
                if not result.ok:
                    errors.append(f"pdf_export_failed:{md_path.name}:{result.error}")

        rebuttal_en = ctx.artifacts["rebuttal"]["en"]["bundle"]
        for item in rebuttal_en["items"]:
            if item["char_count"] > item["char_limit"]:
                errors.append(f"rebuttal_overflow:{item['review_id']}")

        if ctx.input_data.options.language_mode.value == "en_zh":
            en_scores = ctx.artifacts["reports"]["en"]["decision_json"]["scores"]
            zh_scores = ctx.artifacts["reports"]["zh"]["decision_json"]["scores"]
            if en_scores != zh_scores:
                errors.append("bilingual_score_mismatch")

        if errors:
            ctx.qa_issues.extend(errors)
            ctx.status = RunStatus.PARTIAL_FAILED
            (ctx.run_dir / "export_errors.log").write_text("\n".join(errors) + "\n", encoding="utf-8")

        summary = {
            "run_id": ctx.run_id,
            "status": ctx.status.value,
            "qa_issues": ctx.qa_issues,
            "step_statuses": ctx.step_statuses,
        }
        ctx.dump_json("run_summary.json", summary)

    @classmethod
    def _pdf_export_md_paths(
        cls,
        run_dir: Path,
        deliverables: dict[str, object],
    ) -> list[Path]:
        out: list[Path] = []
        for rel in deliverables:
            normalized = str(rel).replace("\\", "/")
            if not normalized.endswith(".md"):
                continue
            if normalized.startswith(cls._PDF_EXPORT_PREFIXES):
                out.append(run_dir / rel)
        return out

    @staticmethod
    def _collect_deliverables(ctx: PipelineContext) -> dict[str, object]:
        reports = ctx.artifacts["reports"]
        rebuttal = ctx.artifacts["rebuttal"]
        risk_payload = ctx.artifacts.get("risk_ranking", {})

        feedback_template = build_feedback_template(
            run_id=ctx.run_id,
            paper_title=Path(ctx.input_data.paper.path).stem,
            venue=str(ctx.input_data.venue.name or ""),
            year=int(ctx.input_data.venue.year or 0),
            risks=risk_payload.get("risks", []),
        )
        feedback_readme = (
            "# Risk Feedback Template\n\n"
            "1. Open `feedback_template.json`.\n"
            "2. For each item, set `verdict` to `correct` or `incorrect`.\n"
            "3. Optional: set `confidence` (0.0~1.0, default 0.8).\n"
            "4. Add short notes in `comment` when `incorrect`.\n"
            "5. Submit:\n"
            "   `python -m agent_paper_reviewers.cli submit-feedback --input <path/to/feedback_template.json>`\n"
        )

        deliverables: dict[str, object] = {
            "decision_brief.en.md": reports["en"]["decision_md"],
            "decision_brief.en.json": reports["en"]["decision_json"],
            "submission_readiness.en.json": reports["en"]["decision_json"].get("submission_readiness", {}),
            "full_review.en.md": reports["en"]["full_md"],
            "full_review.en.json": reports["en"]["full_json"],
            "diagnosis_report.en.md": reports["en"]["diagnosis_md"],
            "diagnosis_report.en.json": reports["en"]["diagnosis_json"],
            "rebuttal.en.md": rebuttal["en"]["markdown"],
            "rebuttal.en.json": rebuttal["en"]["bundle"],
            "rebuttal_plan.json": ctx.artifacts.get("rebuttal_plan", {}),
            "rebuttal_precheck.json": ctx.artifacts.get("rebuttal_precheck", {}),
            "paper_qa_gate.json": ctx.artifacts.get("paper_qa_gate", {}),
            "claim_evidence_matrix.json": ctx.artifacts["claim_evidence_matrix"],
            "claim_discovery.json": ctx.artifacts.get("claim_discovery", {}),
            "reviewer_questions.json": ctx.artifacts.get("reviewer_questions", {}),
            "remediation_plan.json": ctx.artifacts["remediation_plan"],
            "venue_recommendations.json": ctx.artifacts.get("venue_recommendations", {}),
            "venue_profile_used.json": ctx.artifacts["venue_profile"],
            "skill_flow_used.json": ctx.artifacts.get("skill_flow", {}),
            "mcp_runtime.json": ctx.artifacts.get("mcp_runtime", {}),
            "feedback_template.json": feedback_template,
            "feedback_README.en.md": feedback_readme,
        }
        deliverables.update(ExporterAndQAGateStep._collect_student_pack_deliverables(ctx, reports, rebuttal))

        if ctx.input_data.options.language_mode.value == "en_zh":
            deliverables.update(
                {
                    "decision_brief.zh.md": reports["zh"]["decision_md"],
                    "decision_brief.zh.json": reports["zh"]["decision_json"],
                    "submission_readiness.zh.json": reports["zh"]["decision_json"].get("submission_readiness", {}),
                    "full_review.zh.md": reports["zh"]["full_md"],
                    "full_review.zh.json": reports["zh"]["full_json"],
                    "diagnosis_report.zh.md": reports["zh"]["diagnosis_md"],
                    "diagnosis_report.zh.json": reports["zh"]["diagnosis_json"],
                    "rebuttal.zh.md": rebuttal["zh"]["markdown"],
                    "rebuttal.zh.json": rebuttal["zh"]["bundle"],
                    "feedback_README.zh.md": (
                        "# 风险反馈模板说明\n\n"
                        "1. 打开 `feedback_template.json`。\n"
                        "2. 将每条风险的 `verdict` 设置为 `correct` 或 `incorrect`。\n"
                        "3. 可选：设置 `confidence`（0.0~1.0，默认 0.8）。\n"
                        "4. 对 `incorrect` 项补充 `comment`。\n"
                        "5. 提交命令：\n"
                        "   `python -m agent_paper_reviewers.cli submit-feedback --input <feedback_template.json路径>`\n"
                    ),
                }
            )
        return deliverables
    @staticmethod
    def _collect_student_pack_deliverables(
        ctx: PipelineContext,
        reports: dict[str, object],
        rebuttal: dict[str, object],
    ) -> dict[str, object]:
        decision_en = reports.get("en", {}).get("decision_json", {})
        diagnosis_en = reports.get("en", {}).get("diagnosis_json", {})
        rebuttal_en = rebuttal.get("en", {}).get("bundle", {})
        remediation_tasks = ctx.artifacts.get("remediation_plan", {}).get("tasks", [])
        rebuttal_plan = ctx.artifacts.get("rebuttal_plan", {})
        risk_rows = ctx.artifacts.get("risk_ranking", {}).get("risks", [])

        risk_to_review = ExporterAndQAGateStep._risk_to_review_map(rebuttal_plan)
        risk_index = {str(x.get("id", "")): x for x in risk_rows if isinstance(x, dict)}

        student_pack_agent = ctx.artifacts.get("student_pack_agent", {})
        agent_en = student_pack_agent.get("en", {}) if isinstance(student_pack_agent, dict) else {}
        agent_zh = student_pack_agent.get("zh", {}) if isinstance(student_pack_agent, dict) else {}

        has_agent_en = all(isinstance(agent_en.get(k), str) and agent_en.get(k).strip() for k in ("001", "002", "003"))
        has_agent_zh = all(isinstance(agent_zh.get(k), str) and agent_zh.get(k).strip() for k in ("001", "002", "003"))

        if has_agent_en:
            decision_en_md = str(agent_en["001"])
            action_en_md = str(agent_en["002"])
            rebuttal_en_md = str(agent_en["003"])
        else:
            decision_en_md = ExporterAndQAGateStep._student_decision_en(
                ctx=ctx,
                decision=decision_en,
                diagnosis=diagnosis_en,
                risk_index=risk_index,
                remediation_tasks=remediation_tasks,
                risk_to_review=risk_to_review,
            )
            action_en_md = ExporterAndQAGateStep._student_actions_en(
                diagnosis=diagnosis_en,
                remediation_tasks=remediation_tasks,
                risk_index=risk_index,
                risk_to_review=risk_to_review,
            )
            rebuttal_en_md = ExporterAndQAGateStep._student_rebuttal(
                bundle=rebuttal_en,
                risk_to_review=risk_to_review,
                risk_index=risk_index,
                zh=False,
            )

        deliverables: dict[str, object] = {
            "student_pack/en/001-submission-decision.md": decision_en_md,
            "student_pack/en/002-action-items.md": action_en_md,
            "student_pack/en/003-rebuttal-draft.md": rebuttal_en_md,
            "START_HERE.en.md": ExporterAndQAGateStep._start_here_en(),
        }

        if ctx.input_data.options.language_mode.value == "en_zh":
            decision_zh = reports.get("zh", {}).get("decision_json", {})
            diagnosis_zh = reports.get("zh", {}).get("diagnosis_json", {})
            rebuttal_zh = rebuttal.get("zh", {}).get("bundle", {})

            if has_agent_zh:
                decision_zh_md = str(agent_zh["001"])
                action_zh_md = str(agent_zh["002"])
                rebuttal_zh_md = str(agent_zh["003"])
            else:
                decision_zh_md = ExporterAndQAGateStep._student_decision_zh(
                    ctx=ctx,
                    decision=decision_zh,
                    diagnosis=diagnosis_zh,
                    risk_index=risk_index,
                    remediation_tasks=remediation_tasks,
                    risk_to_review=risk_to_review,
                )
                action_zh_md = ExporterAndQAGateStep._student_actions_zh(
                    diagnosis=diagnosis_zh,
                    remediation_tasks=remediation_tasks,
                    risk_index=risk_index,
                    risk_to_review=risk_to_review,
                )
                rebuttal_zh_md = ExporterAndQAGateStep._student_rebuttal(
                    bundle=rebuttal_zh,
                    risk_to_review=risk_to_review,
                    risk_index=risk_index,
                    zh=True,
                )

            deliverables.update(
                {
                    "student_pack/zh/001-submission-decision.md": decision_zh_md,
                    "student_pack/zh/002-action-items.md": action_zh_md,
                    "student_pack/zh/003-rebuttal-draft.md": rebuttal_zh_md,
                    "START_HERE.zh.md": ExporterAndQAGateStep._start_here_zh(),
                    "START_HERE.md": ExporterAndQAGateStep._start_here_bilingual(),
                }
            )
        else:
            deliverables["START_HERE.md"] = deliverables["START_HERE.en.md"]
        return deliverables


    @staticmethod
    def _risk_to_review_map(rebuttal_plan: object) -> dict[str, str]:
        out: dict[str, str] = {}
        if not isinstance(rebuttal_plan, dict):
            return out
        items = rebuttal_plan.get("plan_items", [])
        if not isinstance(items, list):
            return out
        for row in items:
            if not isinstance(row, dict):
                continue
            risk_id = str(row.get("risk_id", "")).strip()
            review_id = str(row.get("review_id", "")).strip()
            if risk_id and review_id and risk_id not in out:
                out[risk_id] = review_id
        return out

    @staticmethod
    def _top_diagnosis_rows(diagnosis: dict[str, Any], risk_index: dict[str, dict[str, Any]], n: int) -> list[dict]:
        rows = diagnosis.get("items", [])
        if not isinstance(rows, list):
            return []
        severity_order = {"P0": 0, "P1": 1, "P2": 2}
        parsed: list[tuple[tuple[int, float], dict]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("issue_id", "")).strip()
            r = risk_index.get(rid, {})
            sev = str(row.get("severity", r.get("severity", "P2"))).upper()
            score = float(r.get("score", 0.0) or 0.0)
            parsed.append(((severity_order.get(sev, 9), -score), row))
        parsed.sort(key=lambda x: x[0])
        return [x[1] for x in parsed[:n]]

    @staticmethod
    def _anchor_text(item: dict[str, Any], risk_row: dict[str, Any]) -> str:
        anchors = item.get("evidence_anchors", [])
        if isinstance(anchors, list) and anchors and isinstance(anchors[0], dict):
            row = anchors[0]
            section = str(row.get("section", "")).strip()
            passage = str(row.get("passage_id", "")).strip()
            page = int(float(row.get("page", 0) or 0))
            label = str(row.get("anchor_label", "")).strip()
            parts = [x for x in [section, f"p.{page}" if page > 0 else "", label, passage] if x]
            if parts:
                return " / ".join(parts)
        refs = risk_row.get("evidence_refs", [])
        if isinstance(refs, list) and refs and isinstance(refs[0], dict):
            row = refs[0]
            sec = str(row.get("section", "")).strip()
            pid = str(row.get("passage_id", "")).strip()
            return " / ".join(x for x in [sec, pid] if x) or "No reliable anchor."
        return str(item.get("evidence_anchor", "")).strip() or "No reliable anchor."

    @staticmethod
    def _tasks_for_risk(remediation_tasks: object, risk_id: str) -> list[dict[str, Any]]:
        if not isinstance(remediation_tasks, list):
            return []
        out = []
        for row in remediation_tasks:
            if not isinstance(row, dict):
                continue
            if str(row.get("risk_id", "")).strip() == risk_id:
                out.append(row)
        return out

    @staticmethod
    def _student_decision_en(
        *,
        ctx: PipelineContext,
        decision: dict[str, Any],
        diagnosis: dict[str, Any],
        risk_index: dict[str, dict[str, Any]],
        remediation_tasks: object,
        risk_to_review: dict[str, str],
    ) -> str:
        top = ExporterAndQAGateStep._top_diagnosis_rows(diagnosis, risk_index, 3)
        lines = [
            "# 001 Submission Decision (Student Version)",
            "",
            f"- Paper: `{Path(ctx.input_data.paper.path).stem}`",
            f"- Venue: **{ctx.input_data.venue.name} {ctx.input_data.venue.year}**",
            f"- Decision: **{decision.get('decision', 'N/A')}**",
            f"- Meaning: {decision.get('decision_interpretation', '')}",
            "",
            "## Top 3 must-fix issues",
        ]
        for i, row in enumerate(top, start=1):
            rid = str(row.get("issue_id", f"RISK-{i:03d}"))
            risk_row = risk_index.get(rid, {})
            tasks = ExporterAndQAGateStep._tasks_for_risk(remediation_tasks, rid)
            first_fix = ""
            if isinstance(row.get("fix_actions", []), list) and row.get("fix_actions"):
                first = row["fix_actions"][0]
                if isinstance(first, dict):
                    first_fix = str(first.get("action", "")).strip()
            if not first_fix and tasks:
                first_fix = str(tasks[0].get("title", "")).strip()
            anchor = ExporterAndQAGateStep._anchor_text(row, risk_row)
            lines.extend(
                [
                    "",
                    f"### {i}) {rid} [{row.get('severity', risk_row.get('severity', 'P2'))}]",
                    f"- Problem: {row.get('problem_statement', row.get('issue', ''))}",
                    f"- Why: {row.get('root_cause_analysis', row.get('why_happened', ''))}",
                    f"- Impact: {row.get('impact_analysis', row.get('why_it_matters', ''))}",
                    f"- Evidence anchor: {anchor}",
                    f"- First fix: {first_fix or row.get('fix_summary', '')}",
                    f"- Rebuttal mapping: {risk_to_review.get(rid, 'see 003-rebuttal-draft.md')}",
                ]
            )
        lines.append("\nNext: open `student_pack/en/002-action-items.md`")
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _student_actions_en(
        *,
        diagnosis: dict[str, Any],
        remediation_tasks: object,
        risk_index: dict[str, dict[str, Any]],
        risk_to_review: dict[str, str],
    ) -> str:
        top = ExporterAndQAGateStep._top_diagnosis_rows(diagnosis, risk_index, 8)
        lines = ["# 002 Action Items", "", "Execute from top to bottom:"]
        for i, row in enumerate(top, start=1):
            rid = str(row.get("issue_id", f"RISK-{i:03d}"))
            risk_row = risk_index.get(rid, {})
            tasks = ExporterAndQAGateStep._tasks_for_risk(remediation_tasks, rid)
            anchor = ExporterAndQAGateStep._anchor_text(row, risk_row)
            lines.extend(
                [
                    "",
                    f"## {i}) {rid} [{row.get('severity', risk_row.get('severity', 'P2'))}]",
                    f"- Problem: {row.get('problem_statement', row.get('issue', ''))}",
                    f"- Anchor: {anchor}",
                    f"- Linked rebuttal: {risk_to_review.get(rid, 'see 003')}",
                ]
            )
            fix_actions = row.get("fix_actions", [])
            if isinstance(fix_actions, list) and fix_actions:
                lines.append("- Steps:")
                for j, step in enumerate(fix_actions[:4], start=1):
                    if isinstance(step, dict):
                        lines.append(f"  {j}. {step.get('action', '')}")
            if tasks:
                t = tasks[0]
                lines.append(
                    f"- Experiment: {t.get('id', 'EXP')} {t.get('title', '')} "
                    f"(days={t.get('est_time_days', '?')}, gpu={t.get('est_gpu_hours', '?')}h)"
                )
        lines.append("\nNext: open `student_pack/en/003-rebuttal-draft.md`")
        return "\n".join(lines).strip() + "\n"
    @staticmethod
    def _student_decision_zh(
        *,
        ctx: PipelineContext,
        decision: dict[str, Any],
        diagnosis: dict[str, Any],
        risk_index: dict[str, dict[str, Any]],
        remediation_tasks: object,
        risk_to_review: dict[str, str],
    ) -> str:
        top = ExporterAndQAGateStep._top_diagnosis_rows(diagnosis, risk_index, 3)
        lines = [
            "# 001 投稿决策（研究生版）",
            "",
            f"- 论文: `{Path(ctx.input_data.paper.path).stem}`",
            f"- 会议: **{ctx.input_data.venue.name} {ctx.input_data.venue.year}**",
            f"- 结论: **{decision.get('decision', 'N/A')}**",
            f"- 解释: {decision.get('decision_interpretation', '')}",
            "",
            "## Top 3 必修问题",
        ]
        for i, row in enumerate(top, start=1):
            rid = str(row.get("issue_id", f"RISK-{i:03d}"))
            risk_row = risk_index.get(rid, {})
            tasks = ExporterAndQAGateStep._tasks_for_risk(remediation_tasks, rid)
            first_fix = ""
            if isinstance(row.get("fix_actions", []), list) and row.get("fix_actions"):
                first = row["fix_actions"][0]
                if isinstance(first, dict):
                    first_fix = str(first.get("action", "")).strip()
            if not first_fix and tasks:
                first_fix = str(tasks[0].get("title", "")).strip()
            anchor = ExporterAndQAGateStep._anchor_text(row, risk_row)
            lines.extend(
                [
                    "",
                    f"### {i}) {rid} [{row.get('severity', risk_row.get('severity', 'P2'))}]",
                    f"- 问题: {row.get('problem_statement', row.get('issue', ''))}",
                    f"- 原因: {row.get('root_cause_analysis', row.get('why_happened', ''))}",
                    f"- 影响: {row.get('impact_analysis', row.get('why_it_matters', ''))}",
                    f"- 证据锚点: {anchor}",
                    f"- 第一优先动作: {first_fix or row.get('fix_summary', '')}",
                    f"- Rebuttal 映射: {risk_to_review.get(rid, '见 003-rebuttal-draft.md')}",
                ]
            )
        lines.append("\n下一步: 打开 `student_pack/zh/002-action-items.md`")
        return "\n".join(lines).strip() + "\n"
    @staticmethod
    def _student_actions_zh(
        *,
        diagnosis: dict[str, Any],
        remediation_tasks: object,
        risk_index: dict[str, dict[str, Any]],
        risk_to_review: dict[str, str],
    ) -> str:
        top = ExporterAndQAGateStep._top_diagnosis_rows(diagnosis, risk_index, 8)
        lines = ["# 002 行动清单", "", "请按顺序执行:"]
        for i, row in enumerate(top, start=1):
            rid = str(row.get("issue_id", f"RISK-{i:03d}"))
            risk_row = risk_index.get(rid, {})
            tasks = ExporterAndQAGateStep._tasks_for_risk(remediation_tasks, rid)
            anchor = ExporterAndQAGateStep._anchor_text(row, risk_row)
            lines.extend(
                [
                    "",
                    f"## {i}) {rid} [{row.get('severity', risk_row.get('severity', 'P2'))}]",
                    f"- 问题: {row.get('problem_statement', row.get('issue', ''))}",
                    f"- 锚点: {anchor}",
                    f"- 对应 rebuttal: {risk_to_review.get(rid, '见 003')}",
                ]
            )
            fix_actions = row.get("fix_actions", [])
            if isinstance(fix_actions, list) and fix_actions:
                lines.append("- 步骤:")
                for j, step in enumerate(fix_actions[:4], start=1):
                    if isinstance(step, dict):
                        lines.append(f"  {j}. {step.get('action', '')}")
            if tasks:
                t = tasks[0]
                lines.append(
                    f"- 关联实验: {t.get('id', 'EXP')} {t.get('title', '')} "
                    f"(天数={t.get('est_time_days', '?')}, GPU={t.get('est_gpu_hours', '?')}h)"
                )
        lines.append("\n下一步: 打开 `student_pack/zh/003-rebuttal-draft.md`")
        return "\n".join(lines).strip() + "\n"
    @staticmethod
    def _student_rebuttal(
        *,
        bundle: dict[str, Any],
        risk_to_review: dict[str, str],
        risk_index: dict[str, dict[str, Any]],
        zh: bool,
    ) -> str:
        reverse = {v: k for k, v in risk_to_review.items()}
        rows = bundle.get("items", []) if isinstance(bundle, dict) else []
        title = "# 003 Rebuttal 草稿（风险映射）" if zh else "# 003 Rebuttal Draft (Risk-Mapped)"
        lines = [title, ""]
        for row in rows:
            if not isinstance(row, dict):
                continue
            review_id = str(row.get("review_id", "R?")).strip() or "R?"
            concern = str(row.get("concern", "")).strip()
            risk_id = reverse.get(review_id, "")
            risk_row = risk_index.get(risk_id, {})
            sev = str(risk_row.get("severity", ""))
            score = float(risk_row.get("score", 0.0) or 0.0)
            anchor = str(row.get("evidence_anchor_hint", "")).strip()
            if not anchor:
                refs = row.get("evidence_anchor_refs", [])
                if isinstance(refs, list) and refs and isinstance(refs[0], dict):
                    anchor = ExporterAndQAGateStep._anchor_text({"evidence_anchors": refs}, {})
            lines.extend(
                [
                    "",
                    f"## {review_id}",
                    f"- Concern: {concern}" if not zh else f"- 审稿 concern: {concern}",
                    (
                        f"- Linked risk: {risk_id} [{sev}] (score={score:.3f})"
                        if not zh
                        else f"- 对应风险: {risk_id} [{sev}] (分数={score:.3f})"
                    ),
                    (
                        f"- Anchor to cite: {anchor or 'Add concrete section/table/figure anchor.'}"
                        if not zh
                        else f"- 引用锚点: {anchor or '补充具体 section/table/figure 锚点。'}"
                    ),
                    f"- Character budget: {row.get('char_count', 0)} / {row.get('char_limit', 0)}"
                    if not zh
                    else f"- 字数预算: {row.get('char_count', 0)} / {row.get('char_limit', 0)}",
                    "",
                    "> " + ("Draft response" if not zh else "草稿回复"),
                    f"> {row.get('response', '')}",
                ]
            )
        return "\n".join(lines).strip() + "\n"


    @staticmethod
    def _start_here_en() -> str:
        return (
            "# START HERE (Student-First)\n\n"
            "Read these first:\n"
            "1. `student_pack/en/001-submission-decision.md`\n"
            "2. `student_pack/en/002-action-items.md`\n"
            "3. `student_pack/en/003-rebuttal-draft.md`\n\n"
            "Other JSON files are debug/trace artifacts.\n"
        )

    @staticmethod
    def _start_here_zh() -> str:
        return (
            "# 从这里开始（研究生优先）\n\n"
            "先读这 3 个文件：\n"
            "1. `student_pack/zh/001-submission-decision.md`\n"
            "2. `student_pack/zh/002-action-items.md`\n"
            "3. `student_pack/zh/003-rebuttal-draft.md`\n\n"
            "其余 JSON 主要用于调试和追溯。\n"
        )

    @staticmethod
    def _start_here_bilingual() -> str:
        return (
            "# START HERE / 从这里开始\n\n"
            "Chinese:\n"
            "- `START_HERE.zh.md`\n"
            "- `student_pack/zh/001-submission-decision.md`\n"
            "- `student_pack/zh/002-action-items.md`\n"
            "- `student_pack/zh/003-rebuttal-draft.md`\n\n"
            "English:\n"
            "- `START_HERE.en.md`\n"
            "- `student_pack/en/001-submission-decision.md`\n"
            "- `student_pack/en/002-action-items.md`\n"
            "- `student_pack/en/003-rebuttal-draft.md`\n"
        )

