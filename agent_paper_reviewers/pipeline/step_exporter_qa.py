from __future__ import annotations

from pathlib import Path

from ..models import RunStatus
from ..services.feedback_store import build_feedback_template
from ..services.pdf_export import export_markdown_to_pdf
from .base import PipelineContext, PipelineStep


class ExporterAndQAGateStep(PipelineStep):
    name = "ExporterAndQAGate"
    _PDF_EXPORT_PREFIXES = (
        "START_HERE",
        "RUN_GUIDE",
        "STUDENT_BRIEF",
        "PERSONA_PLAYBOOK",
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
        top_risks = ExporterAndQAGateStep._top_risks(ctx, n=5)
        top_actions = ExporterAndQAGateStep._top_action_tasks(ctx, n=6)

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
        feedback_readme_zh = (
            "# 风险反馈模板说明\n\n"
            "1. 打开 `feedback_template.json`。\n"
            "2. 对每条风险，将 `verdict` 设置为 `correct` 或 `incorrect`。\n"
            "3. 可选：设置 `confidence`（0.0~1.0，默认 0.8）。\n"
            "4. 当 `verdict=incorrect` 时，建议补充 `comment`。\n"
            "5. 提交命令：\n"
            "   `python -m agent_paper_reviewers.cli submit-feedback --input <feedback_template.json路径>`\n"
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
            "runtime_context.json": ctx.artifacts.get("runtime_context", {}),
            "feedback_template.json": feedback_template,
            "feedback_README.en.md": feedback_readme,
            "RUN_GUIDE.en.md": ExporterAndQAGateStep._run_guide_en(ctx),
            "STUDENT_BRIEF.en.md": ExporterAndQAGateStep._student_brief_en(ctx, top_risks, top_actions),
            "PERSONA_PLAYBOOK.en.md": ExporterAndQAGateStep._persona_playbook_en(ctx, top_risks, top_actions),
            "AGENT_HANDOFF.json": ExporterAndQAGateStep._agent_handoff_payload(ctx, top_risks, top_actions),
        }
        deliverables.update(ExporterAndQAGateStep._collect_student_pack_deliverables(ctx))

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
                    "feedback_README.zh.md": feedback_readme_zh,
                    "RUN_GUIDE.zh.md": ExporterAndQAGateStep._run_guide_zh(ctx),
                    "STUDENT_BRIEF.zh.md": ExporterAndQAGateStep._student_brief_zh(ctx, top_risks, top_actions),
                    "PERSONA_PLAYBOOK.zh.md": ExporterAndQAGateStep._persona_playbook_zh(ctx, top_risks, top_actions),
                    "STUDENT_BRIEF.md": ExporterAndQAGateStep._student_brief_bilingual(),
                    "PERSONA_PLAYBOOK.md": ExporterAndQAGateStep._persona_playbook_bilingual(),
                    "RUN_GUIDE.md": ExporterAndQAGateStep._run_guide_bilingual(),
                }
            )
        else:
            deliverables["RUN_GUIDE.md"] = deliverables["RUN_GUIDE.en.md"]
            deliverables["STUDENT_BRIEF.md"] = deliverables["STUDENT_BRIEF.en.md"]
            deliverables["PERSONA_PLAYBOOK.md"] = deliverables["PERSONA_PLAYBOOK.en.md"]
        return deliverables

    @staticmethod
    def _collect_student_pack_deliverables(
        ctx: PipelineContext,
    ) -> dict[str, object]:
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
            ctx.add_qa_issue("student_pack_generation_failed:real_agent_output_required")
            ctx.status = RunStatus.PARTIAL_FAILED
            decision_en_md, action_en_md, rebuttal_en_md = ExporterAndQAGateStep._student_pack_error_notice_en()

        deliverables: dict[str, object] = {
            "student_pack/en/001-submission-decision.md": decision_en_md,
            "student_pack/en/002-action-items.md": action_en_md,
            "student_pack/en/003-rebuttal-draft.md": rebuttal_en_md,
            "START_HERE.en.md": ExporterAndQAGateStep._start_here_en(),
        }

        if ctx.input_data.options.language_mode.value == "en_zh":
            if has_agent_zh:
                decision_zh_md = str(agent_zh["001"])
                action_zh_md = str(agent_zh["002"])
                rebuttal_zh_md = str(agent_zh["003"])
            else:
                ctx.add_qa_issue("student_pack_generation_failed:real_agent_output_required_zh")
                ctx.status = RunStatus.PARTIAL_FAILED
                decision_zh_md, action_zh_md, rebuttal_zh_md = ExporterAndQAGateStep._student_pack_error_notice_zh()

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
    def _student_pack_error_notice_en() -> tuple[str, str, str]:
        notice = (
            "Student pack generation is blocked because real Agent output is required.\n"
            "Fallback template output is disabled to avoid low-quality watered-down guidance.\n"
            "Please configure a live executor backend (OPENAI/Anthropic/OpenClaw/other agent API) and rerun."
        )
        return (
            "# 001 Submission Decision\n\n"
            f"- Status: generation blocked\n- Reason: {notice}\n",
            "# 002 Action Items\n\n"
            f"- Status: generation blocked\n- Reason: {notice}\n",
            "# 003 Rebuttal Draft\n\n"
            f"- Status: generation blocked\n- Reason: {notice}\n",
        )

    @staticmethod
    def _student_pack_error_notice_zh() -> tuple[str, str, str]:
        notice = (
            "Student pack 必须由真实 Agent 生成。\n"
            "为了避免低质量模板化内容，系统已禁用 deterministic fallback。\n"
            "请配置可用的在线执行器（OPENAI/Anthropic/OpenClaw/其他 agent API）后重新运行。"
        )
        return (
            "# 001 投稿决策\n\n"
            f"- 状态：生成被阻止\n- 原因：{notice}\n",
            "# 002 行动清单\n\n"
            f"- 状态：生成被阻止\n- 原因：{notice}\n",
            "# 003 Rebuttal 草稿\n\n"
            f"- 状态：生成被阻止\n- 原因：{notice}\n",
        )


    @staticmethod
    def _start_here_en() -> str:
        return (
            "# START HERE (Student-First)\n\n"
            "Read these first:\n"
            "1. `STUDENT_BRIEF.en.md`\n"
            "2. `PERSONA_PLAYBOOK.en.md`\n"
            "3. `student_pack/en/001-submission-decision.md`\n"
            "4. `student_pack/en/002-action-items.md`\n"
            "5. `student_pack/en/003-rebuttal-draft.md`\n\n"
            "Other JSON files are debug/trace artifacts.\n"
        )

    @staticmethod
    def _start_here_zh() -> str:
        return (
            "# 从这里开始（研究生优先）\n\n"
            "先阅读这 4 个文件：\n"
            "1. `STUDENT_BRIEF.zh.md`\n"
            "2. `PERSONA_PLAYBOOK.zh.md`\n"
            "3. `student_pack/zh/001-submission-decision.md`\n"
            "4. `student_pack/zh/002-action-items.md`\n"
            "5. `student_pack/zh/003-rebuttal-draft.md`\n\n"
            "其余 JSON 主要用于调试和追踪。\n"
        )

    @staticmethod
    def _start_here_bilingual() -> str:
        return (
            "# START HERE / 从这里开始\n\n"
            "Chinese:\n"
            "- `START_HERE.zh.md`\n"
            "- `STUDENT_BRIEF.zh.md`\n"
            "- `PERSONA_PLAYBOOK.zh.md`\n"
            "- `student_pack/zh/001-submission-decision.md`\n"
            "- `student_pack/zh/002-action-items.md`\n"
            "- `student_pack/zh/003-rebuttal-draft.md`\n\n"
            "English:\n"
            "- `START_HERE.en.md`\n"
            "- `STUDENT_BRIEF.en.md`\n"
            "- `PERSONA_PLAYBOOK.en.md`\n"
            "- `student_pack/en/001-submission-decision.md`\n"
            "- `student_pack/en/002-action-items.md`\n"
            "- `student_pack/en/003-rebuttal-draft.md`\n"
        )

    @staticmethod
    def _run_guide_en(ctx: PipelineContext) -> str:
        status = ctx.status.value
        qa_issues = list(ctx.qa_issues)
        top_issues = qa_issues[:6]
        if not top_issues:
            top_issues = ["none"]

        lines = [
            "# RUN GUIDE (Execution + Next Actions)",
            "",
            f"- Run status: **{status}**",
            f"- Venue: **{ctx.input_data.venue.name} {ctx.input_data.venue.year}**",
            f"- Paper: `{Path(ctx.input_data.paper.path).stem}`",
            "",
            "## What to open first",
            "1. `START_HERE.en.md`",
            "2. `STUDENT_BRIEF.en.md`",
            "3. `PERSONA_PLAYBOOK.en.md`",
            "4. `student_pack/en/001-submission-decision.md`",
            "5. `student_pack/en/002-action-items.md`",
            "6. `student_pack/en/003-rebuttal-draft.md`",
            "",
            "## Health Check",
        ]
        for item in top_issues:
            lines.append(f"- {item}")

        if any("student_pack_generation_failed" in x for x in qa_issues):
            lines.extend(
                [
                    "",
                    "## Blocking Issue",
                    "- Student pack is blocked because real agent output is required.",
                    "- Fix: configure a live executor backend and rerun.",
                ]
            )

        lines.extend(
            [
                "",
                "## Tips",
                "- If this run is for `meta_review_discussion`, ensure reviewer comments are passed in input.",
                "- If you uploaded a PDF in chat but no local path exists, save it into the repo first and rerun.",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _run_guide_zh(ctx: PipelineContext) -> str:
        status = ctx.status.value
        qa_issues = list(ctx.qa_issues)
        top_issues = qa_issues[:6]
        if not top_issues:
            top_issues = ["无"]

        lines = [
            "# 运行指南（执行状态 + 下一步）",
            "",
            f"- 运行状态：**{status}**",
            f"- 目标会议：**{ctx.input_data.venue.name} {ctx.input_data.venue.year}**",
            f"- 论文：`{Path(ctx.input_data.paper.path).stem}`",
            "",
            "## 建议先看",
            "1. `START_HERE.zh.md`",
            "2. `STUDENT_BRIEF.zh.md`",
            "3. `PERSONA_PLAYBOOK.zh.md`",
            "4. `student_pack/zh/001-submission-decision.md`",
            "5. `student_pack/zh/002-action-items.md`",
            "6. `student_pack/zh/003-rebuttal-draft.md`",
            "",
            "## 运行健康检查",
        ]
        for item in top_issues:
            lines.append(f"- {item}")

        if any("student_pack_generation_failed" in x for x in qa_issues):
            lines.extend(
                [
                    "",
                    "## 阻断问题",
                    "- Student pack 被阻断：当前必须使用真实 Agent 生成。",
                    "- 处理方式：配置可用执行器后重新运行。",
                ]
            )

        lines.extend(
            [
                "",
                "## 使用建议",
                "- 处于 `meta_review_discussion` 阶段时，请在输入中附上 reviewer comments。",
                "- 如果 PDF 只在对话窗口里但没有落盘路径，请先保存到仓库再运行。",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _run_guide_bilingual() -> str:
        return (
            "# RUN GUIDE / 运行指南\n\n"
            "Chinese: `RUN_GUIDE.zh.md`\n"
            "English: `RUN_GUIDE.en.md`\n"
        )

    @staticmethod
    def _top_risks(ctx: PipelineContext, n: int) -> list[dict]:
        risk_rows = ctx.artifacts.get("risk_ranking", {}).get("risks", [])
        if not isinstance(risk_rows, list):
            return []
        severity_order = {"P0": 0, "P1": 1, "P2": 2}

        def _key(row: dict) -> tuple[int, float]:
            sev = str(row.get("severity", "P2")).upper()
            score = float(row.get("score", 0.0) or 0.0)
            return (severity_order.get(sev, 9), -score)

        rows = [r for r in risk_rows if isinstance(r, dict)]
        rows.sort(key=_key)
        return rows[:n]

    @staticmethod
    def _top_action_tasks(ctx: PipelineContext, n: int) -> list[dict]:
        tasks = ctx.artifacts.get("remediation_plan", {}).get("tasks", [])
        if not isinstance(tasks, list):
            return []
        priority_order = {"high": 0, "medium": 1, "low": 2}

        def _key(row: dict) -> tuple[int, float]:
            prio = str(row.get("priority", "medium")).lower()
            days = float(row.get("est_time_days", 999.0) or 999.0)
            return (priority_order.get(prio, 9), days)

        rows = [r for r in tasks if isinstance(r, dict)]
        rows.sort(key=_key)
        return rows[:n]

    @staticmethod
    def _agent_handoff_payload(
        ctx: PipelineContext,
        top_risks: list[dict],
        top_actions: list[dict],
    ) -> dict:
        qa_issues = list(ctx.qa_issues)
        has_executor_warning = any("executor_warning" in x for x in qa_issues)
        student_pack_blocked = any("student_pack_generation_failed" in x for x in qa_issues)
        strict_failed = has_executor_warning or student_pack_blocked
        paper_path = str(Path(ctx.input_data.paper.path).resolve())
        rerun_cmd = (
            "python -m agent_paper_reviewers.cli review-pdf "
            f"--paper-path \"{paper_path}\" "
            f"--venue \"{ctx.input_data.venue.name}\" "
            f"--year {ctx.input_data.venue.year} "
            f"--executor-backend {ctx.input_data.options.executor_backend.value} "
            f"--language-mode {ctx.input_data.options.language_mode.value} "
            "--output-dir output --ai-summary --strict-quality"
        )
        return {
            "run_id": ctx.run_id,
            "status": ctx.status.value,
            "paper": {
                "title_stem": Path(ctx.input_data.paper.path).stem,
                "path": paper_path,
                "format": ctx.input_data.paper.format,
            },
            "venue": {
                "name": ctx.input_data.venue.name,
                "year": ctx.input_data.venue.year,
            },
            "review_context": {
                "manuscript_stage": ctx.input_data.review_context.manuscript_stage.value,
                "reviewer_comments_count": len(ctx.input_data.review_context.reviewer_comments),
            },
            "quality": {
                "qa_issue_count": len(qa_issues),
                "has_executor_warning": has_executor_warning,
                "student_pack_blocked": student_pack_blocked,
                "strict_quality_would_fail": strict_failed,
            },
            "top_risks": [
                {
                    "id": str(x.get("id", "")),
                    "severity": str(x.get("severity", "")),
                    "score": float(x.get("score", 0.0) or 0.0),
                    "reason": str(x.get("reason", "")),
                }
                for x in top_risks
            ],
            "top_actions": [
                {
                    "id": str(x.get("id", "")),
                    "risk_id": str(x.get("risk_id", "")),
                    "title": str(x.get("title", "")),
                    "priority": str(x.get("priority", "")),
                    "est_time_days": float(x.get("est_time_days", 0.0) or 0.0),
                    "est_gpu_hours": int(float(x.get("est_gpu_hours", 0) or 0)),
                }
                for x in top_actions
            ],
            "next_commands": {
                "rerun_strict": rerun_cmd,
                "open_run_guide": "open RUN_GUIDE.en.md",
                "open_student_brief": "open STUDENT_BRIEF.en.md",
                "open_persona_playbook": "open PERSONA_PLAYBOOK.en.md",
            },
            "notes": [
                "Use this file as the machine handoff contract for next agent turn.",
                "If strict_quality_would_fail=true, prefer fixing backend/quality blockers first.",
            ],
        }

    @staticmethod
    def _student_brief_en(
        ctx: PipelineContext,
        top_risks: list[dict],
        top_actions: list[dict],
    ) -> str:
        decision = str(ctx.artifacts.get("reports", {}).get("en", {}).get("decision_json", {}).get("decision", "N/A"))
        lines = [
            "# STUDENT BRIEF (Do This First)",
            "",
            f"- Decision now: **{decision}**",
            f"- Venue target: **{ctx.input_data.venue.name} {ctx.input_data.venue.year}**",
            "",
            "## Top blockers",
        ]
        if not top_risks:
            lines.append("- No risks found in this run.")
        for idx, row in enumerate(top_risks[:3], start=1):
            lines.extend(
                [
                    f"{idx}. [{row.get('severity', 'P2')}] {row.get('id', 'RISK-?')}: {row.get('reason', '')}",
                    f"   - Quick fix hint: {row.get('fix_hint', 'See 002-action-items.')}",
                ]
            )
        lines.extend(["", "## First 24-hour plan"])
        if top_actions:
            for idx, task in enumerate(top_actions[:3], start=1):
                lines.append(
                    f"{idx}. {task.get('id', 'TASK')} {task.get('title', '')} "
                    f"(priority={task.get('priority', 'medium')}, "
                    f"time~{task.get('est_time_days', '?')}d, gpu~{task.get('est_gpu_hours', '?')}h)"
                )
        else:
            lines.append("1. Open `RUN_GUIDE.en.md` and resolve blocking issues first.")
        lines.extend(
            [
                "",
                "## Read in order",
                "1. `student_pack/en/001-submission-decision.md`",
                "2. `student_pack/en/002-action-items.md`",
                "3. `student_pack/en/003-rebuttal-draft.md`",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _student_brief_zh(
        ctx: PipelineContext,
        top_risks: list[dict],
        top_actions: list[dict],
    ) -> str:
        decision = str(ctx.artifacts.get("reports", {}).get("zh", {}).get("decision_json", {}).get("decision", "N/A"))
        lines = [
            "# STUDENT BRIEF（先做这个）",
            "",
            f"- 当前结论：**{decision}**",
            f"- 目标会议：**{ctx.input_data.venue.name} {ctx.input_data.venue.year}**",
            "",
            "## 当前最关键阻断",
        ]
        if not top_risks:
            lines.append("- 本次运行未检测到风险项。")
        for idx, row in enumerate(top_risks[:3], start=1):
            lines.extend(
                [
                    f"{idx}. [{row.get('severity', 'P2')}] {row.get('id', 'RISK-?')}：{row.get('reason', '')}",
                    f"   - 快速修复提示：{row.get('fix_hint', '请看 002-action-items。')}",
                ]
            )
        lines.extend(["", "## 前 24 小时行动"])
        if top_actions:
            for idx, task in enumerate(top_actions[:3], start=1):
                lines.append(
                    f"{idx}. {task.get('id', 'TASK')} {task.get('title', '')} "
                    f"(优先级={task.get('priority', 'medium')}，"
                    f"时间约{task.get('est_time_days', '?')}天，GPU约{task.get('est_gpu_hours', '?')}小时)"
                )
        else:
            lines.append("1. 先打开 `RUN_GUIDE.zh.md` 解决阻断项。")
        lines.extend(
            [
                "",
                "## 阅读顺序",
                "1. `student_pack/zh/001-submission-decision.md`",
                "2. `student_pack/zh/002-action-items.md`",
                "3. `student_pack/zh/003-rebuttal-draft.md`",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _student_brief_bilingual() -> str:
        return (
            "# STUDENT BRIEF / 研究生摘要\n\n"
            "Chinese: `STUDENT_BRIEF.zh.md`\n"
            "English: `STUDENT_BRIEF.en.md`\n"
        )

    @staticmethod
    def _persona_playbook_en(
        ctx: PipelineContext,
        top_risks: list[dict],
        top_actions: list[dict],
    ) -> str:
        qa_issues = [str(x) for x in ctx.qa_issues]
        has_blocker = any(
            token in issue
            for issue in qa_issues
            for token in ("student_pack_generation_failed", "executor_warning", "fallback", "pdf_parse_quality")
        )
        status = ctx.status.value
        paper_path = str(Path(ctx.input_data.paper.path).resolve())
        lines = [
            "# PERSONA PLAYBOOK (Agent + Graduate Student)",
            "",
            f"- Run status: **{status}**",
            f"- Venue: **{ctx.input_data.venue.name} {ctx.input_data.venue.year}**",
            f"- Paper path: `{paper_path}`",
            "",
            "## Persona A: Agent Operator (automation-first)",
            "1. Open `AGENT_HANDOFF.json` and read `quality.strict_quality_would_fail`.",
            "2. If strict-quality would fail, fix backend/quality blockers before any content rewrite.",
            "3. Then open `RUN_GUIDE.en.md` and execute `next_commands.rerun_strict` from AGENT_HANDOFF.",
            "4. After rerun passes, use `student_pack/en/002-action-items.md` as the execution backlog.",
            "",
            "## Persona B: Graduate Student Author (revision-first)",
            "1. Open `STUDENT_BRIEF.en.md` to lock today's top 3 tasks.",
            "2. Read `student_pack/en/001-submission-decision.md` to understand submit/hold recommendation.",
            "3. Execute `student_pack/en/002-action-items.md` strictly by priority.",
            "4. Edit response text in `student_pack/en/003-rebuttal-draft.md` with real numbers/anchors.",
            "",
            "## Current blockers",
        ]
        if has_blocker:
            for row in qa_issues[:6]:
                lines.append(f"- {row}")
        else:
            lines.append("- No hard blocker detected in this run.")

        lines.extend(["", "## Top risk -> top action map"])
        for idx, risk in enumerate(top_risks[:3], start=1):
            task = top_actions[idx - 1] if idx - 1 < len(top_actions) else {}
            lines.append(
                f"{idx}. {risk.get('id', 'RISK-?')} [{risk.get('severity', 'P2')}] "
                f"-> {task.get('id', 'TASK')} {task.get('title', 'check 002-action-items')}"
            )
        if not top_risks:
            lines.append("- No risk ranked in this run.")

        lines.extend(
            [
                "",
                "## One-line command for rerun",
                (
                    "python -m agent_paper_reviewers.cli review-pdf "
                    f"--paper-path \"{paper_path}\" "
                    f"--venue \"{ctx.input_data.venue.name}\" --year {ctx.input_data.venue.year} "
                    f"--executor-backend {ctx.input_data.options.executor_backend.value} "
                    f"--language-mode {ctx.input_data.options.language_mode.value} "
                    "--output-dir output --ai-summary --strict-quality"
                ),
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _persona_playbook_zh(
        ctx: PipelineContext,
        top_risks: list[dict],
        top_actions: list[dict],
    ) -> str:
        qa_issues = [str(x) for x in ctx.qa_issues]
        has_blocker = any(
            token in issue
            for issue in qa_issues
            for token in ("student_pack_generation_failed", "executor_warning", "fallback", "pdf_parse_quality")
        )
        status = ctx.status.value
        paper_path = str(Path(ctx.input_data.paper.path).resolve())
        lines = [
            "# PERSONA PLAYBOOK（Agent + 研究生）",
            "",
            f"- 运行状态：**{status}**",
            f"- 目标会议：**{ctx.input_data.venue.name} {ctx.input_data.venue.year}**",
            f"- 论文路径：`{paper_path}`",
            "",
            "## 人设 A：Agent 编排者（自动化优先）",
            "1. 先看 `AGENT_HANDOFF.json` 的 `quality.strict_quality_would_fail`。",
            "2. 若 strict-quality 会失败，先处理后端/解析阻断，不要直接改内容。",
            "3. 再看 `RUN_GUIDE.zh.md`，执行 AGENT_HANDOFF 里的 `next_commands.rerun_strict`。",
            "4. 通过后，把 `student_pack/zh/002-action-items.md` 当作执行 backlog。",
            "",
            "## 人设 B：研究生作者（改稿优先）",
            "1. 先读 `STUDENT_BRIEF.zh.md`，锁定今天最重要 3 件事。",
            "2. 再看 `student_pack/zh/001-submission-decision.md`，确认是否建议投稿。",
            "3. 按 `student_pack/zh/002-action-items.md` 的优先级逐条执行。",
            "4. 在 `student_pack/zh/003-rebuttal-draft.md` 中填入真实数字与证据锚点。",
            "",
            "## 当前阻断",
        ]
        if has_blocker:
            for row in qa_issues[:6]:
                lines.append(f"- {row}")
        else:
            lines.append("- 本次运行未检测到硬阻断。")

        lines.extend(["", "## 风险 -> 动作映射（Top）"])
        for idx, risk in enumerate(top_risks[:3], start=1):
            task = top_actions[idx - 1] if idx - 1 < len(top_actions) else {}
            lines.append(
                f"{idx}. {risk.get('id', 'RISK-?')} [{risk.get('severity', 'P2')}] "
                f"-> {task.get('id', 'TASK')} {task.get('title', '请看 002-action-items')}"
            )
        if not top_risks:
            lines.append("- 本次运行没有生成风险排序。")

        lines.extend(
            [
                "",
                "## 一键重跑命令",
                (
                    "python -m agent_paper_reviewers.cli review-pdf "
                    f"--paper-path \"{paper_path}\" "
                    f"--venue \"{ctx.input_data.venue.name}\" --year {ctx.input_data.venue.year} "
                    f"--executor-backend {ctx.input_data.options.executor_backend.value} "
                    f"--language-mode {ctx.input_data.options.language_mode.value} "
                    "--output-dir output --ai-summary --strict-quality"
                ),
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _persona_playbook_bilingual() -> str:
        return (
            "# PERSONA PLAYBOOK / 双人设执行手册\n\n"
            "Chinese: `PERSONA_PLAYBOOK.zh.md`\n"
            "English: `PERSONA_PLAYBOOK.en.md`\n"
        )


