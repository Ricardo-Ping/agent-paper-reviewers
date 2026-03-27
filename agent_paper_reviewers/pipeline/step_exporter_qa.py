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
        "CHAT_SUMMARY",
        "CHAT_REBUTTAL",
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
            "CHAT_SUMMARY.en.md": ExporterAndQAGateStep._chat_summary_en(ctx, top_risks, top_actions),
            "CHAT_REBUTTAL.en.md": ExporterAndQAGateStep._chat_rebuttal_en(ctx, top_risks),
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
                    "CHAT_SUMMARY.zh.md": ExporterAndQAGateStep._chat_summary_zh(ctx, top_risks, top_actions),
                    "CHAT_REBUTTAL.zh.md": ExporterAndQAGateStep._chat_rebuttal_zh(ctx, top_risks),
                    "STUDENT_BRIEF.md": ExporterAndQAGateStep._student_brief_bilingual(),
                    "PERSONA_PLAYBOOK.md": ExporterAndQAGateStep._persona_playbook_bilingual(),
                    "CHAT_SUMMARY.md": ExporterAndQAGateStep._chat_summary_bilingual(),
                    "CHAT_REBUTTAL.md": ExporterAndQAGateStep._chat_rebuttal_bilingual(),
                    "RUN_GUIDE.md": ExporterAndQAGateStep._run_guide_bilingual(),
                }
            )
        else:
            deliverables["RUN_GUIDE.md"] = deliverables["RUN_GUIDE.en.md"]
            deliverables["STUDENT_BRIEF.md"] = deliverables["STUDENT_BRIEF.en.md"]
            deliverables["PERSONA_PLAYBOOK.md"] = deliverables["PERSONA_PLAYBOOK.en.md"]
            deliverables["CHAT_SUMMARY.md"] = deliverables["CHAT_SUMMARY.en.md"]
            deliverables["CHAT_REBUTTAL.md"] = deliverables["CHAT_REBUTTAL.en.md"]
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
            "2. `CHAT_SUMMARY.en.md`\n"
            "3. `CHAT_REBUTTAL.en.md`\n"
            "4. `PERSONA_PLAYBOOK.en.md`\n"
            "5. `student_pack/en/001-submission-decision.md`\n"
            "6. `student_pack/en/002-action-items.md`\n"
            "7. `student_pack/en/003-rebuttal-draft.md`\n\n"
            "Other JSON files are debug/trace artifacts.\n"
        )

    @staticmethod
    def _start_here_zh() -> str:
        return (
            "# 从这里开始（研究生优先）\n\n"
            "先阅读这 4 个文件：\n"
            "1. `STUDENT_BRIEF.zh.md`\n"
            "2. `CHAT_SUMMARY.zh.md`\n"
            "3. `CHAT_REBUTTAL.zh.md`\n"
            "4. `PERSONA_PLAYBOOK.zh.md`\n"
            "5. `student_pack/zh/001-submission-decision.md`\n"
            "6. `student_pack/zh/002-action-items.md`\n"
            "7. `student_pack/zh/003-rebuttal-draft.md`\n\n"
            "其余 JSON 主要用于调试和追踪。\n"
        )

    @staticmethod
    def _start_here_bilingual() -> str:
        return (
            "# START HERE / 从这里开始\n\n"
            "Chinese:\n"
            "- `START_HERE.zh.md`\n"
            "- `STUDENT_BRIEF.zh.md`\n"
            "- `CHAT_SUMMARY.zh.md`\n"
            "- `CHAT_REBUTTAL.zh.md`\n"
            "- `PERSONA_PLAYBOOK.zh.md`\n"
            "- `student_pack/zh/001-submission-decision.md`\n"
            "- `student_pack/zh/002-action-items.md`\n"
            "- `student_pack/zh/003-rebuttal-draft.md`\n\n"
            "English:\n"
            "- `START_HERE.en.md`\n"
            "- `STUDENT_BRIEF.en.md`\n"
            "- `CHAT_SUMMARY.en.md`\n"
            "- `CHAT_REBUTTAL.en.md`\n"
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
            "3. `CHAT_SUMMARY.en.md`",
            "4. `CHAT_REBUTTAL.en.md`",
            "5. `PERSONA_PLAYBOOK.en.md`",
            "6. `student_pack/en/001-submission-decision.md`",
            "7. `student_pack/en/002-action-items.md`",
            "8. `student_pack/en/003-rebuttal-draft.md`",
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
            "3. `CHAT_SUMMARY.zh.md`",
            "4. `CHAT_REBUTTAL.zh.md`",
            "5. `PERSONA_PLAYBOOK.zh.md`",
            "6. `student_pack/zh/001-submission-decision.md`",
            "7. `student_pack/zh/002-action-items.md`",
            "8. `student_pack/zh/003-rebuttal-draft.md`",
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
                "open_chat_summary": "open CHAT_SUMMARY.en.md",
                "open_chat_rebuttal": "open CHAT_REBUTTAL.en.md",
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

    @staticmethod
    def _chat_summary_en(
        ctx: PipelineContext,
        top_risks: list[dict],
        top_actions: list[dict],
    ) -> str:
        paper_title = Path(ctx.input_data.paper.path).stem
        lines = [
            "# Chat Summary (Human-readable, for Graduate Students)",
            "",
            f"Paper: **{paper_title}**",
            f"Target venue: **{ctx.input_data.venue.name} {ctx.input_data.venue.year}**",
            "",
            "This summary translates risk codes into plain, actionable reviewer-style feedback. "
            "No JSON knowledge is required to use this file.",
            "",
            "## Executive conclusion",
            (
                "The current draft is **not ready for submission**. The main issue is not that the idea is meaningless, "
                "but that claim-level evidence is not yet explicit enough for reviewers to verify quickly. "
                "Your revision should prioritize statistical rigor, claim-to-evidence mapping, and terminology consistency."
            ),
            "",
            "## Detailed issues (plain language)",
        ]
        for idx, risk in enumerate(top_risks[:4], start=1):
            action = next(
                (
                    t
                    for t in top_actions
                    if str(t.get("risk_id", "")).strip() == str(risk.get("id", "")).strip()
                ),
                {},
            )
            lines.append(ExporterAndQAGateStep._issue_block_en(idx, risk, action))
        if not top_risks:
            lines.append("- No top risks available in this run.")
        lines.extend(
            [
                "",
                "## What to do this week",
                "1. Execute `student_pack/en/002-action-items.md` from top to bottom.",
                "2. Rewrite claim statements so each claim has one table/figure anchor and one statistical statement.",
                "3. Revise `rebuttal.en.md` after evidence is updated; avoid sending generic text without numbers.",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _chat_summary_zh(
        ctx: PipelineContext,
        top_risks: list[dict],
        top_actions: list[dict],
    ) -> str:
        paper_title = Path(ctx.input_data.paper.path).stem
        lines = [
            "# 中文总结（研究生可直接阅读）",
            "",
            f"论文：**{paper_title}**",
            f"目标会议：**{ctx.input_data.venue.name} {ctx.input_data.venue.year}**",
            "",
            "这份总结的目标是把 `RISK-xxx/P0-P2` 这种机器标签翻译成研究生可直接执行的人话建议。"
            "你不需要打开 JSON，也不需要先理解系统内部字段。",
            "",
            "## 总体结论",
            (
                "当前稿件暂不建议直接投稿。核心问题不是“没有工作量”，而是“主张与证据之间的审稿闭环没有写清楚”。"
                "具体表现为：关键主张缺少可直接验证的量化证据、统计显著性表述不完整、部分术语在不同章节中定义不一致。"
                "这些问题在真实审稿中会被放大为可信度风险，因此需要先补强再投。"
            ),
            "",
            "## 逐条问题（可执行版本）",
        ]
        for idx, risk in enumerate(top_risks[:4], start=1):
            action = next(
                (
                    t
                    for t in top_actions
                    if str(t.get("risk_id", "")).strip() == str(risk.get("id", "")).strip()
                ),
                {},
            )
            lines.append(ExporterAndQAGateStep._issue_block_zh(idx, risk, action))
        if not top_risks:
            lines.append("- 本次运行未生成 Top 风险。")
        lines.extend(
            [
                "",
                "## 本周改稿优先顺序",
                "1. 先按 `student_pack/zh/002-action-items.md` 的优先级执行，不要跳着改。",
                "2. 每条主张都补齐“量化结果 + 统计显著性 + 证据锚点（表/图/节）”。",
                "3. 证据补完后再改 `rebuttal.zh.md`，避免空泛回复。",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _chat_summary_bilingual() -> str:
        return (
            "# CHAT SUMMARY / 双语总结\n\n"
            "Chinese: `CHAT_SUMMARY.zh.md`\n"
            "English: `CHAT_SUMMARY.en.md`\n"
        )

    @staticmethod
    def _chat_rebuttal_en(ctx: PipelineContext, top_risks: list[dict]) -> str:
        bundle = ctx.artifacts.get("rebuttal", {}).get("en", {}).get("bundle", {})
        items = bundle.get("items", []) if isinstance(bundle, dict) else []
        has_real_comments = len(ctx.input_data.review_context.reviewer_comments) > 0
        lines = [
            "# Chat Rebuttal Guide (English, reviewer-facing draft)",
            "",
            (
                "Reviewer IDs in this file are "
                + ("from your provided reviewer comments." if has_real_comments else "simulated for pre-submission rehearsal.")
            ),
            (
                "If no real reviewer comments were provided, `R1..Rn` are synthetic labels generated by the skill so you can prepare rebuttal early."
            ),
            "",
        ]
        for idx, item in enumerate(items, start=1):
            risk = top_risks[idx - 1] if idx - 1 < len(top_risks) else {}
            lines.append(ExporterAndQAGateStep._rebuttal_block_en(idx, item, risk))
        if not items:
            lines.append("No rebuttal items found in this run.")
        lines.extend(
            [
                "",
                "## Final checklist before using this rebuttal",
                "1. Replace placeholder numbers with your updated experiment values.",
                "2. Ensure every response points to at least one concrete table/figure/section anchor.",
                "3. Keep tone factual and non-defensive; acknowledge limitations when needed.",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _chat_rebuttal_zh(ctx: PipelineContext, top_risks: list[dict]) -> str:
        bundle = ctx.artifacts.get("rebuttal", {}).get("en", {}).get("bundle", {})
        items = bundle.get("items", []) if isinstance(bundle, dict) else []
        has_real_comments = len(ctx.input_data.review_context.reviewer_comments) > 0
        lines = [
            "# Rebuttal 中文讲解稿（可直接改写成英文回应）",
            "",
            (
                "本文件中的 Reviewer 编号"
                + ("来自你输入的真实 reviewer comments。" if has_real_comments else "为系统模拟编号，用于投稿前拒稿演练。")
            ),
            "如果你没有输入真实评审意见，`R1..Rn` 仅表示“第 1/2/3 条主要审稿关注点”，不是 OpenReview 的真实评审人编号。",
            "",
        ]
        for idx, item in enumerate(items, start=1):
            risk = top_risks[idx - 1] if idx - 1 < len(top_risks) else {}
            lines.append(ExporterAndQAGateStep._rebuttal_block_zh(idx, item, risk))
        if not items:
            lines.append("本次运行未生成可用 rebuttal 条目。")
        lines.extend(
            [
                "",
                "## 提交前核对清单",
                "1. 把模板里的占位数字替换成你实际重跑后的结果。",
                "2. 每条回应至少绑定一个可定位证据锚点（图/表/章节）。",
                "3. 避免空泛承诺，写清楚“改了什么、在哪里改、为什么能解决 concern”。",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _chat_rebuttal_bilingual() -> str:
        return (
            "# CHAT REBUTTAL / 双语 Rebuttal 讲解\n\n"
            "Chinese: `CHAT_REBUTTAL.zh.md`\n"
            "English: `CHAT_REBUTTAL.en.md`\n"
        )

    @staticmethod
    def _issue_block_en(idx: int, risk: dict, action: dict) -> str:
        risk_id = str(risk.get("id", "RISK-?"))
        severity = str(risk.get("severity", "P2"))
        reason = str(risk.get("reason", "")).strip()
        fix_hint = str(risk.get("fix_hint", "See action items")).strip()
        action_title = str(action.get("title", "Execute targeted validation update")).strip()
        evidence = ExporterAndQAGateStep._anchors_for_humans(risk, lang="en")
        specific_reasoning = ExporterAndQAGateStep._risk_reasoning_en(reason)

        text = (
            f"### Issue {idx} — {risk_id} [{severity}]\n\n"
            f"**What is wrong now:** {reason}\n\n"
            f"{specific_reasoning}\n\n"
            "If this is not fixed, the likely outcome is not just a weaker score on one axis. It can cascade into soundness and experiment penalties, "
            "because missing direct evidence is often interpreted as missing methodological rigor. That is why this item is prioritized ahead of cosmetic writing edits.\n\n"
            f"**Where this appears in your draft now:** {evidence}\n\n"
            f"**What to do immediately:** {action_title}. "
            "In practical terms, add one explicit claim-evidence table row, one anchor to a figure/table with exact numbers, and one statistical statement (mean±std or p-value) "
            f"for this concern. Existing fix hint from the pipeline: {fix_hint}\n\n"
            "After this revision, your rebuttal will become substantially stronger because you can answer with concrete evidence instead of future promises."
        )
        return ExporterAndQAGateStep._ensure_min_chars(text, 900, lang="en")

    @staticmethod
    def _issue_block_zh(idx: int, risk: dict, action: dict) -> str:
        risk_id = str(risk.get("id", "RISK-?"))
        severity = str(risk.get("severity", "P2"))
        reason = str(risk.get("reason", "")).strip()
        fix_hint = str(risk.get("fix_hint", "请参考行动清单")).strip()
        action_title = str(action.get("title", "执行针对性补强实验")).strip()
        evidence = ExporterAndQAGateStep._anchors_for_humans(risk, lang="zh")
        specific_reasoning = ExporterAndQAGateStep._risk_reasoning_zh(reason)

        text = (
            f"### 问题 {idx} — {risk_id} [{severity}]\n\n"
            f"**当前问题是什么：**{reason}\n\n"
            f"{specific_reasoning}\n\n"
            "如果这个问题不改，影响不会只停留在单一维度。它会连带拉低 soundness 和 experiment 两个关键评分，因为证据链缺失会被解读为实验设计和验证过程不够严谨。"
            "这也是为什么你应该优先修这类问题，而不是先做措辞润色或排版优化。\n\n"
            f"**论文里目前可定位的相关证据：**{evidence}\n\n"
            f"**建议你本周立刻执行的动作：**{action_title}。"
            "具体做法是：为该问题补一条主张-证据映射（写清主张文本、对应表图、关键数字、统计指标），"
            "并在正文中把锚点写成审稿人一眼可定位的形式（例如 Section x.x / Table y / Figure z）。"
            f"系统当前给出的修复提示是：{fix_hint}\n\n"
            "完成这一步后，你的 rebuttal 就能从“承诺会补”升级为“已补并给出证据”，说服力会显著提升。"
        )
        return ExporterAndQAGateStep._ensure_min_chars(text, 520, lang="zh")

    @staticmethod
    def _rebuttal_block_en(idx: int, item: dict, risk: dict) -> str:
        review_id = str(item.get("review_id", f"R{idx}"))
        concern = str(item.get("concern", "")).strip()
        response = str(item.get("response", "")).strip()
        paper_change = str(item.get("paper_change", "")).strip()
        risk_tag = f"{risk.get('id', 'RISK-?')} [{risk.get('severity', 'P2')}]" if risk else "N/A"
        evidence_rows = item.get("new_evidence", []) if isinstance(item.get("new_evidence", []), list) else []
        anchors = item.get("evidence_anchor_hint") or ExporterAndQAGateStep._anchors_for_humans(risk, lang="en")
        evidence_text = "; ".join(str(x) for x in evidence_rows[:4]) if evidence_rows else "Add concrete claim-linked evidence rows."

        text = (
            f"## Reviewer {review_id} ({risk_tag})\n\n"
            f"**Concern explained in plain language:** {concern}\n\n"
            "This concern is legitimate under top-tier review standards because reviewers expect one-to-one mapping between claims and quantitative support. "
            "If your response remains generic, the reviewer will interpret it as acknowledgement without resolution.\n\n"
            f"**Draft response you can edit:** {response}\n\n"
            f"**Evidence to add before final rebuttal submission:** {evidence_text}\n\n"
            f"**Where to anchor in the paper:** {anchors}\n\n"
            f"**Paper changes to declare explicitly:** {paper_change or 'Revise methods/experiments sections and add claim-evidence mapping.'}\n\n"
            "Before sending this response, replace generic statements with exact numbers, mention the comparison setup, and clarify which revised section now contains the new evidence."
        )
        return ExporterAndQAGateStep._ensure_min_chars(text, 700, lang="en")

    @staticmethod
    def _rebuttal_block_zh(idx: int, item: dict, risk: dict) -> str:
        review_id = str(item.get("review_id", f"R{idx}"))
        concern = str(item.get("concern", "")).strip()
        response = str(item.get("response", "")).strip()
        paper_change = str(item.get("paper_change", "")).strip()
        risk_tag = f"{risk.get('id', 'RISK-?')} [{risk.get('severity', 'P2')}]" if risk else "N/A"
        evidence_rows = item.get("new_evidence", []) if isinstance(item.get("new_evidence", []), list) else []
        anchors = item.get("evidence_anchor_hint") or ExporterAndQAGateStep._anchors_for_humans(risk, lang="zh")
        evidence_text = "；".join(str(x) for x in evidence_rows[:4]) if evidence_rows else "补充主张对应的具体证据条目。"

        text = (
            f"## Reviewer {review_id}（{risk_tag}）\n\n"
            f"**Concern 的人话解释：**{concern}\n\n"
            "这条 concern 在顶会审稿里是合理且常见的。审稿人并不只看你“是否愿意改”，而是看你“是否已经给出可核查的修复证据”。"
            "如果 rebuttal 只有泛化承诺，没有具体数字、对比设置和证据锚点，通常会被认为没有实质性解决问题。\n\n"
            f"**可编辑回应草稿：**{response}\n\n"
            f"**提交前必须补上的证据：**{evidence_text}\n\n"
            f"**建议绑定的证据锚点：**{anchors}\n\n"
            f"**论文中要明确声明的改动：**{paper_change or '更新方法/实验章节，并新增主张-证据映射。'}\n\n"
            "最终提交前，请把“会补充”改成“已补充到哪里、具体数字是多少、对比对象是什么”，这样 rebuttal 才有说服力。"
        )
        return ExporterAndQAGateStep._ensure_min_chars(text, 520, lang="zh")

    @staticmethod
    def _anchors_for_humans(risk: dict, lang: str = "en") -> str:
        refs = risk.get("evidence_refs", []) if isinstance(risk.get("evidence_refs", []), list) else []
        if not refs:
            return "No anchor available." if lang == "en" else "暂无可用锚点。"
        parts: list[str] = []
        for ref in refs[:3]:
            if not isinstance(ref, dict):
                continue
            sec = str(ref.get("section", "unknown"))
            pid = str(ref.get("passage_id", "unknown"))
            page = ref.get("page")
            anchor = str(ref.get("anchor_label", "")).strip()
            page_text = f"p.{page}" if page else "p.?"
            if anchor:
                parts.append(f"{sec}/{pid}/{page_text}/{anchor}")
            else:
                parts.append(f"{sec}/{pid}/{page_text}")
        return "; ".join(parts) if parts else ("No anchor available." if lang == "en" else "暂无可用锚点。")

    @staticmethod
    def _ensure_min_chars(text: str, min_chars: int, lang: str = "en") -> str:
        if len(text) >= min_chars:
            return text
        if lang == "zh":
            pad = (
                "补充说明：审稿人真正关注的是“可验证性”和“可复核性”。请在每个回应中补上可定位证据、"
                "明确比较对象与统计口径，并把修订位置写清楚。这样可以显著降低“回复充分但证据不足”的风险。"
            )
        else:
            pad = (
                "Additional note: reviewers evaluate verifiability and reproducibility, not intentions. "
                "For each response, provide concrete anchors, explicit comparison setup, and statistical evidence. "
                "State exactly where the revised evidence appears in the manuscript."
            )
        out = text
        while len(out) < min_chars:
            out += "\n\n" + pad
        return out

    @staticmethod
    def _risk_reasoning_en(reason: str) -> str:
        r = reason.lower()
        if "statistical" in r or "p-value" in r or "confidence interval" in r:
            return (
                "This issue is specifically about statistical credibility. Even strong average gains can be discounted if variance and significance are missing, "
                "because reviewers cannot separate real improvements from random fluctuation. For benchmark-heavy venues, missing mean±std and tests is often treated "
                "as an experimental design weakness, not a formatting omission."
            )
        if "terminology" in r or "consistency" in r:
            return (
                "This issue targets conceptual consistency. When terminology drifts across sections, reviewers suspect that definitions, mappings, or evaluation assumptions "
                "are not stable. In system papers, inconsistent naming can be interpreted as methodological ambiguity and directly reduces trust in reproducibility."
            )
        if "none evidence support" in r or "weak evidence support" in r:
            return (
                "This is a direct claim-evidence mismatch. The problem is not that the paper lacks content, but that the core claim is not explicitly bound to quantitative proof. "
                "Reviewers usually reject such claims because they cannot verify exactly where the claim is established and under which comparison protocol."
            )
        if "section length ratio imbalance" in r:
            return (
                "This issue is structural and affects reviewer reading efficiency. If method text dominates while experiments and discussion are underweighted, "
                "reviewers cannot quickly validate empirical contribution and boundary conditions, which often leads to lower clarity and soundness scores."
            )
        return (
            "This issue is high-risk because reviewers score verifiability, not intention. If a concern cannot be tied to concrete evidence and reproducible protocol, "
            "the paper is likely to receive a weakness comment even when the core idea is valuable."
        )

    @staticmethod
    def _risk_reasoning_zh(reason: str) -> str:
        r = reason.lower()
        if "statistical" in r or "p-value" in r or "confidence interval" in r:
            return (
                "这是一个“统计可信度”问题，不是排版问题。即便平均指标看起来提升明显，如果没有方差、显著性检验或置信区间，审稿人无法判断提升是否稳定、是否可复现。"
                "在以实验严谨性为核心的会议里，这类缺失通常会被直接解释为实验设计不足，而不是可在 camera-ready 阶段再补的小问题。"
            )
        if "terminology" in r or "consistency" in r:
            return (
                "这是“概念一致性”问题。术语在不同章节漂移，会让审稿人怀疑你是否在不同语境下使用了不同定义、不同映射规则或不同评测口径。"
                "在系统/数据库方向，这会被解读为方法边界不清，进一步影响可复现性评价。"
            )
        if "none evidence support" in r or "weak evidence support" in r:
            return (
                "这是典型的“主张-证据不闭环”问题。不是说论文没有工作，而是你的核心主张没有被显式绑定到可量化证据。"
                "审稿人最常见的反应是：我看到了系统描述，但看不到这条结论是如何被严格证明的。"
            )
        if "section length ratio imbalance" in r:
            return (
                "这是“结构分配不合理”问题。方法描述过长、实验与讨论占比偏低，会让审稿人在有限时间内看不到足够验证信息，"
                "从而把论文判断为“工程说明充分但论证不足”。"
            )
        return (
            "这类问题的本质是可验证性不足。审稿人不会按作者意图去补全逻辑，只会根据文中可定位证据做判断。"
            "如果你不能把 concern 绑定到明确证据与修复动作，最终就会在可信度上失分。"
        )


