from __future__ import annotations

from pathlib import Path

from ..models import RunStatus
from ..services.pdf_export import export_markdown_to_pdf
from ..services.feedback_store import build_feedback_template
from .base import PipelineContext, PipelineStep


class ExporterAndQAGateStep(PipelineStep):
    name = "ExporterAndQAGate"

    def run(self, ctx: PipelineContext) -> None:
        deliverables = self._collect_deliverables(ctx)
        errors = []

        for relpath, content in deliverables.items():
            path = ctx.run_dir / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            if relpath.endswith(".json"):
                ctx.dump_json(relpath, content)
            else:
                # Write Markdown with UTF-8 BOM for better default rendering in Windows editors.
                md_encoding = "utf-8-sig" if relpath.endswith(".md") else "utf-8"
                path.write_text(str(content), encoding=md_encoding)

        if ctx.input_data.options.always_export_pdf:
            md_paths = [ctx.run_dir / rel for rel in deliverables if rel.endswith(".md")]
            for md_path in md_paths:
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
            "3. Add short notes in `comment` when `incorrect`.\n"
            "4. Submit with:\n"
            "   `python -m agent_paper_reviewers.cli submit-feedback --input <path/to/feedback_template.json>`\n\n"
            "After submit, records are saved under `<repo>/feedback/<venue>/<year>/` and will be used to calibrate future runs.\n"
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
                        "3. 对 `incorrect` 条目补充 `comment`（简要说明误判原因）。\n"
                        "4. 使用命令提交：\n"
                        "   `python -m agent_paper_reviewers.cli submit-feedback --input <反馈模板路径>`\n\n"
                        "提交后，反馈会写入 `<repo>/feedback/<venue>/<year>/`，并在后续运行中用于风险校准。\n"
                    ),
                }
            )
        return deliverables
