from __future__ import annotations

from pathlib import Path

from ..models import RunStatus
from ..services.pdf_export import export_markdown_to_pdf
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
        }
        ctx.dump_json("run_summary.json", summary)

    @staticmethod
    def _collect_deliverables(ctx: PipelineContext) -> dict[str, object]:
        reports = ctx.artifacts["reports"]
        rebuttal = ctx.artifacts["rebuttal"]

        deliverables: dict[str, object] = {
            "decision_brief.en.md": reports["en"]["decision_md"],
            "decision_brief.en.json": reports["en"]["decision_json"],
            "full_review.en.md": reports["en"]["full_md"],
            "full_review.en.json": reports["en"]["full_json"],
            "rebuttal.en.md": rebuttal["en"]["markdown"],
            "rebuttal.en.json": rebuttal["en"]["bundle"],
            "claim_evidence_matrix.json": ctx.artifacts["claim_evidence_matrix"],
            "remediation_plan.json": ctx.artifacts["remediation_plan"],
            "venue_profile_used.json": ctx.artifacts["venue_profile"],
            "skill_flow_used.json": ctx.artifacts.get("skill_flow", {}),
            "mcp_runtime.json": ctx.artifacts.get("mcp_runtime", {}),
        }

        if ctx.input_data.options.language_mode.value == "en_zh":
            deliverables.update(
                {
                    "decision_brief.zh.md": reports["zh"]["decision_md"],
                    "decision_brief.zh.json": reports["zh"]["decision_json"],
                    "full_review.zh.md": reports["zh"]["full_md"],
                    "full_review.zh.json": reports["zh"]["full_json"],
                    "rebuttal.zh.md": rebuttal["zh"]["markdown"],
                    "rebuttal.zh.json": rebuttal["zh"]["bundle"],
                }
            )
        return deliverables
