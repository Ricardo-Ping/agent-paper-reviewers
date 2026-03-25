from __future__ import annotations

from pathlib import Path

from ..services.paper_parser import parse_markdown, parse_pdf
from .base import PipelineContext, PipelineStep


class PaperParserStep(PipelineStep):
    name = "PaperParser"

    def run(self, ctx: PipelineContext) -> None:
        paper_path = Path(ctx.input_data.paper.path)
        if not paper_path.exists():
            raise FileNotFoundError(f"paper file does not exist: {paper_path}")

        if ctx.input_data.paper.format == "md":
            structured = parse_markdown(paper_path)
        else:
            structured = parse_pdf(paper_path)

        section_names = [s.get("name", "") for s in structured.get("sections", [])]
        warnings = []
        for required in ["method", "experiments", "limitations"]:
            if not any(required in sec for sec in section_names):
                warnings.append(f"missing_section:{required}")

        payload = {
            "paper_path": str(paper_path),
            "title": structured.get("title", "Untitled"),
            "sections": structured.get("sections", []),
            "raw_text": structured.get("raw_text", ""),
            "warnings": warnings,
        }

        if warnings:
            ctx.qa_issues.extend(warnings)

        ctx.artifacts["paper_structured"] = payload
        ctx.dump_json("artifacts/paper_structured.json", payload)
