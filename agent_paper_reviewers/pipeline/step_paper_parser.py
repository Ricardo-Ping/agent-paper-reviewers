from __future__ import annotations

import re
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

        raw_text = str(structured.get("raw_text", ""))
        sections = self._attach_section_ids(structured.get("sections", []))
        section_names = [str(s.get("name", "")) for s in sections if isinstance(s, dict)]
        warnings = []
        for required in ["method", "experiments", "limitations"]:
            if not any(required in sec for sec in section_names):
                warnings.append(f"missing_section:{required}")

        nonempty_sections = [
            s
            for s in sections
            if isinstance(s, dict) and str(s.get("text", "")).strip()
        ]
        word_count = len(re.findall(r"\b[\w\-]+\b", raw_text))
        section_count = len(nonempty_sections)
        parse_backend = str(structured.get("parse_backend", "unknown"))

        if ctx.input_data.paper.format == "pdf":
            warnings.extend(
                self._pdf_parse_quality_warnings(
                    raw_text=raw_text,
                    word_count=word_count,
                    section_count=section_count,
                    parse_backend=parse_backend,
                )
            )

        payload = {
            "paper_path": str(paper_path),
            "title": structured.get("title", "Untitled"),
            "sections": sections,
            "section_locator": self._build_section_locator(sections),
            "raw_text": raw_text,
            "pages": structured.get("pages", []),
            "parse_backend": parse_backend,
            "parse_quality": {
                "word_count": word_count,
                "section_count": section_count,
            },
            "warnings": warnings,
        }

        for warning in warnings:
            ctx.add_qa_issue(warning)

        ctx.artifacts["paper_structured"] = payload
        ctx.dump_json("artifacts/paper_structured.json", payload)

    @staticmethod
    def _attach_section_ids(raw_sections: object) -> list[dict]:
        out: list[dict] = []
        if not isinstance(raw_sections, list):
            return out

        for idx, sec in enumerate(raw_sections, start=1):
            if not isinstance(sec, dict):
                continue
            item = dict(sec)
            section_id = str(item.get("section_id") or "").strip()
            if not section_id:
                section_id = f"S{idx:03d}"
            item["section_id"] = section_id
            item["section_index"] = idx
            item["name"] = str(item.get("name") or f"section_{idx}").strip().lower()
            item["text"] = str(item.get("text") or "")
            out.append(item)
        return out

    @staticmethod
    def _build_section_locator(sections: list[dict]) -> list[dict]:
        locator: list[dict] = []
        for sec in sections:
            section_id = str(sec.get("section_id") or "").strip()
            if not section_id:
                continue
            text = str(sec.get("text") or "").strip()
            locator.append(
                {
                    "section_id": section_id,
                    "section_index": int(sec.get("section_index") or 0),
                    "name": str(sec.get("name") or ""),
                    "preview": text[:180],
                }
            )
        return locator

    @staticmethod
    def _pdf_parse_quality_warnings(
        *,
        raw_text: str,
        word_count: int,
        section_count: int,
        parse_backend: str,
    ) -> list[str]:
        warnings: list[str] = []

        if word_count < 350:
            warnings.append(
                "pdf_parse_quality_low_word_count:"
                f"{word_count}:verify_text_layer_or_use_ocr_before_submission_review"
            )

        if section_count < 3:
            warnings.append(
                "pdf_parse_quality_low_section_count:"
                f"{section_count}:verify_pdf_structure_or_convert_to_clean_markdown"
            )

        mojibake_patterns = [
            r"\uFFFD",          # replacement character
            r"(?:Ã.|Â.){2,}",   # common latin-1 mojibake
            r"[锟閳楗椤绱閿]",   # frequent corrupted glyphs seen in parsed PDFs
        ]
        noise_hits = sum(len(re.findall(pattern, raw_text)) for pattern in mojibake_patterns)
        if noise_hits >= 6:
            warnings.append(
                "pdf_parse_quality_encoding_noise_detected:"
                f"{noise_hits}:{parse_backend}:verify_pdf_encoding_or_use_ocr_export"
            )

        return warnings
