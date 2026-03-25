from __future__ import annotations

import re
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency fallback
    PdfReader = None


SECTION_FALLBACK = [
    "title",
    "abstract",
    "method",
    "experiments",
    "ablation",
    "limitations",
    "appendix",
]


def parse_markdown(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    sections: list[dict] = []
    current = {"name": "preamble", "content": []}

    for line in lines:
        if line.startswith("#"):
            if current["content"]:
                sections.append({"name": current["name"], "text": "\n".join(current["content"]).strip()})
            current = {"name": line.lstrip("#").strip().lower(), "content": []}
        else:
            current["content"].append(line)

    if current["content"]:
        sections.append({"name": current["name"], "text": "\n".join(current["content"]).strip()})

    return {
        "title": _extract_title(lines),
        "sections": sections,
        "raw_text": raw,
    }


def parse_pdf(path: Path) -> dict:
    if PdfReader is None:
        raise RuntimeError("pypdf is required for PDF parsing but is not installed.")

    reader = PdfReader(str(path))
    pages = []
    all_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text})
        all_text.append(text)

    raw = "\n\n".join(all_text)
    sections = _naive_sections_from_text(raw)
    title = _extract_title(raw.splitlines())
    return {"title": title, "sections": sections, "raw_text": raw, "pages": pages}


def split_passages(structured: dict) -> list[dict]:
    passages: list[dict] = []
    for sec in structured.get("sections", []):
        name = sec.get("name", "unknown")
        blocks = re.split(r"\n\s*\n", sec.get("text", ""))
        for idx, block in enumerate(blocks):
            clean = block.strip()
            if not clean:
                continue
            passages.append(
                {
                    "id": f"{name}:{idx}",
                    "section": name,
                    "text": clean,
                }
            )
    return passages


def _extract_title(lines: list[str]) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped:
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            return stripped[:180]
    return "Untitled"


def _naive_sections_from_text(raw_text: str) -> list[dict]:
    patterns = {
        "abstract": r"\babstract\b",
        "method": r"\b(method|approach|model)\b",
        "experiments": r"\b(experiment|evaluation|results)\b",
        "ablation": r"\bablation\b",
        "limitations": r"\b(limitations?|failure cases?)\b",
        "appendix": r"\bappendix\b",
    }

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    chunks: list[dict] = []
    current = {"name": "body", "text_lines": []}

    for line in lines:
        lowered = line.lower()
        matched = None
        for name, pattern in patterns.items():
            if re.search(pattern, lowered):
                matched = name
                break
        if matched and len(line.split()) <= 12:
            if current["text_lines"]:
                chunks.append({"name": current["name"], "text": "\n".join(current["text_lines"])})
            current = {"name": matched, "text_lines": []}
            continue
        current["text_lines"].append(line)

    if current["text_lines"]:
        chunks.append({"name": current["name"], "text": "\n".join(current["text_lines"])})

    if not chunks:
        return [{"name": name, "text": ""} for name in SECTION_FALLBACK]
    return chunks
