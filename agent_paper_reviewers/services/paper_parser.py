from __future__ import annotations

import re
from pathlib import Path

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    fitz = None

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    pdfplumber = None

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
    pages, backend = _extract_pdf_pages(path)
    if not pages:
        raise RuntimeError("unable_to_parse_pdf_with_available_backends")

    raw = "\n\n".join(page["text"] for page in pages if page["text"].strip())
    sections = _naive_sections_from_text(raw)
    title = _extract_title(raw.splitlines())
    return {
        "title": title,
        "sections": sections,
        "raw_text": raw,
        "pages": pages,
        "parse_backend": backend,
    }


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
        if matched and _is_heading_line(line):
            if current["text_lines"]:
                chunks.append({"name": current["name"], "text": "\n".join(current["text_lines"])})
            current = {"name": matched, "text_lines": []}
            continue

        normalized_heading = _normalize_heading_label(line)
        if normalized_heading and _is_heading_line(line):
            if current["text_lines"]:
                chunks.append({"name": current["name"], "text": "\n".join(current["text_lines"])})
            current = {"name": normalized_heading, "text_lines": []}
            continue

        current["text_lines"].append(line)

    if current["text_lines"]:
        chunks.append({"name": current["name"], "text": "\n".join(current["text_lines"])})

    if not chunks:
        return [{"name": name, "text": ""} for name in SECTION_FALLBACK]
    return chunks


def _extract_pdf_pages(path: Path) -> tuple[list[dict], str]:
    pages = _extract_with_pymupdf(path)
    if pages:
        return pages, "pymupdf"

    pages = _extract_with_pdfplumber(path)
    if pages:
        return pages, "pdfplumber"

    pages = _extract_with_pypdf(path)
    if pages:
        return pages, "pypdf"

    return [], "none"


def _extract_with_pymupdf(path: Path) -> list[dict]:
    if fitz is None:
        return []
    try:
        doc = fitz.open(str(path))
    except Exception:  # noqa: BLE001
        return []

    pages: list[dict] = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        pages.append({"page": i + 1, "text": text})
    return pages


def _extract_with_pdfplumber(path: Path) -> list[dict]:
    if pdfplumber is None:
        return []
    pages: list[dict] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append({"page": i + 1, "text": text})
    except Exception:  # noqa: BLE001
        return []
    return pages


def _extract_with_pypdf(path: Path) -> list[dict]:
    if PdfReader is None:
        return []
    try:
        reader = PdfReader(str(path))
    except Exception:  # noqa: BLE001
        return []

    pages: list[dict] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text})
    return pages


def _is_heading_line(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line.strip())
    if not compact:
        return False
    if len(compact.split()) > 14:
        return False
    if re.fullmatch(r"[\dIVXivx\.\s]{1,12}", compact):
        return False
    alpha_ratio = sum(ch.isalpha() for ch in compact) / max(1, len(compact))
    return alpha_ratio >= 0.45


def _normalize_heading_label(line: str) -> str | None:
    cleaned = re.sub(r"^\d+(\.\d+)*\s*", "", line.strip()).lower()
    patterns = [
        ("abstract", r"\babstract\b"),
        ("introduction", r"\bintroduction\b"),
        ("related work", r"\brelated work\b"),
        ("method", r"\b(method|approach|model)\b"),
        ("experiments", r"\b(experiment|evaluation|results)\b"),
        ("ablation", r"\bablation\b"),
        ("analysis", r"\banalysis\b"),
        ("limitations", r"\blimitations?\b"),
        ("conclusion", r"\bconclusion\b"),
        ("appendix", r"\bappendix\b"),
    ]
    for label, pattern in patterns:
        if re.search(pattern, cleaned):
            return label
    return None
