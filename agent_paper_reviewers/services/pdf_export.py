from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PdfExportResult:
    ok: bool
    engine: str | None
    error: str | None


def export_markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> PdfExportResult:
    engines = ["xelatex", "lualatex", "tectonic"]
    is_zh = markdown_path.name.endswith(".zh.md")

    for engine in engines:
        cmd = [
            "pandoc",
            str(markdown_path),
            "-o",
            str(pdf_path),
            f"--pdf-engine={engine}",
        ]

        # Improve CJK rendering in zh PDFs.
        if is_zh:
            cmd.extend([
                "-V",
                "mainfont=Microsoft YaHei",
                "-V",
                "CJKmainfont=Microsoft YaHei",
            ])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return PdfExportResult(ok=False, engine=None, error="pandoc_not_found")

        if result.returncode == 0:
            return PdfExportResult(ok=True, engine=engine, error=None)

    err = (result.stderr or result.stdout or "pdf_export_failed").strip()  # type: ignore[name-defined]
    return PdfExportResult(ok=False, engine=engine, error=err)  # type: ignore[name-defined]
