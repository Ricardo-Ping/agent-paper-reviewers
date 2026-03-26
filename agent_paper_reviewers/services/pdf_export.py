from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class PdfExportResult:
    ok: bool
    engine: str | None
    error: str | None


@dataclass
class PdfToolchainStatus:
    pandoc_available: bool
    engines_available: list[str]
    engines_missing: list[str]
    preferred_engine: str | None

    @property
    def ready(self) -> bool:
        return self.pandoc_available and bool(self.engines_available)


def detect_pdf_export_capability(
    which: Callable[[str], str | None] = shutil.which,
) -> PdfToolchainStatus:
    engines = ["xelatex", "lualatex", "tectonic"]
    pandoc_available = which("pandoc") is not None
    engines_available = [engine for engine in engines if which(engine) is not None]
    engines_missing = [engine for engine in engines if engine not in engines_available]
    preferred_engine = engines_available[0] if engines_available else None
    return PdfToolchainStatus(
        pandoc_available=pandoc_available,
        engines_available=engines_available,
        engines_missing=engines_missing,
        preferred_engine=preferred_engine,
    )


def export_markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> PdfExportResult:
    toolchain = detect_pdf_export_capability()
    if not toolchain.pandoc_available:
        return PdfExportResult(ok=False, engine=None, error="pandoc_not_found")
    if not toolchain.engines_available:
        return PdfExportResult(ok=False, engine=None, error="no_pdf_engine_found")

    is_zh = markdown_path.name.endswith(".zh.md")

    for engine in toolchain.engines_available:
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
