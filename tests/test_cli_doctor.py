from __future__ import annotations

import json

from typer.testing import CliRunner

import agent_paper_reviewers.cli as cli
from agent_paper_reviewers.services.pdf_export import PdfToolchainStatus


def test_doctor_reports_pdf_capability_and_hint(monkeypatch) -> None:
    which_map = {
        "python": "C:/python.exe",
        "pandoc": "C:/pandoc.exe",
        "conda": "C:/conda.exe",
    }
    monkeypatch.setattr(cli.shutil, "which", lambda name: which_map.get(name))
    monkeypatch.setattr(
        cli,
        "detect_pdf_export_capability",
        lambda: PdfToolchainStatus(
            pandoc_available=True,
            engines_available=[],
            engines_missing=["xelatex", "lualatex", "tectonic"],
            preferred_engine=None,
        ),
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "agent-paper-reviewers doctor" in result.stdout
    assert "no LaTeX engine" in result.stdout
    assert "hint: install PDF toolchain" in result.stdout


def test_doctor_json_reports_pdf_ready(monkeypatch) -> None:
    which_map = {
        "python": "C:/python.exe",
        "pandoc": "C:/pandoc.exe",
        "xelatex": "C:/xelatex.exe",
        "conda": "C:/conda.exe",
    }
    monkeypatch.setattr(cli.shutil, "which", lambda name: which_map.get(name))
    monkeypatch.setattr(
        cli,
        "detect_pdf_export_capability",
        lambda: PdfToolchainStatus(
            pandoc_available=True,
            engines_available=["xelatex"],
            engines_missing=["lualatex", "tectonic"],
            preferred_engine="xelatex",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pdf_export"]["ready"] is True
    assert payload["pdf_export"]["toolchain"]["preferred_engine"] == "xelatex"
