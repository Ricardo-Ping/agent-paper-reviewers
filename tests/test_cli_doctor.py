from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import agent_paper_reviewers.cli as cli
from agent_paper_reviewers.models import ReviewRunInput
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


def test_validate_executor_or_die_for_openai_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_PAPER_REVIEWERS_CODEX_API_KEY", raising=False)
    paper = tmp_path / "paper.md"
    paper.write_text("# T", encoding="utf-8")
    review_input = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "ICLR", "year": 2026},
            "options": {"executor_backend": "openai"},
        }
    )
    with pytest.raises(Exception):
        cli._validate_executor_or_die(review_input)


def test_run_command_fails_fast_when_openai_key_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_PAPER_REVIEWERS_CODEX_API_KEY", raising=False)
    paper = tmp_path / "paper.md"
    paper.write_text("# T", encoding="utf-8")
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps(
            {
                "paper": {"format": "md", "path": str(paper)},
                "venue": {"name": "ICLR", "year": 2026},
                "claims": ["c1"],
                "options": {"executor_backend": "openai"},
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", "--input", str(input_json), "--output-dir", str(tmp_path / "out")])
    assert result.exit_code != 0
    text = (result.stdout or "") + (result.stderr or "") + (result.output or "")
    assert "FATAL: No real model backend is ready" in text
