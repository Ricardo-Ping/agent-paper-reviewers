from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import agent_paper_reviewers.cli as cli


def test_submit_feedback_command_writes_feedback_record(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "schema_version": 1,
        "run_id": "run-1",
        "paper_title": "paper",
        "venue": "ICLR",
        "year": 2026,
        "items": [
            {
                "risk_id": "RISK-001",
                "reason": "Claim C1 has weak evidence support.",
                "likely_reject_phrase": "Core novelty claims are not yet supported by direct evidence.",
                "verdict": "incorrect",
                "comment": "False alarm for this draft.",
            }
        ],
    }
    input_path = tmp_path / "feedback_template.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["submit-feedback", "--input", str(input_path)])
    assert result.exit_code == 0
    assert "accepted_items: 1" in result.stdout

    out_dir = tmp_path / "feedback" / "iclr" / "2026"
    files = list(out_dir.glob("*.json"))
    assert files
