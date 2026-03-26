from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import agent_paper_reviewers.cli as cli


def test_tool_venue_profile_json_output() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "tool-venue-profile",
            "--venue",
            "ICLR",
            "--year",
            "2026",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert '"venue": "iclr"' in result.stdout
    assert '"required_checks"' in result.stdout


def test_tool_parse_paper_markdown_output_file(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text(
        "# Demo Paper\n\n## Abstract\nA short abstract.\n\n## Method\nMethod text.\n\n## Experiments\nResults.",
        encoding="utf-8",
    )
    out = tmp_path / "parsed.json"
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "tool-parse-paper",
            "--paper-path",
            str(paper),
            "--paper-format",
            "md",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["title"] == "Demo Paper"
    assert payload["paper_format"] == "md"
    assert len(payload["sections"]) >= 3
    assert "raw_text" in payload


def test_tool_format_student_pack(tmp_path: Path) -> None:
    analysis = {
        "paper_title": "Demo",
        "language": "en",
        "decision": {
            "recommendation": "Hold for Revision",
            "one_line_reason": "Two P0 issues are unresolved.",
            "top_issues": [
                {
                    "id": "P0-1",
                    "priority": "P0",
                    "title": "Missing statistical test",
                    "problem": "No p-value reported.",
                    "impact": "Soundness risk.",
                    "evidence_anchor": "Section 6, Table 2",
                    "fix_steps": ["Add 5-seed mean+/-std", "Add paired t-test"],
                    "time_estimate": "2 days",
                    "rebuttal_hint": "R1 significance",
                }
            ],
        },
        "action_items": [
            {
                "id": "A-001",
                "priority": "P0",
                "title": "Add significance block",
                "problem": "No p-value.",
                "evidence_anchor": "Table 2",
                "steps": ["Run paired t-test", "Report p-value in table"],
                "time_estimate": "1 day",
                "gpu_estimate": "4 GPU-hours",
                "rebuttal_link": "R1",
            }
        ],
        "rebuttal_items": [
            {
                "review_id": "R1",
                "concern": "Significance not clear.",
                "risk_id": "P0-1",
                "status": "in_progress",
                "response": "We added significance tests and updated Table 2.",
                "new_evidence": ["Table 2 mean+/-std", "Appendix p-values"],
            }
        ],
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    out_dir = tmp_path / "pack"

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "tool-format-student-pack",
            "--analysis-json",
            str(analysis_path),
            "--output-dir",
            str(out_dir),
            "--language",
            "en",
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "001-submission-decision.md").exists()
    assert (out_dir / "002-action-items.md").exists()
    assert (out_dir / "003-rebuttal-draft.md").exists()
    assert "Missing statistical test" in (out_dir / "001-submission-decision.md").read_text(
        encoding="utf-8"
    )

