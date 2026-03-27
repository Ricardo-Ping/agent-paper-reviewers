from __future__ import annotations

import json
from pathlib import Path

from agent_paper_reviewers.cli import _build_ai_summary_payload
from agent_paper_reviewers.models import RunStatus, RunSummary


def test_ai_summary_payload_basic(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "decision_brief.en.json").write_text(
        json.dumps(
            {
                "decision": "Not Ready",
                "decision_interpretation": "High reject risk before submission.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "artifacts" / "risk_ranking.json").write_text(
        json.dumps(
            {
                "risks": [
                    {
                        "id": "RISK-001",
                        "severity": "P0",
                        "score": 0.82,
                        "reason": "Statistical significance is missing.",
                        "generation_source": "executor_fallback",
                    },
                    {
                        "id": "RISK-002",
                        "severity": "P2",
                        "score": 0.31,
                        "reason": "Minor clarity issue.",
                        "generation_source": "executor",
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "rebuttal.en.md").write_text("# R", encoding="utf-8")

    summary = RunSummary(
        run_id="run-001",
        status=RunStatus.PARTIAL_FAILED,
        output_dir=str(run_dir),
        qa_issues=["rebuttal_warning:coverage_low"],
        step_statuses=[],
        produced_artifacts=[],
        historical_profile={},
    )
    payload = _build_ai_summary_payload(summary, run_dir)
    assert payload["run_id"] == "run-001"
    assert payload["run_dir"] == str(run_dir.resolve())
    assert payload["status"] == "partial_failed"
    assert payload["verdict"].startswith("Not Ready")
    assert payload["top_risks"][0]["id"] == "RISK-001"
    assert payload["top_risks"][0]["blocking"] is True
    assert payload["must_fix_before_submit"] == ["RISK-001"]
    assert payload["rebuttal_ready"] is not None
    assert isinstance(payload["confidence"], float)
    assert payload["degraded"] is True
    assert payload["student_pack_ready"] is False
    assert isinstance(payload["recommended_next_action"], str)
    assert "step_overview" in payload
    assert "key_files" in payload
    assert "persona_routes" in payload
    assert "persona_playbook" in payload["key_files"]
    assert "chat_summary" in payload["key_files"]
    assert "chat_rebuttal" in payload["key_files"]
    assert "minimal_checks" in payload["persona_routes"]
