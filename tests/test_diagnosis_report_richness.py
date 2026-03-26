from __future__ import annotations

import json
from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput
from agent_paper_reviewers.orchestrator import ReviewOrchestrator


def test_diagnosis_report_is_actionable_and_not_placeholder(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text(
        "\n".join(
            [
                "# SQL Dialect Translation",
                "",
                "## Abstract",
                "We propose a system that improves SQL dialect translation quality.",
                "",
                "## Method",
                "Our method combines syntax embedding and iterative translation.",
                "",
                "## Experiments",
                "We compare against baselines and report headline improvements.",
                "",
                "## Limitations",
                "Long nested SQL queries remain difficult.",
            ]
        ),
        encoding="utf-8",
    )

    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": [
            "The system improves translation accuracy against strong baselines.",
            "The method generalizes across dialect pairs.",
        ],
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "mcp_backend": "disabled",
            "always_export_pdf": False,
        },
    }

    review_input = ReviewRunInput.model_validate(payload)
    orch = ReviewOrchestrator(Path(__file__).resolve().parents[1])
    summary = orch.run(review_input, tmp_path / "runs")
    run_dir = Path(summary.output_dir)

    diagnosis_json = json.loads((run_dir / "diagnosis_report.en.json").read_text(encoding="utf-8"))
    diagnosis_md = (run_dir / "diagnosis_report.en.md").read_text(encoding="utf-8-sig")

    assert diagnosis_json["summary"]["risk_count"] >= 1
    item = diagnosis_json["items"][0]
    assert len(str(item.get("root_cause_analysis", ""))) >= 40
    assert len(str(item.get("impact_analysis", ""))) >= 40
    assert len(str(item.get("fix_summary", ""))) >= 25
    assert isinstance(item.get("fix_actions"), list) and len(item.get("fix_actions")) >= 1
    assert "expected_impact" in item and str(item["expected_impact"]).strip()

    assert "## Parse Quality Snapshot" in diagnosis_md
    assert "- Action Plan:" in diagnosis_md
    assert "- Evidence Bundle:" in diagnosis_md or "- Evidence Anchor:" in diagnosis_md
    assert len(diagnosis_md) > 1000
