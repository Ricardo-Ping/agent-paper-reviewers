from __future__ import annotations

import json
from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput
from agent_paper_reviewers.orchestrator import ReviewOrchestrator


def test_skill_flow_and_runtime_context_outputs(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text(
        "# Title\n\n## Abstract\nA.\n\n## Method\nB.\n\n## Experiments\nC.\n\n## Limitations\nD.",
        encoding="utf-8",
    )

    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": ["Claim one."],
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "always_export_pdf": False,
        },
    }

    review_input = ReviewRunInput.model_validate(payload)
    orch = ReviewOrchestrator(Path(__file__).resolve().parents[1])
    summary = orch.run(review_input, tmp_path / "runs")
    run_dir = Path(summary.output_dir)

    skill_flow = json.loads((run_dir / "skill_flow_used.json").read_text(encoding="utf-8"))
    runtime_context = json.loads((run_dir / "runtime_context.json").read_text(encoding="utf-8"))

    assert "VenueProfileResolver" in skill_flow["steps"]
    assert "source" in skill_flow
    assert runtime_context["mode"] == "local_skill_tools_only"
    assert runtime_context["rules_source"] == "local_venue_rules"
    assert "notes" in runtime_context
    assert not (run_dir / "mcp_runtime.json").exists()



