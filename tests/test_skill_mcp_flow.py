from __future__ import annotations

import json
from pathlib import Path

from reviewers_sim.models import ReviewRunInput
from reviewers_sim.orchestrator import ReviewOrchestrator


def test_skill_flow_and_mcp_runtime_outputs(tmp_path: Path) -> None:
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
            "executor_backend": "codex",
            "mcp_backend": "http",
            "always_export_pdf": False,
        },
    }

    review_input = ReviewRunInput.model_validate(payload)
    orch = ReviewOrchestrator(Path(__file__).resolve().parents[1])
    summary = orch.run(review_input, tmp_path / "runs")
    run_dir = Path(summary.output_dir)

    skill_flow = json.loads((run_dir / "skill_flow_used.json").read_text(encoding="utf-8"))
    mcp_runtime = json.loads((run_dir / "mcp_runtime.json").read_text(encoding="utf-8"))

    assert "VenueProfileResolver" in skill_flow["steps"]
    assert mcp_runtime["provider"] in {"http_mcp", "noop_mcp"}
    assert "openreview_policy_resolver" in mcp_runtime["capabilities"]

