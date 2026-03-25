from __future__ import annotations

import json
from pathlib import Path

from reviewers_sim.models import ReviewRunInput
from reviewers_sim.orchestrator import ReviewOrchestrator


def test_rebuttal_char_limit(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text(
        "# Title\n\n## Abstract\nWe propose something.\n\n## Method\nMethod details.\n\n## Experiments\nNeed more evidence.",
        encoding="utf-8",
    )

    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "NeurIPS", "year": 2025},
        "claims": ["Core claim with limited support."],
        "options": {
            "language_mode": "en",
            "executor_backend": "codex",
            "always_export_pdf": False,
        },
    }

    review_input = ReviewRunInput.model_validate(payload)
    orch = ReviewOrchestrator(Path(__file__).resolve().parents[1])
    summary = orch.run(review_input, tmp_path / "runs")

    rebuttal = json.loads((Path(summary.output_dir) / "rebuttal.en.json").read_text(encoding="utf-8"))
    for item in rebuttal["items"]:
        assert item["char_count"] <= item["char_limit"]
        assert item["char_limit"] == 10000
