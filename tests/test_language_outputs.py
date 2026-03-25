from __future__ import annotations

import json
from pathlib import Path

import pytest

from reviewers_sim.models import ReviewRunInput
from reviewers_sim.orchestrator import ReviewOrchestrator


@pytest.fixture
def sample_input(tmp_path: Path) -> Path:
    paper = tmp_path / "paper.md"
    paper.write_text(
        "# Title\n\n## Abstract\nWe improve accuracy.\n\n## Method\nDetails.\n\n## Experiments\nResults and baseline.\n\n## Ablation\nAblation details.\n\n## Limitations\nSome limits.",
        encoding="utf-8",
    )

    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "NeurIPS", "year": 2025},
        "claims": ["We improve accuracy over baselines."],
        "options": {
            "language_mode": "en_zh",
            "executor_backend": "codex",
            "always_export_pdf": False,
        },
    }

    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_bilingual_outputs(sample_input: Path, tmp_path: Path) -> None:
    review_input = ReviewRunInput.model_validate(
        json.loads(sample_input.read_text(encoding="utf-8"))
    )
    orch = ReviewOrchestrator(Path(__file__).resolve().parents[1])
    summary = orch.run(review_input, tmp_path / "runs")

    run_dir = Path(summary.output_dir)
    assert (run_dir / "decision_brief.en.md").exists()
    assert (run_dir / "decision_brief.zh.md").exists()
    assert (run_dir / "full_review.en.json").exists()
    assert (run_dir / "full_review.zh.json").exists()
