from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, RunStatus
from agent_paper_reviewers.orchestrator import ReviewOrchestrator


def test_smoke_en_output(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text(
        "# Title\n\n## Abstract\nClaim text.\n\n## Method\nMethod text.\n\n## Experiments\nBaseline comparison available.",
        encoding="utf-8",
    )

    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "ICML", "year": 2026},
        "claims": ["Our method is better."],
        "options": {
            "language_mode": "en",
            "executor_backend": "agent_api",
            "always_export_pdf": False,
        },
    }

    review_input = ReviewRunInput.model_validate(payload)
    orch = ReviewOrchestrator(Path(__file__).resolve().parents[1])
    summary = orch.run(review_input, tmp_path / "runs")

    run_dir = Path(summary.output_dir)
    assert summary.status in {RunStatus.SUCCESS, RunStatus.PARTIAL_FAILED}
    assert (run_dir / "decision_brief.en.md").exists()
    assert (run_dir / "full_review.en.md").exists()
    assert (run_dir / "rebuttal.en.md").exists()

