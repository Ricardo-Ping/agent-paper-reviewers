from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, RunStatus
from agent_paper_reviewers.orchestrator import ReviewOrchestrator
import agent_paper_reviewers.pipeline.step_exporter_qa as exporter_step


def test_pdf_fallback_partial_failed(tmp_path: Path, monkeypatch) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text(
        "# Title\n\n## Abstract\nClaim.\n\n## Method\nMethod.\n\n## Experiments\nExperiments.",
        encoding="utf-8",
    )

    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "NeurIPS", "year": 2025},
        "claims": ["Claim."],
        "options": {
            "language_mode": "en",
            "executor_backend": "codex",
            "always_export_pdf": True,
        },
    }

    class FailResult:
        ok = False
        engine = None
        error = "forced_failure"

    monkeypatch.setattr(exporter_step, "export_markdown_to_pdf", lambda *_args, **_kwargs: FailResult())

    review_input = ReviewRunInput.model_validate(payload)
    orch = ReviewOrchestrator(Path(__file__).resolve().parents[1])
    summary = orch.run(review_input, tmp_path / "runs")

    run_dir = Path(summary.output_dir)
    assert summary.status == RunStatus.PARTIAL_FAILED
    assert (run_dir / "export_errors.log").exists()

