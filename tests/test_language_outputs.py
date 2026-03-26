from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_paper_reviewers.models import ReviewRunInput
from agent_paper_reviewers.orchestrator import ReviewOrchestrator


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
            "executor_backend": "local_vllm",
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
    assert (run_dir / "submission_readiness.en.json").exists()
    assert (run_dir / "submission_readiness.zh.json").exists()
    assert (run_dir / "full_review.en.json").exists()
    assert (run_dir / "full_review.zh.json").exists()
    assert (run_dir / "diagnosis_report.en.md").exists()
    assert (run_dir / "diagnosis_report.zh.md").exists()

    decision_en = json.loads((run_dir / "decision_brief.en.json").read_text(encoding="utf-8"))
    decision_zh = json.loads((run_dir / "decision_brief.zh.json").read_text(encoding="utf-8"))
    assert "score_explanations" in decision_en
    assert "novelty" in decision_en["score_explanations"]
    assert decision_en["score_explanations"]["novelty"]["reasoning"]
    assert "score_explanations" in decision_zh
    assert "paper_qa_gate" in decision_en
    assert "paper_qa_gate" in decision_zh
    assert "submission_readiness" in decision_en
    assert "submission_readiness" in decision_zh
    assert decision_zh["submission_readiness"]["overall_status"] in {"通过", "预警", "阻断"}

