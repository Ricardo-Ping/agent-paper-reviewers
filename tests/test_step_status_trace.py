from __future__ import annotations

import json
from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, RunStatus
from agent_paper_reviewers.orchestrator import ReviewOrchestrator


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_result_records_step_statuses_on_success(tmp_path: Path) -> None:
    paper = tmp_path / "ok_paper.md"
    paper.write_text(
        "# Title\n\n## Abstract\nClaim.\n\n## Method\nMethod.\n\n## Experiments\nResults.\n",
        encoding="utf-8",
    )
    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": ["We improve baseline accuracy."],
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "always_export_pdf": False,
        },
    }
    summary = ReviewOrchestrator(_repo_root()).run(
        ReviewRunInput.model_validate(payload),
        tmp_path / "runs",
    )

    assert summary.status in {RunStatus.SUCCESS, RunStatus.PARTIAL_FAILED}
    run_result = json.loads((Path(summary.output_dir) / "run_result.json").read_text(encoding="utf-8"))
    steps = run_result["step_statuses"]
    assert steps
    assert all(s["status"] == "success" for s in steps)
    assert run_result.get("failed_step") is None
    assert "decision_brief.en.md" in run_result["produced_artifacts"]


def test_run_result_records_failed_and_skipped_steps(tmp_path: Path) -> None:
    missing_paper = tmp_path / "missing.md"
    payload = {
        "paper": {"format": "md", "path": str(missing_paper)},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": ["We improve baseline accuracy."],
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "always_export_pdf": False,
        },
    }
    summary = ReviewOrchestrator(_repo_root()).run(
        ReviewRunInput.model_validate(payload),
        tmp_path / "runs",
    )

    assert summary.status == RunStatus.FAILED
    run_dir = Path(summary.output_dir)
    run_result = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    steps = run_result["step_statuses"]
    by_name = {s["name"]: s for s in steps}

    assert by_name["Intake"]["status"] == "success"
    assert by_name["PaperParser"]["status"] == "failed"
    assert run_result.get("failed_step") == "PaperParser"

    # Steps after PaperParser should be skipped and still visible in status trace.
    parser_idx = [idx for idx, step in enumerate(steps) if step["name"] == "PaperParser"][0]
    for step in steps[parser_idx + 1 :]:
        assert step["status"] == "skipped"

    # Partial artifacts should be preserved.
    assert "artifacts/skill_flow_used.json" in run_result["produced_artifacts"]
    assert (run_dir / "pipeline_exception.log").exists()
