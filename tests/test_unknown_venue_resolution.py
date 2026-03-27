from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.executors.deterministic import DeterministicExecutor
from agent_paper_reviewers.models import ReviewRunInput
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_venue_profile import VenueProfileResolverStep


def test_unknown_venue_uses_executor_bootstrap_on_fallback(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Abstract\nA.\n", encoding="utf-8")
    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "UnknownConf", "year": 2026},
        "claims": ["Claim one."],
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "always_export_pdf": False,
        },
    }
    review_input = ReviewRunInput.model_validate(payload)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(
        run_id="run_unknown_bootstrap",
        run_dir=run_dir,
        input_data=review_input,
    )

    step = VenueProfileResolverStep(
        Path(__file__).resolve().parents[1],
        executor=DeterministicExecutor(),
    )
    step.run(ctx)

    vp = ctx.artifacts["venue_profile"]
    checks = vp["profile"]["required_checks"]
    specs = vp["profile"]["required_check_specs"]
    assert vp["used_fallback"] is True
    assert "executor_bootstrap" in vp["source"]
    assert len(checks) >= 10
    assert "contribution_alignment" in checks
    assert "error_analysis" in checks
    assert "robustness_checks" in checks
    assert "practical_impact" in checks
    assert "top_venue_related_work_coverage" in checks
    assert "baseline_coverage" in specs
    assert "statistical_significance" in specs


def test_unknown_venue_marks_openreview_disabled_note(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Abstract\nA.\n", encoding="utf-8")
    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "UnknownConf", "year": 2026},
        "claims": [],
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "always_export_pdf": False,
        },
    }
    review_input = ReviewRunInput.model_validate(payload)
    run_dir = tmp_path / "run2"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(
        run_id="run_unknown_note",
        run_dir=run_dir,
        input_data=review_input,
    )

    step = VenueProfileResolverStep(Path(__file__).resolve().parents[1], executor=None)
    step.run(ctx)
    vp = ctx.artifacts["venue_profile"]
    assert vp["used_fallback"] is True
    assert str(vp["source"]).startswith("fallback_global")
    assert all("policy_resolver_warning" not in x for x in ctx.qa_issues)
