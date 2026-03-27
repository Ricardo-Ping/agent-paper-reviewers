from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_venue_profile import VenueProfileResolverStep


def test_venue_profile_is_local_rule_driven(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Abstract\nA", encoding="utf-8")
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
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = PipelineContext(
        run_id="run_001",
        run_dir=run_dir,
        input_data=review_input,
    )
    step = VenueProfileResolverStep(Path(__file__).resolve().parents[1])
    step.run(ctx)

    venue_profile = ctx.artifacts["venue_profile"]
    profile = venue_profile["profile"]
    assert venue_profile["policy_needs_manual_check"] is False
    assert profile["rebuttal_policy"]["per_review_char_limit"] == 2500
    assert "section_length_ratio" in profile["required_checks"]
    assert "section_length_ratio" in profile["required_check_specs"]
    assert "terminology_consistency" in profile["required_checks"]
    assert "terminology_consistency" in profile["required_check_specs"]


def test_venue_profile_does_not_emit_policy_resolver_warnings(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Abstract\nA", encoding="utf-8")
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
    run_dir = tmp_path / "run2"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(
        run_id="run_002",
        run_dir=run_dir,
        input_data=review_input,
    )

    step = VenueProfileResolverStep(Path(__file__).resolve().parents[1])
    step.run(ctx)
    assert all("policy_resolver_warning" not in x for x in ctx.qa_issues)
