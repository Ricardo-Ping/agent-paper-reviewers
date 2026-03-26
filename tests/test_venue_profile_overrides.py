from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.mcp.base import MCPToolProvider, PolicyResolveResult
from agent_paper_reviewers.models import RebuttalPolicy, ReviewRunInput
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_venue_profile import VenueProfileResolverStep


class _FakeMCPProvider(MCPToolProvider):
    def resolve_openreview_policy(self, group_id: str) -> PolicyResolveResult:
        return PolicyResolveResult(
            policy=RebuttalPolicy(
                mode="per_review_only",
                per_review_char_limit=4321,
                global_char_limit=0,
                allow_attachment_pdf=False,
                attachment_page_limit=0,
                allow_links=True,
                dynamic_from_openreview=True,
            ),
            profile_overrides={
                "scoring_axes": ["novelty", "soundness", "clarity"],
                "weights": {"novelty": 4, "soundness": 4, "clarity": 2},
                "common_reject_reasons": ["New OpenReview-derived risk."],
            },
            warning="openreview_partial_fields",
        )


class _FailingMCPProvider(MCPToolProvider):
    def resolve_openreview_policy(self, group_id: str) -> PolicyResolveResult:
        raise RuntimeError("network_unavailable")


class _EmptyPolicyMCPProvider(MCPToolProvider):
    def resolve_openreview_policy(self, group_id: str) -> PolicyResolveResult:
        return PolicyResolveResult(policy=None, warning="openreview_forbidden_with_token")


def test_venue_profile_merges_openreview_overrides(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Abstract\nA", encoding="utf-8")

    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": ["Claim one."],
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "mcp_backend": "http",
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
        mcp_tools=_FakeMCPProvider(),
    )

    step = VenueProfileResolverStep(Path(__file__).resolve().parents[1])
    step.run(ctx)

    venue_profile = ctx.artifacts["venue_profile"]
    profile = venue_profile["profile"]

    assert venue_profile["policy_needs_manual_check"] is False
    assert profile["rebuttal_policy"]["per_review_char_limit"] == 4321
    assert profile["scoring_axes"] == ["novelty", "soundness", "clarity"]
    assert "section_length_ratio" in profile["required_checks"]
    assert "section_length_ratio" in profile["required_check_specs"]
    assert "terminology_consistency" in profile["required_checks"]
    assert "terminology_consistency" in profile["required_check_specs"]

    weights = profile["weights"]
    assert abs(sum(weights.values()) - 1.0) < 0.0001
    assert weights["novelty"] == weights["soundness"]
    assert weights["clarity"] < weights["novelty"]

    reasons = profile["common_reject_reasons"]
    assert reasons[0] == "New OpenReview-derived risk."
    assert any("policy_resolver_warning:openreview_partial_fields" == issue for issue in ctx.qa_issues)


def test_venue_profile_silent_fallback_when_mcp_missing(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Abstract\nA", encoding="utf-8")

    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": ["Claim one."],
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "mcp_backend": "disabled",
            "always_export_pdf": False,
        },
    }

    review_input = ReviewRunInput.model_validate(payload)
    run_dir = tmp_path / "run_missing"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(
        run_id="run_missing",
        run_dir=run_dir,
        input_data=review_input,
        mcp_tools=None,
    )

    step = VenueProfileResolverStep(Path(__file__).resolve().parents[1])
    step.run(ctx)

    venue_profile = ctx.artifacts["venue_profile"]
    assert venue_profile["policy_needs_manual_check"] is False
    assert ctx.qa_issues == []
    assert venue_profile["profile"]["rebuttal_policy"]["per_review_char_limit"] == 2500
    assert "section_length_ratio" in venue_profile["profile"]["required_checks"]
    assert "terminology_consistency" in venue_profile["profile"]["required_checks"]


def test_venue_profile_silent_fallback_when_mcp_fails_or_returns_empty_policy(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Abstract\nA", encoding="utf-8")

    base_payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": ["Claim one."],
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "mcp_backend": "http",
            "always_export_pdf": False,
        },
    }

    step = VenueProfileResolverStep(Path(__file__).resolve().parents[1])
    for run_id, provider in [
        ("run_fail", _FailingMCPProvider()),
        ("run_empty", _EmptyPolicyMCPProvider()),
    ]:
        review_input = ReviewRunInput.model_validate(base_payload)
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ctx = PipelineContext(
            run_id=run_id,
            run_dir=run_dir,
            input_data=review_input,
            mcp_tools=provider,
        )

        step.run(ctx)
        venue_profile = ctx.artifacts["venue_profile"]
        assert venue_profile["policy_needs_manual_check"] is False
        assert ctx.qa_issues == []
        assert venue_profile["profile"]["rebuttal_policy"]["per_review_char_limit"] == 2500

