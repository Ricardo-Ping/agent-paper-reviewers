from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.executors.deterministic import DeterministicExecutor
from agent_paper_reviewers.mcp.base import MCPToolProvider, PolicyResolveResult
from agent_paper_reviewers.models import RebuttalPolicy, ReviewRunInput
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_venue_profile import VenueProfileResolverStep


class _DiscoveryMCP(MCPToolProvider):
    def resolve_openreview_policy(self, group_id: str) -> PolicyResolveResult:
        return PolicyResolveResult(policy=None, warning=None)

    def resolve_openreview_policy_by_venue(self, venue_name: str, year: int) -> PolicyResolveResult:
        _ = (venue_name, year)
        return PolicyResolveResult(
            policy=RebuttalPolicy(
                mode="per_review_only",
                per_review_char_limit=3200,
                global_char_limit=0,
                allow_attachment_pdf=False,
                attachment_page_limit=0,
                allow_links=True,
                dynamic_from_openreview=True,
            ),
            profile_overrides={
                "required_checks": ["baseline_coverage", "statistical_significance"],
                "required_check_specs": {
                    "baseline_coverage": {
                        "keywords": ["baseline", "sota"],
                        "min_hits": 2,
                    }
                },
                "common_reject_reasons": ["Recent OpenReview trend: baseline comparisons are often considered insufficient."],
                "dynamic_focus_weaknesses": ["baseline"],
            },
            warning=None,
            resolved_group_id="UNKNOWN.cc/2026/Conference",
        )


def test_unknown_venue_can_use_mcp_discovery(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Abstract\nA.\n", encoding="utf-8")
    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "UnknownConf", "year": 2026},
        "claims": ["Claim one."],
        "options": {
            "language_mode": "en",
            "executor_backend": "codex",
            "mcp_backend": "http",
            "always_export_pdf": False,
        },
    }

    review_input = ReviewRunInput.model_validate(payload)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(
        run_id="run_unknown",
        run_dir=run_dir,
        input_data=review_input,
        mcp_tools=_DiscoveryMCP(),
    )

    step = VenueProfileResolverStep(Path(__file__).resolve().parents[1], executor=None)
    step.run(ctx)

    vp = ctx.artifacts["venue_profile"]
    assert vp["used_fallback"] is True
    assert "openreview_discovered" in vp["source"]
    assert vp["profile"]["rebuttal_policy"]["per_review_char_limit"] == 3200
    assert vp["profile"]["openreview_group_id"] == "UNKNOWN.cc/2026/Conference"
    assert "baseline_coverage" in vp["profile"]["required_checks"]
    assert "baseline_coverage" in vp["profile"]["required_check_specs"]


def test_unknown_venue_executor_bootstrap_does_not_shrink_fallback_checks(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Abstract\nA.\n", encoding="utf-8")
    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "UnknownConf", "year": 2026},
        "claims": ["Claim one."],
        "options": {
            "language_mode": "en",
            "executor_backend": "codex",
            "mcp_backend": "disabled",
            "always_export_pdf": False,
        },
    }
    review_input = ReviewRunInput.model_validate(payload)
    run_dir = tmp_path / "run2"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(
        run_id="run_unknown_bootstrap",
        run_dir=run_dir,
        input_data=review_input,
        mcp_tools=None,
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
