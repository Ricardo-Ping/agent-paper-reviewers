from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.services.venue_loader import load_venue_profile


def test_venue_loader_new_layout_exact_and_fallback() -> None:
    root = Path(__file__).resolve().parents[1]

    exact, used_fallback, source = load_venue_profile(root, "ICLR", 2026)
    assert used_fallback is False
    assert source == "exact_match"
    assert exact.openreview_group_id == "ICLR.cc/2026/Conference"
    assert exact.decision_policy.strictness_tier == "high_competition"

    fallback_year, used_fallback2, source2 = load_venue_profile(root, "ICLR", 2027)
    assert used_fallback2 is True
    assert source2 == "fallback_to_2026"
    assert fallback_year.openreview_group_id == "ICLR.cc/2026/Conference"


def test_venue_loader_global_fallback_for_unknown_venue() -> None:
    root = Path(__file__).resolve().parents[1]
    profile, used_fallback, source = load_venue_profile(root, "UNKNOWN-VENUE", 2026)

    assert used_fallback is True
    assert source == "fallback_global"
    assert profile.scoring_axes == ["novelty", "soundness", "experiment", "clarity"]


def test_db_venues_include_database_specific_required_checks() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "baseline_config_fairness",
        "workload_diversity",
        "scalability_evaluation",
        "efficiency_tradeoff_reporting",
        "system_setting_reproducibility",
        "top_venue_related_work_coverage",
    }

    for venue, year in [("SIGMOD", 2024), ("VLDB", 2024), ("ICDE", 2026)]:
        profile, used_fallback, _ = load_venue_profile(root, venue, year)
        assert used_fallback is False
        checks = set(profile.required_checks)
        assert expected.issubset(checks)
        assert profile.decision_policy.strictness_tier == "medium_competition"
