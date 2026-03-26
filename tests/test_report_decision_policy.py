from __future__ import annotations

from agent_paper_reviewers.pipeline.step_report_builder import ReportBuilderStep


def _risks(p0: int, p1: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(p0):
        rows.append({"id": f"P0-{i}", "severity": "P0"})
    for i in range(p1):
        rows.append({"id": f"P1-{i}", "severity": "P1"})
    return rows


def test_high_competition_venue_treats_two_p1_as_not_ready() -> None:
    decision, policy, _ = ReportBuilderStep._decision(
        risks=_risks(p0=0, p1=2),
        scores={"overall": 7.6},
        venue_name="ICLR",
        venue_profile={},
    )

    assert policy["strictness_tier"] == "high_competition"
    assert decision == "Not Ready"


def test_medium_competition_venue_with_two_p1_is_borderline() -> None:
    decision, policy, _ = ReportBuilderStep._decision(
        risks=_risks(p0=0, p1=2),
        scores={"overall": 6.6},
        venue_name="SIGMOD",
        venue_profile={},
    )

    assert policy["strictness_tier"] == "medium_competition"
    assert decision == "Borderline"


def test_profile_policy_override_takes_priority() -> None:
    decision, policy, _ = ReportBuilderStep._decision(
        risks=_risks(p0=0, p1=2),
        scores={"overall": 6.3},
        venue_name="NeurIPS",
        venue_profile={
            "decision_policy": {
                "strictness_tier": "custom",
                "p1_not_ready_threshold": 5,
                "p1_borderline_threshold": 3,
                "min_overall_ready": 6.2,
                "min_overall_borderline": 5.8,
            }
        },
    )

    assert policy["strictness_tier"] == "custom"
    assert decision == "Ready"
