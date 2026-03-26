from __future__ import annotations

from agent_paper_reviewers.pipeline.step_report_builder import ReportBuilderStep


def test_default_fix_actions_are_gap_specific_and_not_reused_verbatim() -> None:
    statistical_risk = {
        "id": "RISK-STAT",
        "reason": "Statistical significance evidence appears incomplete.",
        "severity": "P1",
    }
    baseline_risk = {
        "id": "RISK-BASE",
        "reason": "Core comparisons should include stronger baselines.",
        "severity": "P1",
    }

    baseline_check_trace = {
        "gap_code": "missing_baseline",
        "check_name": "baseline_coverage",
        "description": "Need stronger baseline coverage.",
    }

    stat_actions = ReportBuilderStep._default_fix_actions(
        risk=statistical_risk,
        linked_tasks=[],
        related_claims=[],
        check_trace=baseline_check_trace,
    )
    base_actions = ReportBuilderStep._default_fix_actions(
        risk=baseline_risk,
        linked_tasks=[],
        related_claims=[],
        check_trace=baseline_check_trace,
    )

    assert stat_actions and base_actions
    assert stat_actions[0]["action"] != base_actions[0]["action"]
    assert "significance" in stat_actions[0]["action"].lower() or "mean" in stat_actions[0]["action"].lower()
    assert "baseline" in base_actions[0]["action"].lower()

    stat_action_blob = " ".join(str(x.get("action", "")) for x in stat_actions).lower()
    assert "baseline_coverage" not in stat_action_blob


def test_check_trace_not_forced_when_reason_has_no_overlap() -> None:
    risk = {"reason": "Writing clarity is still weak in introduction."}
    checks = [
        {
            "check_name": "baseline_coverage",
            "gap_code": "missing_baseline",
            "description": "Need baseline coverage",
            "passed": False,
        },
        {
            "check_name": "statistical_significance",
            "gap_code": "missing_significance",
            "description": "Need significance tests",
            "passed": False,
        },
    ]

    trace = ReportBuilderStep._check_trace_for_risk(risk, checks)
    assert trace == {}

