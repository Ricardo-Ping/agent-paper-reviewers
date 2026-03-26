from __future__ import annotations

from agent_paper_reviewers.pipeline.step_report_builder import ReportBuilderStep


def test_submission_readiness_marks_warning_from_common_gaps() -> None:
    payload = ReportBuilderStep._build_submission_readiness(
        venue_name="ICLR",
        venue_recommendations={
            "recommended_venues": [
                {"venue": "iclr", "match_score": 0.41},
            ]
        },
        gaps=[
            {"code": "missing_significance"},
            {"code": "missing_baseline"},
        ],
        risks=[
            {"id": "RISK-001", "severity": "P1", "reason": "Statistical significance evidence appears missing."}
        ],
        paper_qa_gate={"accepted": True, "rewrites_applied": 1},
        qa_issues=[],
        manuscript_stage="initial_submission",
    )
    assert payload["overall_status"] == "warning"
    assert payload["warning_count"] >= 1
    assert payload["human_review_recommended"] is True
    ids = {row["id"] for row in payload["checks"]}
    assert "CHK-004" in ids
    assert "CHK-003" in ids


def test_submission_readiness_marks_critical_when_claim_p0_exists() -> None:
    payload = ReportBuilderStep._build_submission_readiness(
        venue_name="ICLR",
        venue_recommendations={"recommended_venues": []},
        gaps=[],
        risks=[
            {"id": "RISK-009", "severity": "P0", "reason": "Core claim evidence is contradictory."}
        ],
        paper_qa_gate={"accepted": False, "rewrites_applied": 0},
        qa_issues=["paper_parser_warning:low_text_quality"],
        manuscript_stage="initial_submission",
    )
    assert payload["overall_status"] == "critical"
    assert payload["critical_count"] >= 1
    assert payload["human_review_recommended"] is True
    check_by_id = {row["id"]: row for row in payload["checks"]}
    assert check_by_id["CHK-002"]["status"] == "critical"
    assert check_by_id["CHK-009"]["status"] == "critical"
    assert check_by_id["CHK-008"]["status"] == "warning"
