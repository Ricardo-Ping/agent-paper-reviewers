from __future__ import annotations

from agent_paper_reviewers.pipeline.step_report_builder import ReportBuilderStep
from agent_paper_reviewers.services import citation_graph as citation_module


def test_score_leverage_prioritizes_high_weight_low_score_axis() -> None:
    payload = ReportBuilderStep._score_leverage_analysis(
        scores={
            "novelty": 5.7,
            "soundness": 4.5,
            "experiment": 4.6,
            "clarity": 7.6,
            "overall": 5.6,
        },
        decision_policy={"min_overall_ready": 7.0},
        venue_profile={"weights": {"novelty": 0.25, "soundness": 0.30, "experiment": 0.30, "clarity": 0.15}},
    )
    assert payload["fastest_axis"] == "soundness"
    assert payload["axes"][0]["axis"] == "soundness"
    assert payload["axes"][0]["priority_index"] >= payload["axes"][1]["priority_index"]


def test_score_leverage_reports_no_urgent_axis_when_all_meet_target() -> None:
    payload = ReportBuilderStep._score_leverage_analysis(
        scores={
            "novelty": 7.7,
            "soundness": 7.0,
            "experiment": 7.1,
            "clarity": 8.2,
            "overall": 7.39,
        },
        decision_policy={"min_overall_ready": 7.0},
        venue_profile={"weights": {"novelty": 0.25, "soundness": 0.30, "experiment": 0.30, "clarity": 0.15}},
    )
    assert payload["fastest_axis"] == "none"
    assert payload["no_urgent_axis"] is True
    assert sum(x["weighted_gap_to_target"] for x in payload["axes"]) == 0.0


def test_content_novelty_signal_works_when_citation_graph_missing(monkeypatch) -> None:
    def fake_fetch(self, title: str):
        return None, ["semantic_scholar_rate_limited"]

    monkeypatch.setattr(citation_module.SemanticScholarClient, "fetch_citation_graph", fake_fetch)
    graph = citation_module.build_citation_graph(
        {
            "title": "A Novel End-to-End SQL Dialect Translation System",
            "raw_text": (
                "Abstract We propose a novel framework for cross-dialect SQL translation. "
                "Introduction This is the first end-to-end system-level pipeline for this setting. "
                "Method Our architecture introduces a new module and mechanism. "
                "Experiments We provide benchmark and ablation results."
            ),
            "sections": [
                {"name": "abstract", "text": "We propose a novel framework for cross-dialect SQL translation."},
                {"name": "introduction", "text": "This is the first end-to-end system-level pipeline."},
                {"name": "method", "text": "Our architecture introduces a new module and mechanism."},
                {"name": "experiments", "text": "We provide benchmark and ablation results."},
                {"name": "conclusion", "text": "We define a new problem setting."},
            ],
        }
    )
    stats = graph.get("stats", {})
    assert float(stats.get("content_novelty_score", 0.0)) >= 0.5
    assert "content_novelty_components" in stats
