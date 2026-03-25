from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_citation_graph import CitationGraphStep
from agent_paper_reviewers.pipeline.step_gap_detector import GapDetectorStep
from agent_paper_reviewers.services import citation_graph as citation_module


def _ctx(tmp_path: Path) -> PipelineContext:
    paper = tmp_path / "paper.md"
    paper.write_text("# T", encoding="utf-8")
    data = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "ICLR", "year": 2026},
            "claims": ["c1"],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return PipelineContext(run_id="r1", run_dir=run_dir, input_data=data)


def test_citation_graph_step_falls_back_to_local_references(tmp_path: Path, monkeypatch) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "title": "Sample Paper",
        "raw_text": (
            "# Intro\nSome text.\n\nReferences\n"
            "[1] Strong Baseline for SQL Translation. 2021.\n"
            "[2] Benchmarking LLM-based SQL Systems. 2022.\n"
        ),
        "sections": [],
    }

    def fake_fetch(self, title: str):
        return None, ["semantic_scholar_rate_limited"]

    monkeypatch.setattr(citation_module.SemanticScholarClient, "fetch_citation_graph", fake_fetch)

    CitationGraphStep().run(ctx)

    graph = ctx.artifacts["citation_graph"]
    assert graph["source"] == "local_only"
    assert graph["stats"]["outgoing_count"] >= 2
    assert not any("semantic_scholar_rate_limited" in x for x in ctx.qa_issues)


def test_gap_detector_adds_citation_coverage_gaps(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {"raw_text": "Main content with little related work."}
    ctx.artifacts["evidence_index"] = {
        "passages": [{"id": "p1", "section": "related work", "text": "few references"}]
    }
    ctx.artifacts["venue_profile"] = {"profile": {"required_checks": []}}
    ctx.artifacts["claim_evidence_matrix"] = {
        "alignments": [
            {
                "claim_id": "C1",
                "claim_text": "c1",
                "strength": "Strong",
                "score": 0.8,
                "evidence_refs": [],
            }
        ]
    }
    ctx.artifacts["citation_graph"] = {
        "paper": {"year": 2022},
        "outgoing_references": [{"title": "A survey", "paper_id": "", "year": 2020}],
        "incoming_citations": [],
        "stats": {
            "outgoing_count": 1,
            "incoming_count": 0,
            "baseline_like_reference_count": 0,
            "reference_coverage_score": 0.1,
            "novelty_signal_score": 0.1,
        },
        "source": "semantic_scholar",
    }

    GapDetectorStep().run(ctx)
    codes = {x["code"] for x in ctx.artifacts["gaps"]["gaps"]}

    assert "missing_reference_coverage" in codes
    assert "weak_novelty_signal_from_citations" in codes
