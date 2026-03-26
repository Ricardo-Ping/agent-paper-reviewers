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
    assert graph["stats"]["top_venue_reference_count"] >= 0
    assert not any("semantic_scholar_rate_limited" in x for x in ctx.qa_issues)


def test_citation_graph_adds_venue_filtered_counts_from_local_references(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "title": "Sample Paper",
        "raw_text": (
            "# Intro\nText.\n\nReferences\n"
            "[1] A method from NeurIPS 2024 for SQL generation.\n"
            "[2] Data layout optimization in SIGMOD 2023.\n"
            "[3] Hybrid query execution in VLDB 2022.\n"
        ),
        "sections": [],
    }

    def fake_fetch(self, title: str):
        return None, ["semantic_scholar_forbidden_check_api_key_or_quota"]

    monkeypatch.setattr(citation_module.SemanticScholarClient, "fetch_citation_graph", fake_fetch)

    CitationGraphStep().run(ctx)
    stats = ctx.artifacts["citation_graph"]["stats"]

    assert stats["venue_reference_counts"]["neurips"] == 1
    assert stats["venue_reference_counts"]["sigmod"] == 1
    assert stats["venue_reference_counts"]["vldb"] == 1
    assert stats["venue_year_reference_counts"]["neurips:2024"] == 1
    assert stats["top_venue_reference_count"] >= 3
    assert not any("semantic_scholar_forbidden_check_api_key_or_quota" in x for x in ctx.qa_issues)


def test_novelty_signal_not_only_incoming_citations_for_recent_paper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now_year = citation_module.datetime.now().year
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "title": "Recent Paper",
        "raw_text": "Main text.",
        "sections": [],
    }

    remote = {
        "paper": {"paper_id": "p1", "title": "Recent Paper", "year": now_year, "url": "", "venue": "icde"},
        "outgoing_references": [
            {"paper_id": "a", "title": "NeurIPS work", "year": now_year, "venue": "neurips"},
            {"paper_id": "b", "title": "SIGMOD work", "year": now_year - 1, "venue": "sigmod"},
            {"paper_id": "c", "title": "VLDB work", "year": now_year - 1, "venue": "vldb"},
            {"paper_id": "d", "title": "ICML work", "year": now_year, "venue": "icml"},
            {"paper_id": "e", "title": "ICLR work", "year": now_year - 1, "venue": "iclr"},
            {"paper_id": "f", "title": "KDD work", "year": now_year, "venue": "kdd"},
            {"paper_id": "g", "title": "AAAI work", "year": now_year, "venue": "aaai"},
            {"paper_id": "h", "title": "ACL work", "year": now_year - 1, "venue": "acl"},
            {"paper_id": "i", "title": "EMNLP work", "year": now_year, "venue": "emnlp"},
            {"paper_id": "j", "title": "CVPR work", "year": now_year, "venue": "cvpr"},
        ],
        "incoming_citations": [],
        "stats": {"incoming_count": 0},
        "source": "semantic_scholar",
    }

    def fake_fetch(self, title: str):
        return remote, []

    monkeypatch.setattr(citation_module.SemanticScholarClient, "fetch_citation_graph", fake_fetch)

    CitationGraphStep().run(ctx)
    score = float(ctx.artifacts["citation_graph"]["stats"]["novelty_signal_score"])
    assert score >= 0.45
    assert ctx.artifacts["citation_graph"]["stats"]["incoming_count"] == 0


def test_citation_graph_infers_supporting_vs_challenging_contexts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "title": "Sample Paper",
        "raw_text": (
            "# Introduction\n"
            "Following [1], we adopt the same decomposition strategy for SQL normalization. "
            "However, unlike [2], our system avoids the brittle rule-only pipeline and improves robustness.\n\n"
            "References\n"
            "[1] A decomposition strategy for SQL translation. SIGMOD 2021.\n"
            "[2] A rule-only SQL rewrite pipeline. VLDB 2020.\n"
        ),
        "sections": [],
    }

    def fake_fetch(self, title: str):
        return None, ["semantic_scholar_rate_limited"]

    monkeypatch.setattr(citation_module.SemanticScholarClient, "fetch_citation_graph", fake_fetch)
    CitationGraphStep().run(ctx)

    graph = ctx.artifacts["citation_graph"]
    refs = graph.get("outgoing_references", [])
    assert len(refs) >= 2

    by_idx = {
        int(x.get("local_ref_index", -1)): x
        for x in refs
        if isinstance(x, dict) and x.get("local_ref_index") is not None
    }
    assert by_idx[1]["citation_stance"] == "supporting"
    assert by_idx[2]["citation_stance"] == "challenging"
    assert float(by_idx[1].get("citation_stance_confidence", 0.0) or 0.0) > 0.0
    assert float(by_idx[2].get("citation_stance_confidence", 0.0) or 0.0) > 0.0

    stats = graph.get("stats", {})
    counts = stats.get("outgoing_stance_counts", {})
    assert int(counts.get("supporting", 0) or 0) >= 1
    assert int(counts.get("challenging", 0) or 0) >= 1
    assert float(stats.get("outgoing_support_ratio", 0.0) or 0.0) > 0.0
    assert float(stats.get("outgoing_challenge_ratio", 0.0) or 0.0) > 0.0


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
            "top_venue_reference_count": 0,
            "recent_top_venue_reference_count": 0,
        },
        "source": "semantic_scholar",
    }

    GapDetectorStep().run(ctx)
    codes = {x["code"] for x in ctx.artifacts["gaps"]["gaps"]}

    assert "missing_reference_coverage" in codes
    assert "weak_novelty_signal_from_citations" in codes


def test_gap_detector_honors_venue_required_check_specs(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {"raw_text": "Baseline and significance are discussed briefly."}
    ctx.artifacts["evidence_index"] = {
        "passages": [
            {"id": "p1", "section": "experiments", "text": "baseline comparison and p-value results"},
            {"id": "p2", "section": "appendix", "text": "confidence interval details"},
        ]
    }
    ctx.artifacts["venue_profile"] = {
        "profile": {
            "required_checks": ["top_venue_related_work_coverage"],
            "required_check_specs": {
                "top_venue_related_work_coverage": {
                    "gap_code": "missing_top_venue_related_work_coverage",
                    "description": "Need enough recent top-venue citations.",
                    "severity_hint": "P1",
                    "keywords": ["related work", "references"],
                    "min_hits": 1,
                    "min_citation_top_venue_recent": 3,
                }
            },
        }
    }
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}
    ctx.artifacts["citation_graph"] = {
        "paper": {"year": 2026},
        "outgoing_references": [{"title": "one top venue ref"}],
        "incoming_citations": [],
        "stats": {
            "outgoing_count": 1,
            "incoming_count": 0,
            "baseline_like_reference_count": 1,
            "reference_coverage_score": 0.2,
            "novelty_signal_score": 0.3,
            "top_venue_reference_count": 1,
            "recent_top_venue_reference_count": 1,
        },
        "source": "local_only",
    }

    GapDetectorStep().run(ctx)
    codes = {x["code"] for x in ctx.artifacts["gaps"]["gaps"]}
    assert "missing_top_venue_related_work_coverage" in codes
    assert "required_check_outcomes" in ctx.artifacts["gaps"]


def test_gap_detector_adds_citation_context_challenge_gap(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {"raw_text": "Main content."}
    ctx.artifacts["evidence_index"] = {
        "passages": [{"id": "p1", "section": "related work", "text": "prior work discussion"}]
    }
    ctx.artifacts["venue_profile"] = {"profile": {"required_checks": []}}
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}
    ctx.artifacts["citation_graph"] = {
        "paper": {"year": 2026},
        "outgoing_references": [
            {"title": "Ref A", "citation_stance": "challenging"},
            {"title": "Ref B", "citation_stance": "challenging"},
            {"title": "Ref C", "citation_stance": "challenging"},
            {"title": "Ref D", "citation_stance": "supporting"},
            {"title": "Ref E", "citation_stance": "neutral"},
            {"title": "Ref F", "citation_stance": "neutral"},
        ],
        "incoming_citations": [],
        "stats": {
            "outgoing_count": 6,
            "incoming_count": 0,
            "baseline_like_reference_count": 2,
            "reference_coverage_score": 0.35,
            "novelty_signal_score": 0.5,
            "content_novelty_score": 0.6,
            "top_venue_reference_count": 3,
            "recent_top_venue_reference_count": 2,
            "outgoing_stance_counts": {"supporting": 1, "challenging": 3, "neutral": 2},
            "outgoing_support_ratio": 0.167,
            "outgoing_challenge_ratio": 0.5,
            "outgoing_stance_context_coverage_ratio": 0.67,
        },
        "source": "local_only",
    }

    GapDetectorStep().run(ctx)
    rows = ctx.artifacts["gaps"]["gaps"]
    codes = {x["code"] for x in rows}
    assert "citation_context_challenge_dominant" in codes
    row = next(x for x in rows if x["code"] == "citation_context_challenge_dominant")
    assert row["evidence_refs"]
    assert any("[challenging]" in str(ref.get("excerpt", "")).lower() for ref in row["evidence_refs"])
