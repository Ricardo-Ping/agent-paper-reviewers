from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_claim_alignment import ClaimEvidenceAlignerStep
from agent_paper_reviewers.pipeline.step_evidence_indexer import EvidenceIndexerStep
from agent_paper_reviewers.pipeline.step_gap_detector import GapDetectorStep


def _ctx(tmp_path: Path) -> PipelineContext:
    paper = tmp_path / "paper.md"
    paper.write_text("# T\n\n## Method\nM\n\n## Experiments\nE\n", encoding="utf-8")
    data = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "ICLR", "year": 2026},
            "claims": ["Our method improves translation accuracy."],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return PipelineContext(run_id="r1", run_dir=run_dir, input_data=data)


def test_evidence_indexer_builds_vector_passages(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "title": "Paper",
        "raw_text": "Table 1 compares baselines. Figure 2 shows errors.",
        "sections": [
            {
                "name": "experiments",
                "text": (
                    "We compare against strong baselines in Table 1.\n\n"
                    "Figure 2 presents qualitative failure cases and ablation behavior."
                ),
            }
        ],
        "pages": [
            {
                "page": 1,
                "text": (
                    "Figure 2: Accuracy across datasets. Ours 92.3%, Baseline 88.1%.\n"
                    "Table 1: Main results\n"
                    "Method Acc Latency\n"
                    "Ours 92.3 12\n"
                    "Base 88.1 19"
                ),
                "tables": [
                    [
                        ["Method", "Acc", "Latency"],
                        ["Ours", "92.3", "12"],
                        ["Base", "88.1", "19"],
                    ]
                ],
            }
        ],
    }

    EvidenceIndexerStep().run(ctx)
    index = ctx.artifacts["evidence_index"]

    assert index["passage_count"] >= 2
    assert index["index_backend"] == "in_memory_semantic_vector"
    assert index["embedding_dim"] > 0
    assert any(p.get("kind") == "figure_table_mention" for p in index["passages"])
    assert any(p.get("kind") == "figure_content" for p in index["passages"])
    assert any(p.get("kind") == "table_data" for p in index["passages"])
    assert any("92.3" in str(p.get("text", "")) for p in index["passages"])
    assert "evidence_vectors" in ctx.artifacts
    assert len(ctx.artifacts["evidence_vectors"]) == index["passage_count"]


def test_evidence_passage_ids_are_traceable_to_section_ids(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "title": "Paper",
        "raw_text": "A",
        "sections": [
            {"section_id": "S001", "section_index": 1, "name": "method", "text": "Method details."},
            {"section_id": "S002", "section_index": 2, "name": "experiments", "text": "Results in Table 1."},
        ],
        "pages": [],
    }

    EvidenceIndexerStep().run(ctx)
    idx = ctx.artifacts["evidence_index"]
    section_ids = {"S001", "S002"}
    locator = idx.get("passage_locator", {})
    assert isinstance(locator, dict)

    for p in idx["passages"]:
        pid = str(p.get("id", ""))
        if "_para" not in pid:
            continue
        sid = pid.split("_", 1)[0]
        assert sid in section_ids
        assert pid in locator
        assert locator[pid]["section_id"] == sid


def test_claim_aligner_outputs_alignment_with_score_breakdown(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["claims_normalized"] = {
        "claims": [
            {
                "claim_id": "C1",
                "claim_text": "Our method improves baseline performance.",
                "verifiable_claim": "Method outperforms baselines in experiments.",
                "success_criteria": "Higher score than baseline.",
            }
        ]
    }
    ctx.artifacts["evidence_index"] = {
        "passages": [
            {
                "id": "sec0_para0",
                "section": "experiments",
                "text": "We outperform strong baselines by 3.4 points in Table 1.",
            },
            {
                "id": "sec0_para1",
                "section": "limitations",
                "text": "We discuss limitations and ethics.",
            },
        ]
    }
    ctx.artifacts["evidence_vectors"] = {
        "sec0_para0": [1.0, 0.0, 0.0],
        "sec0_para1": [0.0, 1.0, 0.0],
    }

    # Monkeypatch embedding generation for query text to deterministic vector.
    from agent_paper_reviewers.pipeline import step_claim_alignment as module

    original_encode = module.encode_texts
    module.encode_texts = lambda texts: ([[1.0, 0.0, 0.0] for _ in texts], "mock")
    try:
        ClaimEvidenceAlignerStep().run(ctx)
    finally:
        module.encode_texts = original_encode

    row = ctx.artifacts["claim_evidence_matrix"]["alignments"][0]
    assert row["strength"] in {"Strong", "Medium"}
    assert row["score"] > 0.5
    assert row["evidence_refs"]
    assert "score_breakdown" in row


def test_gap_detector_uses_required_checks_and_weak_alignment(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "raw_text": "We provide baseline comparisons and ablation studies but no significance tests.",
    }
    ctx.artifacts["venue_profile"] = {
        "profile": {
            "required_checks": [
                "baseline_coverage",
                "statistical_significance",
                "ablation_completeness",
                "reproducibility_details",
            ]
        }
    }
    ctx.artifacts["evidence_index"] = {
        "passages": [
            {
                "id": "p1",
                "section": "experiments",
                "text": "We compare with strong baselines and provide ablation experiments.",
            }
        ]
    }
    ctx.artifacts["claim_evidence_matrix"] = {
        "alignments": [
            {
                "claim_id": "C1",
                "claim_text": "Claim",
                "strength": "Weak",
                "score": 0.34,
                "evidence_refs": [{"section": "experiments", "passage_id": "p1", "excerpt": "..."}],
            }
        ]
    }

    GapDetectorStep().run(ctx)
    gaps = ctx.artifacts["gaps"]["gaps"]
    codes = {g["code"] for g in gaps}

    assert "missing_significance" in codes
    assert "missing_reproducibility" in codes
    assert "weak_claim_alignment" in codes


def test_claim_aligner_detects_negative_evidence_and_gap(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["claims_normalized"] = {
        "claims": [
            {
                "claim_id": "C1",
                "claim_text": "Our method significantly outperforms all baselines on accuracy.",
                "verifiable_claim": "Accuracy is higher than all baselines.",
                "success_criteria": "Higher accuracy than baseline methods with significance.",
                "claim_type": "statistical",
            }
        ]
    }
    ctx.artifacts["evidence_index"] = {
        "passages": [
            {
                "id": "exp_p1",
                "section": "experiments",
                "text": "Table 2 shows our method is worse than baseline on accuracy for two datasets.",
                "quality_score": 0.92,
                "kind": "table_content",
            },
            {
                "id": "exp_p2",
                "section": "experiments",
                "text": "On one benchmark, ours slightly improves accuracy.",
                "quality_score": 0.91,
                "kind": "paragraph",
            },
        ]
    }
    ctx.artifacts["evidence_vectors"] = {
        "exp_p1": [1.0, 0.0, 0.0],
        "exp_p2": [1.0, 0.0, 0.0],
    }

    from agent_paper_reviewers.pipeline import step_claim_alignment as module

    original_encode = module.encode_texts
    module.encode_texts = lambda texts: ([[1.0, 0.0, 0.0] for _ in texts], "mock")
    try:
        ClaimEvidenceAlignerStep().run(ctx)
    finally:
        module.encode_texts = original_encode

    row = ctx.artifacts["claim_evidence_matrix"]["alignments"][0]
    assert bool(row.get("contradiction_detected")) is True
    assert float(row.get("contradiction_score", 0.0)) >= 0.45
    assert row.get("contradictory_evidence_refs")

    ctx.artifacts["paper_structured"] = {"raw_text": "Main text."}
    ctx.artifacts["venue_profile"] = {"profile": {"required_checks": []}}
    ctx.artifacts["citation_graph"] = {"stats": {}}
    GapDetectorStep().run(ctx)
    codes = {g["code"] for g in ctx.artifacts["gaps"]["gaps"]}
    assert "claim_evidence_contradiction" in codes
