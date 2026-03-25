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
        "pages": [{"page": 1, "text": "Table 1 compares baselines. Figure 2 shows errors."}],
    }

    EvidenceIndexerStep().run(ctx)
    index = ctx.artifacts["evidence_index"]

    assert index["passage_count"] >= 2
    assert index["index_backend"] == "in_memory_semantic_vector"
    assert index["embedding_dim"] > 0
    assert any(p.get("kind") == "figure_table_mention" for p in index["passages"])
    assert "evidence_vectors" in ctx.artifacts
    assert len(ctx.artifacts["evidence_vectors"]) == index["passage_count"]


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
