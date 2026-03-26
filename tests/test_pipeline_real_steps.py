from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, TaskResult, TaskSpec
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_claim_alignment import ClaimEvidenceAlignerStep
from agent_paper_reviewers.pipeline.step_evidence_indexer import EvidenceIndexerStep
from agent_paper_reviewers.pipeline.step_gap_detector import GapDetectorStep


class _StatDetectExecutor:
    def execute(self, spec: TaskSpec) -> TaskResult:
        if spec.task_type != "statistical_significance_detection":
            return TaskResult(ok=True, output={})
        return TaskResult(
            ok=True,
            output={
                "signals": {
                    "mean_std": True,
                    "p_value": True,
                    "confidence_interval": True,
                    "seed_reporting": True,
                    "test_name": True,
                },
                "matched_passage_ids": {
                    "mean_std": ["p1"],
                    "p_value": ["p1"],
                    "confidence_interval": ["p1"],
                    "seed_reporting": ["p1"],
                    "test_name": ["p1"],
                },
                "confidence": 0.88,
                "rationale": "Statistical evidence is explicitly present in the selected passage.",
            },
        )


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
    figure_rows = [p for p in index["passages"] if p.get("kind") == "figure_content"]
    assert figure_rows
    assert str(figure_rows[0].get("anchor_label", "")).startswith("Figure")
    locator = figure_rows[0].get("locator", {})
    assert isinstance(locator, dict)
    assert int(locator.get("line_start", 0) or 0) >= 1
    table_rows = [p for p in index["passages"] if p.get("kind") == "table_data"]
    assert table_rows
    table_locator = table_rows[0].get("locator", {})
    assert isinstance(table_locator, dict)
    assert int(table_locator.get("table_index", 0) or 0) >= 1
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
                "section_id": "S002",
                "section_index": 2,
                "section": "experiments",
                "text": "We outperform strong baselines by 3.4 points in Table 1.",
                "page": 4,
                "kind": "table_content",
                "anchor_label": "Table 1",
                "anchor_type": "table",
                "locator": {
                    "source": "page_visual_caption",
                    "line_start": 21,
                    "line_end": 24,
                    "table_index": 1,
                },
            },
            {
                "id": "sec0_para1",
                "section_id": "S003",
                "section_index": 3,
                "section": "limitations",
                "text": "We discuss limitations and ethics.",
                "page": 6,
                "kind": "paragraph",
                "locator": {"source": "section_paragraph", "paragraph_index": 0},
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
    top_ref = row["evidence_refs"][0]
    assert top_ref.get("page") == 4
    assert top_ref.get("anchor_label") == "Table 1"
    assert top_ref.get("section_id") == "S002"
    assert isinstance(top_ref.get("locator", {}), dict)
    assert int(top_ref.get("locator", {}).get("line_start", 0) or 0) >= 1
    assert top_ref.get("confidence_level") in {"Strong", "Medium", "Weak"}
    assert 0.0 <= float(top_ref.get("confidence_score", 0.0) or 0.0) <= 1.0
    assert bool(top_ref.get("conflict_alert", False)) is False
    assert top_ref.get("relation") == "support"
    confidence = row.get("evidence_confidence", {})
    assert isinstance(confidence, dict)
    assert "support" in confidence
    assert "contradiction" in confidence
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


def test_gap_detector_detects_statistical_signals_by_regex(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "raw_text": (
            "We report mean±std across 5 seeds. "
            "Primary metrics are significant with p < 0.05 and 95% confidence interval."
        ),
    }
    ctx.artifacts["venue_profile"] = {
        "profile": {
            "required_checks": ["statistical_significance"],
            "required_check_specs": {
                "statistical_significance": {
                    "gap_code": "missing_significance",
                    "severity_hint": "P1",
                    "min_required_signals": 2,
                }
            },
        }
    }
    ctx.artifacts["evidence_index"] = {
        "passages": [
            {
                "id": "p1",
                "section": "experiments",
                "text": "We report 5 seeds, mean±std, p < 0.05, and 95% confidence interval for all metrics.",
            }
        ]
    }
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}
    ctx.artifacts["citation_graph"] = {"stats": {}}

    GapDetectorStep().run(ctx)
    payload = ctx.artifacts["gaps"]
    codes = {g["code"] for g in payload["gaps"]}
    assert "missing_significance" not in codes

    outcomes = payload["required_check_outcomes"]
    row = next(x for x in outcomes if x["check_name"] == "statistical_significance")
    assert row["passed"] is True
    signals = row["statistical_detection"]["signals"]
    assert signals["mean_std"] is True
    assert signals["p_value"] is True
    assert signals["confidence_interval"] is True


def test_gap_detector_statistical_executor_can_recover_regex_miss(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "raw_text": "The paper says tests were done but formatting is atypical.",
    }
    ctx.artifacts["venue_profile"] = {
        "profile": {
            "required_checks": ["statistical_significance"],
            "required_check_specs": {
                "statistical_significance": {
                    "gap_code": "missing_significance",
                    "severity_hint": "P1",
                    "min_required_signals": 3,
                }
            },
        }
    }
    ctx.artifacts["evidence_index"] = {
        "passages": [
            {
                "id": "p1",
                "section": "experiments",
                "text": "Atypical notation without explicit p-value token in raw parse.",
            }
        ]
    }
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}
    ctx.artifacts["citation_graph"] = {"stats": {}}

    GapDetectorStep(executor=_StatDetectExecutor()).run(ctx)
    payload = ctx.artifacts["gaps"]
    codes = {g["code"] for g in payload["gaps"]}
    assert "missing_significance" not in codes

    outcomes = payload["required_check_outcomes"]
    row = next(x for x in outcomes if x["check_name"] == "statistical_significance")
    assert row["passed"] is True
    assert row["statistical_detection"]["llm"]["used"] is True
    assert row["statistical_detection"]["llm"]["signals"]["p_value"] is True


def test_gap_detector_detects_section_ratio_imbalance(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "raw_text": "draft text",
        "sections": [
            {"section_id": "S001", "section_index": 1, "name": "introduction", "text": "intro " * 140},
            {"section_id": "S002", "section_index": 2, "name": "method", "text": "method " * 520},
            {"section_id": "S003", "section_index": 3, "name": "experiments", "text": "exp " * 40},
            {"section_id": "S004", "section_index": 4, "name": "discussion", "text": "discussion " * 30},
        ],
    }
    ctx.artifacts["venue_profile"] = {
        "profile": {
            "required_checks": ["section_length_ratio"],
            "required_check_specs": {
                "section_length_ratio": {
                    "gap_code": "section_ratio_imbalance",
                    "description": "Section ratio mismatch.",
                    "severity_hint": "P2",
                    "section_ratio_targets": {
                        "introduction": 0.20,
                        "method": 0.30,
                        "experiments": 0.35,
                        "discussion": 0.15,
                    },
                    "section_ratio_tolerance": 0.08,
                    "section_ratio_min_total_words": 300,
                    "section_aliases": {
                        "introduction": ["introduction"],
                        "method": ["method"],
                        "experiments": ["experiments"],
                        "discussion": ["discussion"],
                    },
                }
            },
        }
    }
    ctx.artifacts["evidence_index"] = {
        "passages": [
            {"id": "S001_para0", "section": "introduction", "text": "intro"},
            {"id": "S002_para0", "section": "method", "text": "method"},
            {"id": "S003_para0", "section": "experiments", "text": "experiments"},
            {"id": "S004_para0", "section": "discussion", "text": "discussion"},
        ]
    }
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}
    ctx.artifacts["citation_graph"] = {"stats": {}}

    GapDetectorStep().run(ctx)
    payload = ctx.artifacts["gaps"]
    codes = {g["code"] for g in payload["gaps"]}
    assert "section_ratio_imbalance" in codes

    outcomes = payload["required_check_outcomes"]
    row = next(x for x in outcomes if x["check_name"] == "section_length_ratio")
    assert row["passed"] is False
    assert "section_ratio" in row
    assert row["section_ratio"]["failing_buckets"]

    gap_row = next(x for x in payload["gaps"] if x["code"] == "section_ratio_imbalance")
    assert gap_row["evidence_refs"]
    assert any(str(ref.get("kind", "")) == "section_ratio_anchor" for ref in gap_row["evidence_refs"])


def test_gap_detector_detects_terminology_inconsistency(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "raw_text": "draft text",
        "sections": [
            {
                "section_id": "S001",
                "section_index": 1,
                "name": "method",
                "text": (
                    "We introduce Cross-Dialect Embedding (CDE) for SQL translation. "
                    "The Cross-Dialect Embedding module is optimized end-to-end."
                ),
            },
            {
                "section_id": "S002",
                "section_index": 2,
                "name": "experiments",
                "text": (
                    "CDE (Cross-Domain Embedding) improves robustness in our benchmark. "
                    "The cross dialect embedding block also reduces latency."
                ),
            },
        ],
    }
    ctx.artifacts["venue_profile"] = {
        "profile": {
            "required_checks": ["terminology_consistency"],
            "required_check_specs": {
                "terminology_consistency": {
                    "gap_code": "terminology_inconsistency",
                    "description": "Terminology mismatch.",
                    "severity_hint": "P2",
                    "terminology_min_mentions": 2,
                    "terminology_min_variant_hits": 1,
                }
            },
        }
    }
    ctx.artifacts["evidence_index"] = {
        "passages": [
            {"id": "S001_para0", "section": "method", "text": "Cross-Dialect Embedding (CDE) for SQL translation."},
            {"id": "S002_para0", "section": "experiments", "text": "CDE (Cross-Domain Embedding) improves robustness."},
        ]
    }
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}
    ctx.artifacts["citation_graph"] = {"stats": {}}

    GapDetectorStep().run(ctx)
    payload = ctx.artifacts["gaps"]
    codes = {g["code"] for g in payload["gaps"]}
    assert "terminology_inconsistency" in codes

    outcomes = payload["required_check_outcomes"]
    row = next(x for x in outcomes if x["check_name"] == "terminology_consistency")
    assert row["passed"] is False
    assert row["terminology_consistency"]["issue_count"] >= 1
    assert row["evidence_refs"]


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
    conflict_ref = row["contradictory_evidence_refs"][0]
    assert conflict_ref.get("relation") == "contradiction"
    assert conflict_ref.get("conflict_alert") is True
    assert conflict_ref.get("confidence_level") in {"Strong", "Medium", "Weak"}
    assert 0.0 <= float(conflict_ref.get("confidence_score", 0.0) or 0.0) <= 1.0
    assert bool(row.get("conflict_alert", False)) is True
    confidence = row.get("evidence_confidence", {})
    assert isinstance(confidence, dict)
    assert bool(confidence.get("conflict_alert", False)) is True

    ctx.artifacts["paper_structured"] = {"raw_text": "Main text."}
    ctx.artifacts["venue_profile"] = {"profile": {"required_checks": []}}
    ctx.artifacts["citation_graph"] = {"stats": {}}
    GapDetectorStep().run(ctx)
    codes = {g["code"] for g in ctx.artifacts["gaps"]["gaps"]}
    assert "claim_evidence_contradiction" in codes
