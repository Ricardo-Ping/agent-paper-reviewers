from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_paper_reviewers.models import ReviewRunInput, RunStatus
from agent_paper_reviewers.orchestrator import ReviewOrchestrator


def _make_real_pdf(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "Cracking SQL Barriers: A Dialect Translation System\n\n"
        "Abstract\n"
        "We propose an LLM-based SQL dialect translation system with strong transfer performance.\n\n"
        "Method\n"
        "The system combines schema-aware prompting and constrained decoding.\n\n"
        "Experiments\n"
        "We compare with rule-based and neural baselines under matched settings.\n"
        "Results are reported with multi-seed statistics.\n\n"
        "Limitations\n"
        "The current system has lower performance on long nested queries.\n\n"
        "References\n"
        "[1] Strong Baseline for SQL Translation. 2021.\n"
        "[2] Benchmarking LLM-based SQL Systems. 2022.\n"
    )
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def test_full_pipeline_with_real_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "real_input.pdf"
    _make_real_pdf(pdf_path)

    payload = {
        "paper": {"format": "pdf", "path": str(pdf_path)},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": [
            "The method improves SQL dialect translation quality across dialect pairs.",
            "The method generalizes better than prior baselines.",
        ],
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "always_export_pdf": False,
        },
    }
    review_input = ReviewRunInput.model_validate(payload)

    orch = ReviewOrchestrator(Path(__file__).resolve().parents[1])
    summary = orch.run(review_input, tmp_path / "runs")

    assert summary.status in {RunStatus.SUCCESS, RunStatus.PARTIAL_FAILED}
    run_dir = Path(summary.output_dir)
    assert (run_dir / "decision_brief.en.md").exists()
    assert (run_dir / "full_review.en.md").exists()
    assert (run_dir / "diagnosis_report.en.md").exists()
    assert (run_dir / "rebuttal.en.md").exists()

    citation_graph = json.loads((run_dir / "artifacts" / "citation_graph.json").read_text(encoding="utf-8"))
    assert "stats" in citation_graph
    assert "reference_coverage_score" in citation_graph["stats"]
    assert "novelty_signal_score" in citation_graph["stats"]

    gaps = json.loads((run_dir / "artifacts" / "gaps.json").read_text(encoding="utf-8"))
    assert "citation_graph_summary" in gaps
    assert "reference_coverage_score" in gaps["citation_graph_summary"]
    assert "novelty_signal_score" in gaps["citation_graph_summary"]
