from __future__ import annotations

import json
from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput
from agent_paper_reviewers.orchestrator import ReviewOrchestrator
from agent_paper_reviewers.services.venue_recommender import recommend_venues


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_recommend_venues_returns_ranked_candidates() -> None:
    repo_root = _repo_root()
    payload = recommend_venues(
        repo_root,
        target_year=2026,
        paper_structured={
            "title": "A Scalable DB Query Engine",
            "sections": [
                {"name": "abstract", "text": "We optimize latency and throughput for OLTP/OLAP workloads."},
                {"name": "experiments", "text": "We evaluate workload diversity, scalability, and benchmark traces."},
            ],
        },
        claims_normalized={
            "claims": [
                {
                    "claim_text": "Our system improves throughput and latency.",
                    "verifiable_claim": "Outperforms baselines under matched settings.",
                }
            ]
        },
        evidence_index={
            "passages": [
                {"section": "experiments", "text": "TPC-C/TPC-H workload benchmarks and scale-out cluster tests."},
                {"section": "experiments", "text": "Throughput-latency trade-off across data sizes."},
            ]
        },
        top_k=5,
    )

    assert payload["recommended_venues"]
    first = payload["recommended_venues"][0]
    assert "venue" in first
    assert "match_score" in first
    assert isinstance(first["reasons"], list)
    assert "rule_readiness" in first
    rr = first["rule_readiness"]
    assert "strict_pass_ratio" in rr
    assert "weighted_coverage" in rr
    assert "formula" in rr
    assert "check_diagnostics" in first
    assert isinstance(first["check_diagnostics"], list)
    assert first["check_diagnostics"]
    assert "required_check_mapping" in first
    assert isinstance(first["required_check_mapping"], list)


def test_pipeline_exports_venue_recommendations(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text(
        "# DB Paper\n\n## Abstract\nWe optimize throughput and latency under workload diversity.\n\n"
        "## Method\nSystem design.\n\n## Experiments\nScale-out evaluation and benchmark traces.\n",
        encoding="utf-8",
    )
    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": ["We improve latency and throughput under scalable workloads."],
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "mcp_backend": "disabled",
            "always_export_pdf": False,
        },
    }
    summary = ReviewOrchestrator(_repo_root()).run(
        ReviewRunInput.model_validate(payload),
        tmp_path / "runs",
    )
    run_dir = Path(summary.output_dir)

    reco_path = run_dir / "venue_recommendations.json"
    assert reco_path.exists()
    reco_payload = json.loads(reco_path.read_text(encoding="utf-8"))
    assert reco_payload["recommended_venues"]
    first = reco_payload["recommended_venues"][0]
    reasons = first.get("reasons", [])
    assert isinstance(reasons, list) and reasons
    # Reasons should contain actionable gap details, not only generic templates.
    assert any("hits=" in str(r) or "missing keywords" in str(r) for r in reasons)

    brief = (run_dir / "decision_brief.en.md").read_text(encoding="utf-8-sig")
    assert "Recommended Venues (If You Are Unsure)" in brief
