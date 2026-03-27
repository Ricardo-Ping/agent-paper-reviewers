from __future__ import annotations

import json
from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput
from agent_paper_reviewers.models import TaskResult, TaskSpec
from agent_paper_reviewers.orchestrator import ReviewOrchestrator
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_venue_recommender import VenueRecommenderStep
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
    assert any(
        token in str(r)
        for r in reasons
        for token in ("hits=", "missing keywords", "venue-specific gaps", "pre-submit weakness", "gap")
    )

    brief = (run_dir / "decision_brief.en.md").read_text(encoding="utf-8-sig")
    assert "Recommended Venues (If You Are Unsure)" in brief


class _FlatVenueScoreExecutor:
    def execute(self, spec: TaskSpec) -> TaskResult:
        if spec.task_type != "venue_recommend":
            return TaskResult(ok=True, output={})
        return TaskResult(
            ok=True,
            output={
                "recommended_venues": [
                    {
                        "venue": "sigmod",
                        "year": 2026,
                        "semantic_fit_score": 0.74,
                        "review_risk_score": 0.33,
                        "match_score": 0.74,
                        "reasons": ["Good fit for systems work."],
                        "fit_summary": "fit",
                        "specific_gap_summary": "gap",
                        "required_check_passed_count": 3,
                        "required_check_total": 5,
                        "passed_checks": ["baseline_coverage"],
                        "failed_checks": ["statistical_significance"],
                    },
                    {
                        "venue": "vldb",
                        "year": 2026,
                        "semantic_fit_score": 0.74,
                        "review_risk_score": 0.33,
                        "match_score": 0.74,
                        "reasons": ["Good fit for systems work."],
                        "fit_summary": "fit",
                        "specific_gap_summary": "gap",
                        "required_check_passed_count": 3,
                        "required_check_total": 5,
                        "passed_checks": ["baseline_coverage"],
                        "failed_checks": ["statistical_significance", "scalability_evaluation"],
                    },
                    {
                        "venue": "icde",
                        "year": 2026,
                        "semantic_fit_score": 0.74,
                        "review_risk_score": 0.33,
                        "match_score": 0.74,
                        "reasons": ["Good fit for systems work."],
                        "fit_summary": "fit",
                        "specific_gap_summary": "gap",
                        "required_check_passed_count": 2,
                        "required_check_total": 5,
                        "passed_checks": ["baseline_coverage"],
                        "failed_checks": ["statistical_significance", "scalability_evaluation", "workload_diversity"],
                    },
                ],
                "method_note": "mock",
            },
        )


def test_agent_venue_recommendation_enforces_score_dispersion(tmp_path: Path) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# T\n\n## Abstract\nSystem paper\n", encoding="utf-8")
    input_data = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "SIGMOD", "year": 2026},
            "claims": ["We improve SQL translation."],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(run_id="r1", run_dir=run_dir, input_data=input_data)

    step = VenueRecommenderStep(executor=_FlatVenueScoreExecutor())
    payload = {
        "recommended_venues": [
            {"venue": "sigmod", "year": 2026, "match_score": 0.73, "reasons": [], "passed_checks": ["baseline_coverage"], "failed_checks": ["statistical_significance"]},
            {"venue": "vldb", "year": 2026, "match_score": 0.72, "reasons": [], "passed_checks": ["baseline_coverage"], "failed_checks": ["statistical_significance", "scalability_evaluation"]},
            {"venue": "icde", "year": 2026, "match_score": 0.71, "reasons": [], "passed_checks": ["baseline_coverage"], "failed_checks": ["statistical_significance", "scalability_evaluation", "workload_diversity"]},
        ],
        "target_year": 2026,
        "candidate_venues_considered": 3,
    }
    out = step._recommend_with_executor(
        ctx=ctx,
        payload=payload,
        paper_structured={"title": "T", "summary": "S", "sections": [{"name": "abstract", "text": "system db"}]},
        claims_normalized={"claims": [{"claim_id": "C1", "claim_type": "novelty", "claim_text": "t"}]},
    )
    assert isinstance(out, dict)
    rows = out["recommended_venues"]
    scores = [float(r.get("match_score", 0.0)) for r in rows]
    assert all(0.55 <= s <= 0.90 for s in scores)
    assert len({round(s, 3) for s in scores}) >= 2
