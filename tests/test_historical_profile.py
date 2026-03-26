from __future__ import annotations

import json
from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput
from agent_paper_reviewers.orchestrator import ReviewOrchestrator
from agent_paper_reviewers.services.historical_profile import (
    load_historical_profile_prior,
    resolve_author_hash,
    update_historical_profiles,
)


def test_historical_profile_accumulates_by_author(tmp_path: Path) -> None:
    payload = {
        "paper": {"format": "md", "path": str(tmp_path / "paper.md")},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": ["c1"],
        "profile": {"author_id": "student-a"},
    }
    (tmp_path / "paper.md").write_text("# T\n\n## Experiments\nE\n", encoding="utf-8")
    review_input = ReviewRunInput.model_validate(payload)

    risk_ranking = {
        "scores": {"novelty": 5.7, "soundness": 4.5, "experiment": 4.6, "clarity": 7.6, "overall": 5.6},
        "risks": [
            {"id": "R1", "severity": "P1", "reason": "Statistical significance evidence appears missing."},
            {"id": "R2", "severity": "P1", "reason": "Ablation study does not look comprehensive."},
        ],
    }
    gaps = {
        "gaps": [
            {"code": "missing_significance"},
            {"code": "missing_ablation"},
        ]
    }
    alignments = {"alignments": [{"strength": "Weak"}]}

    snap1 = update_historical_profiles(
        tmp_path,
        run_id="run-1",
        input_data=review_input,
        risk_ranking=risk_ranking,
        gaps=gaps,
        alignments=alignments,
    )
    snap2 = update_historical_profiles(
        tmp_path,
        run_id="run-2",
        input_data=review_input,
        risk_ranking=risk_ranking,
        gaps=gaps,
        alignments=alignments,
    )

    author_hash = resolve_author_hash(review_input)
    author_profile_path = tmp_path / "profiles" / "authors" / f"{author_hash}.json"
    assert author_profile_path.exists()
    author_doc = json.loads(author_profile_path.read_text(encoding="utf-8"))
    assert author_doc["runs"] == 2
    assert author_doc["weakness_counts"]["statistical_significance"] >= 2
    assert snap1["author_profile"]["runs"] == 1
    assert snap2["author_profile"]["runs"] == 2

    prior = load_historical_profile_prior(tmp_path, review_input)
    assert prior["available"] is True
    assert prior["author_profile"]["runs"] == 2


def test_orchestrator_emits_historical_profile_artifact(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    profile_root = tmp_path / "profile_cache"
    monkeypatch.setenv("AGENT_PAPER_REVIEWERS_PROFILE_ROOT", str(profile_root))

    paper = tmp_path / "paper.md"
    paper.write_text(
        "# T\n\n## Abstract\nClaim\n\n## Method\nM\n\n## Experiments\nE\n",
        encoding="utf-8",
    )
    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": ["Our method improves baseline accuracy."],
        "profile": {"author_id": "student-b"},
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "mcp_backend": "disabled",
            "always_export_pdf": False,
        },
    }
    review_input = ReviewRunInput.model_validate(payload)
    summary = ReviewOrchestrator(repo_root).run(review_input, tmp_path / "runs")

    run_dir = Path(summary.output_dir)
    result = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    assert "historical_profile" in result
    assert (run_dir / "historical_profile.json").exists()
    assert (run_dir / "artifacts" / "historical_profile_prior.json").exists()
    assert "Historical Weakness Profile" in (run_dir / "decision_brief.en.md").read_text(encoding="utf-8-sig")


def test_historical_profile_run_weaknesses_respect_feedback_weight(tmp_path: Path) -> None:
    payload = {
        "paper": {"format": "md", "path": str(tmp_path / "paper.md")},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": ["c1"],
        "profile": {"author_id": "student-c"},
    }
    (tmp_path / "paper.md").write_text("# T\n\n## Experiments\nE\n", encoding="utf-8")
    review_input = ReviewRunInput.model_validate(payload)

    risk_ranking = {
        "scores": {"novelty": 6.0, "soundness": 5.5, "experiment": 5.2, "clarity": 7.0, "overall": 5.9},
        "risks": [
            {
                "id": "R1",
                "severity": "P1",
                "reason": "Statistical significance evidence appears missing.",
                "feedback_adjustment": {
                    "action": "down",
                    "calibration_confidence": 0.9,
                },
            },
            {
                "id": "R2",
                "severity": "P1",
                "reason": "Baseline comparisons are not strong enough.",
            },
        ],
    }

    snap = update_historical_profiles(
        tmp_path,
        run_id="run-weighted-1",
        input_data=review_input,
        risk_ranking=risk_ranking,
        gaps={"gaps": []},
        alignments={"alignments": []},
    )

    run_weaknesses = snap.get("run_weaknesses", [])
    by_name = {str(x.get("name", "")): x for x in run_weaknesses if isinstance(x, dict)}
    assert "statistical_significance" in by_name
    assert "baseline_coverage" in by_name
    assert float(by_name["statistical_significance"].get("weighted_count", 0.0)) < float(
        by_name["statistical_significance"].get("count", 0)
    )
    assert float(by_name["baseline_coverage"].get("weighted_count", 0.0)) >= float(
        by_name["baseline_coverage"].get("count", 0)
    )
