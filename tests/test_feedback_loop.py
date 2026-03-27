from __future__ import annotations

import json
from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, TaskResult, TaskSpec
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_risk_ranker import RiskRankerStep
from agent_paper_reviewers.services.feedback_store import make_risk_fingerprint, submit_feedback


class _FakeExecutor:
    def __init__(self, output: dict, ok: bool = True) -> None:
        self.output = output
        self.ok = ok

    def execute(self, spec: TaskSpec) -> TaskResult:
        return TaskResult(ok=self.ok, output=self.output)


class _CaptureExecutor:
    def __init__(self, output: dict, ok: bool = True) -> None:
        self.output = output
        self.ok = ok
        self.last_spec: TaskSpec | None = None

    def execute(self, spec: TaskSpec) -> TaskResult:
        self.last_spec = spec
        return TaskResult(ok=self.ok, output=self.output)


def _ctx(tmp_path: Path) -> PipelineContext:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Experiments\nnumbers.\n", encoding="utf-8")
    data = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "ICLR", "year": 2026},
            "claims": ["claim 1"],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return PipelineContext(
        run_id="run-feedback",
        run_dir=run_dir,
        input_data=data,
        repo_root=tmp_path,
    )


def test_risk_ranker_applies_feedback_adjustment(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}
    ctx.artifacts["gaps"] = {"gaps": []}
    ctx.artifacts["venue_profile"] = {"profile": {"weights": {}}}

    reason = "Statistical significance evidence appears missing."
    phrase = "Improvements are not statistically validated to reviewer standards."
    fingerprint = make_risk_fingerprint(reason, phrase)

    feedback_payload = {
        "schema_version": 1,
        "run_id": "historical-run",
        "paper_title": "paper",
        "venue": "ICLR",
        "year": 2026,
        "items": [
            {
                "risk_id": "RISK-001",
                "risk_fingerprint": fingerprint,
                "reason": reason,
                "likely_reject_phrase": phrase,
                "verdict": "incorrect",
                "confidence": 0.95,
                "comment": "False positive in my setting.",
            },
            {
                "risk_id": "RISK-001",
                "risk_fingerprint": fingerprint,
                "reason": reason,
                "likely_reject_phrase": phrase,
                "verdict": "incorrect",
                "confidence": 0.9,
                "comment": "Repeated mismatch.",
            },
        ],
    }
    submit_feedback(tmp_path, feedback_payload)

    step = RiskRankerStep(
        _FakeExecutor(
            {
                "risks": [
                    {
                        "id": "RISK-001",
                        "severity": "P1",
                        "score": 0.70,
                        "reason": reason,
                        "evidence_refs": [],
                        "likely_reject_phrase": phrase,
                        "fix_hint": "Add significance tests.",
                    }
                ],
                "scores": {
                    "novelty": 6.0,
                    "soundness": 6.0,
                    "experiment": 6.0,
                    "clarity": 6.0,
                    "overall": 6.0,
                },
            }
        )
    )
    step.run(ctx)

    ranking = ctx.artifacts["risk_ranking"]
    risk = ranking["risks"][0]
    assert float(risk["score"]) < 0.70
    assert float(risk["score"]) > 0.50
    assert risk["severity"] == "P1"
    assert ranking["feedback_loop"]["matched_risks"] == 1
    assert ranking["feedback_loop"]["scores_recomputed"] is True
    adjustment = risk.get("feedback_adjustment", {})
    assert adjustment.get("action") == "down"
    assert float(adjustment.get("weighted_incorrect", 0.0)) > float(adjustment.get("weighted_correct", 0.0))
    assert float(adjustment.get("calibration_confidence", 0.0)) > 0.0
    assert (ctx.run_dir / "artifacts" / "feedback_profile.json").exists()


def test_run_exports_feedback_template(tmp_path: Path) -> None:
    from agent_paper_reviewers.orchestrator import ReviewOrchestrator

    paper = tmp_path / "paper.md"
    paper.write_text(
        "# My Paper\n\n## Abstract\nWe propose X.\n\n## Experiments\nWe compare baselines.\n",
        encoding="utf-8",
    )
    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": ["We outperform baselines."],
        "options": {
            "language_mode": "en",
            "executor_backend": "local_vllm",
            "always_export_pdf": False,
        },
    }
    review_input = ReviewRunInput.model_validate(payload)
    repo_root = Path(__file__).resolve().parents[1]
    summary = ReviewOrchestrator(repo_root).run(review_input, tmp_path / "out")

    run_dir = Path(summary.output_dir)
    template_path = run_dir / "feedback_template.json"
    assert template_path.exists()
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert template["run_id"] == summary.run_id
    assert template["venue"] == "ICLR"
    assert isinstance(template["items"], list)


def test_feedback_loop_can_raise_confidence_for_correct_risk(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}
    ctx.artifacts["gaps"] = {"gaps": []}
    ctx.artifacts["venue_profile"] = {"profile": {"weights": {}}}

    reason = "Core comparisons should include strong baselines."
    phrase = "Baseline comparisons are not strong enough for this venue."
    fingerprint = make_risk_fingerprint(reason, phrase)

    feedback_payload = {
        "schema_version": 1,
        "run_id": "historical-run-up",
        "paper_title": "paper",
        "venue": "ICLR",
        "year": 2026,
        "items": [
            {
                "risk_id": "RISK-007",
                "risk_fingerprint": fingerprint,
                "reason": reason,
                "likely_reject_phrase": phrase,
                "verdict": "correct",
                "confidence": 0.95,
                "comment": "This is a true weakness in my previous submission.",
            },
            {
                "risk_id": "RISK-007",
                "risk_fingerprint": fingerprint,
                "reason": reason,
                "likely_reject_phrase": phrase,
                "verdict": "correct",
                "confidence": 0.9,
                "comment": "Repeatedly confirmed by advisor/reviewer.",
            },
        ],
    }
    submit_feedback(tmp_path, feedback_payload)

    step = RiskRankerStep(
        _FakeExecutor(
            {
                "risks": [
                    {
                        "id": "RISK-007",
                        "severity": "P1",
                        "score": 0.55,
                        "reason": reason,
                        "evidence_refs": [],
                        "likely_reject_phrase": phrase,
                        "fix_hint": "Add stronger baselines.",
                    }
                ],
                "scores": {
                    "novelty": 6.0,
                    "soundness": 6.0,
                    "experiment": 6.0,
                    "clarity": 6.0,
                    "overall": 6.0,
                },
            }
        )
    )
    step.run(ctx)

    risk = ctx.artifacts["risk_ranking"]["risks"][0]
    adjustment = risk.get("feedback_adjustment", {})
    assert float(risk["score"]) > 0.55
    assert adjustment.get("action") == "up"
    assert float(adjustment.get("confidence_after", 0.0)) > float(adjustment.get("confidence_before", 0.0))
    assert float(adjustment.get("calibration_confidence", 0.0)) > 0.0


def test_feedback_prompt_evolution_is_injected_into_risk_ranker(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}
    ctx.artifacts["gaps"] = {"gaps": []}
    ctx.artifacts["venue_profile"] = {"profile": {"weights": {}}}

    reason = "Statistical significance evidence appears missing."
    phrase = "Improvements are not statistically validated."
    fingerprint = make_risk_fingerprint(reason, phrase)
    submit_feedback(
        tmp_path,
        {
            "schema_version": 1,
            "run_id": "historical-run-prompt",
            "paper_title": "paper",
            "venue": "ICLR",
            "year": 2026,
            "items": [
                {
                    "risk_id": "RISK-001",
                    "risk_fingerprint": fingerprint,
                    "reason": reason,
                    "likely_reject_phrase": phrase,
                    "verdict": "incorrect",
                    "confidence": 0.95,
                    "comment": "False positive in this lab setup.",
                },
                {
                    "risk_id": "RISK-001",
                    "risk_fingerprint": fingerprint,
                    "reason": reason,
                    "likely_reject_phrase": phrase,
                    "verdict": "incorrect",
                    "confidence": 0.92,
                    "comment": "Repeated mismatch.",
                },
            ],
        },
    )

    executor = _CaptureExecutor(
        {
            "risks": [
                {
                    "id": "RISK-001",
                    "severity": "P1",
                    "score": 0.62,
                    "reason": reason,
                    "evidence_refs": [],
                    "likely_reject_phrase": phrase,
                    "fix_hint": "Add significance tests.",
                }
            ],
            "scores": {
                "novelty": 6.0,
                "soundness": 6.0,
                "experiment": 6.0,
                "clarity": 6.0,
                "overall": 6.0,
            },
        }
    )
    RiskRankerStep(executor).run(ctx)

    assert executor.last_spec is not None
    fp = executor.last_spec.context.get("feedback_prior", {})
    assert isinstance(fp, dict)
    assert fp.get("applied") is True
    assert int(fp.get("records_loaded", 0) or 0) >= 1
    assert isinstance(fp.get("high_confidence_incorrect_patterns", []), list)
    assert fp.get("high_confidence_incorrect_patterns")

    prompt_evolution_path = ctx.run_dir / "artifacts" / "prompt_evolution.risk_ranker.json"
    assert prompt_evolution_path.exists()
