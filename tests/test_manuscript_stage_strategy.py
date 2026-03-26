from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, TaskResult, TaskSpec
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_rebuttal import RebuttalComposerStep
from agent_paper_reviewers.pipeline.step_report_builder import ReportBuilderStep
from agent_paper_reviewers.pipeline.step_risk_ranker import RiskRankerStep
from agent_paper_reviewers.services.translator import Translator


class _FakeExecutor:
    def execute(self, spec: TaskSpec) -> TaskResult:
        if spec.task_type == "rebuttal_compose":
            concern = str(spec.context.get("concern", ""))
            return TaskResult(
                ok=True,
                output={
                    "response": f"Direct response for: {concern}",
                    "new_evidence": ["Evidence 1", "Evidence 2"],
                    "paper_change": "Update rebuttal-mapped sections.",
                },
            )
        return TaskResult(ok=True, output={})


def _ctx(tmp_path: Path, *, stage: str, comments: list[dict]) -> PipelineContext:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Method\nM\n", encoding="utf-8")
    payload = {
        "paper": {"format": "md", "path": str(paper)},
        "venue": {"name": "ICLR", "year": 2026},
        "claims": ["c1"],
        "review_context": {
            "manuscript_stage": stage,
            "reviewer_comments": comments,
        },
        "options": {"language_mode": "en", "executor_backend": "local_vllm", "always_export_pdf": False},
    }
    data = ReviewRunInput.model_validate(payload)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return PipelineContext(run_id="r1", run_dir=run_dir, input_data=data)


def test_report_builder_has_stage_specific_decision_labels() -> None:
    decision, _, _ = ReportBuilderStep._decision(
        risks=[{"severity": "P1"}, {"severity": "P1"}],
        scores={"overall": 6.1},
        venue_name="ICLR",
        venue_profile={},
        manuscript_stage="meta_review_discussion",
    )
    assert decision in {"Weak Discussion Position", "Recoverable in Discussion", "Strong Discussion Position"}

    decision2, _, _ = ReportBuilderStep._decision(
        risks=[{"severity": "P1"}, {"severity": "P1"}],
        scores={"overall": 6.1},
        venue_name="ICLR",
        venue_profile={},
        manuscript_stage="rejected_after_reviews",
    )
    assert decision2 in {"Major Revision Required", "Resubmission Candidate", "Ready for Resubmission"}


def test_risk_ranker_stage_strategy_prioritizes_reviewer_comments(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        stage="meta_review_discussion",
        comments=[
            {"review_id": "R1", "concern": "Statistical significance is not convincing."},
            {"review_id": "R2", "concern": "Need stronger baseline comparisons."},
        ],
    )
    step = RiskRankerStep()
    payload = {
        "risks": [
            {
                "id": "RISK-001",
                "severity": "P1",
                "score": 0.56,
                "reason": "Baseline comparison is weak.",
                "evidence_refs": [],
                "likely_reject_phrase": "Baseline not enough.",
                "fix_hint": "Add stronger baselines.",
            }
        ],
        "scores": {"novelty": 6.0, "soundness": 6.0, "experiment": 6.0, "clarity": 6.0, "overall": 6.0},
    }
    out = step._apply_stage_strategy(  # noqa: SLF001
        ctx,
        payload,
        venue_profile={"weights": {"novelty": 0.25, "soundness": 0.3, "experiment": 0.3, "clarity": 0.15}},
    )
    assert out["stage_strategy"]["used_for_ranking"] is True
    assert out["focus_risks"]
    assert any(str(r.get("id", "")).startswith("RISK-RV-") for r in out["risks"])


def test_rebuttal_meta_stage_uses_reviewer_concerns_as_targets(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        stage="meta_review_discussion",
        comments=[
            {"review_id": "R1", "concern": "Please justify statistical significance."},
            {"review_id": "R2", "concern": "Why these baselines?"},
        ],
    )
    ctx.artifacts["venue_profile"] = {
        "profile": {
            "rebuttal_policy": {
                "mode": "per_review_only",
                "per_review_char_limit": 1200,
                "global_char_limit": 0,
            }
        }
    }
    ctx.artifacts["risk_ranking"] = {
        "risks": [
            {
                "id": "RISK-001",
                "reason": "Baseline comparison is weak.",
                "severity": "P1",
                "score": 0.61,
                "likely_reject_phrase": "Not enough evidence.",
                "fix_hint": "Add stronger baseline.",
            },
            {
                "id": "RISK-002",
                "reason": "Statistical significance is missing.",
                "severity": "P1",
                "score": 0.58,
                "likely_reject_phrase": "Stats unclear.",
                "fix_hint": "Add significance analysis.",
            },
        ],
        "focus_risks": [
            {
                "id": "RISK-002",
                "reason": "Statistical significance is missing.",
                "severity": "P1",
                "score": 0.58,
                "likely_reject_phrase": "Stats unclear.",
                "fix_hint": "Add significance analysis.",
            },
            {
                "id": "RISK-001",
                "reason": "Baseline comparison is weak.",
                "severity": "P1",
                "score": 0.61,
                "likely_reject_phrase": "Not enough evidence.",
                "fix_hint": "Add stronger baseline.",
            },
        ],
    }
    ctx.artifacts["remediation_plan"] = {"tasks": []}

    executor = _FakeExecutor()
    step = RebuttalComposerStep(Translator(executor), executor)
    step.run(ctx)

    bundle = ctx.artifacts["rebuttal"]["en"]["bundle"]
    assert bundle["manuscript_stage"] == "meta_review_discussion"
    assert bundle["items"][0]["review_id"] == "R1"
    assert bundle["items"][0]["concern"] == "Please justify statistical significance."
    assert bundle["items"][1]["review_id"] == "R2"
    assert bundle["items"][1]["concern"] == "Why these baselines?"

