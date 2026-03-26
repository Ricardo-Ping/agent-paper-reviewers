from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, TaskResult, TaskSpec
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_rebuttal import RebuttalComposerStep
from agent_paper_reviewers.services.translator import Translator


class _FakeExecutor:
    def execute(self, spec: TaskSpec) -> TaskResult:
        if spec.task_type == "rebuttal_compose":
            concern = str(spec.context.get("concern", ""))
            return TaskResult(
                ok=True,
                output={
                    "response": f"Executor response for: {concern}",
                    "new_evidence": ["Evidence A", "Evidence B"],
                    "paper_change": "Update Experiments section.",
                },
            )
        if spec.task_type == "rebuttal_global":
            return TaskResult(ok=True, output={"global_response": "Executor global response."})
        return TaskResult(ok=True, output={})


def _ctx(tmp_path: Path) -> PipelineContext:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Method\nM\n", encoding="utf-8")
    data = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "ICLR", "year": 2026},
            "claims": ["c1"],
            "options": {"language_mode": "en", "executor_backend": "local_vllm", "always_export_pdf": False},
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return PipelineContext(run_id="r1", run_dir=run_dir, input_data=data)


def test_rebuttal_composer_uses_executor_and_keeps_concern_mapping(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["venue_profile"] = {
        "profile": {
            "rebuttal_policy": {
                "mode": "global+per_review",
                "per_review_char_limit": 1200,
                "global_char_limit": 800,
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
        ]
    }
    ctx.artifacts["remediation_plan"] = {
        "tasks": [
            {"id": "EXP-001", "risk_id": "RISK-001", "title": "Strong baseline experiment"},
            {"id": "EXP-002", "risk_id": "RISK-002", "title": "Multi-seed significance run"},
        ]
    }

    executor = _FakeExecutor()
    step = RebuttalComposerStep(Translator(executor), executor)
    step.run(ctx)

    bundle = ctx.artifacts["rebuttal"]["en"]["bundle"]
    assert bundle["global_response"] == "Executor global response."
    assert bundle["items"][0]["concern"] == "Baseline comparison is weak."
    assert "Executor response for: Baseline comparison is weak." in bundle["items"][0]["response"]
    assert bundle["items"][1]["concern"] == "Statistical significance is missing."
    assert "Executor response for: Statistical significance is missing." in bundle["items"][1]["response"]
