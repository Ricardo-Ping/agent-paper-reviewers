from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, TaskResult, TaskSpec
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_rebuttal import RebuttalComposerStep
from agent_paper_reviewers.services.translator import Translator


class _AlwaysFailExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, spec: TaskSpec) -> TaskResult:
        self.calls += 1
        raise RuntimeError("temporary network failure")


class _SecondTryExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, spec: TaskSpec) -> TaskResult:
        self.calls += 1
        if self.calls == 1:
            return TaskResult(ok=False, output={}, warnings=["503"])
        return TaskResult(ok=True, output={"response": "ok"}, warnings=[])


def _ctx(tmp_path: Path) -> PipelineContext:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Method\ntext\n", encoding="utf-8")
    data = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "ICLR", "year": 2026},
            "claims": ["c1"],
            "options": {"language_mode": "en", "executor_backend": "openai", "always_export_pdf": False},
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return PipelineContext(run_id="r", run_dir=run_dir, input_data=data)


def test_rebuttal_retry_guard_has_hard_cap_and_warning(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    executor = _AlwaysFailExecutor()
    step = RebuttalComposerStep(Translator(executor), executor)

    spec = TaskSpec(
        task_type="rebuttal_compose",
        prompt="x",
        context={},
        output_schema={},
        model_profile="judge",
    )
    result = step._execute_with_retry(ctx, spec, call_tag="unit")

    assert result is None
    assert executor.calls == step._MAX_EXECUTE_RETRIES
    assert any("retry_exhausted_error" in issue for issue in ctx.qa_issues)


def test_rebuttal_retry_guard_recovers_on_second_attempt(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    executor = _SecondTryExecutor()
    step = RebuttalComposerStep(Translator(executor), executor)

    spec = TaskSpec(
        task_type="rebuttal_global",
        prompt="x",
        context={},
        output_schema={},
        model_profile="judge",
    )
    result = step._execute_with_retry(ctx, spec, call_tag="unit")

    assert result is not None and result.ok is True
    assert executor.calls == 2
    assert any("attempt_1_not_ok" in issue for issue in ctx.qa_issues)

