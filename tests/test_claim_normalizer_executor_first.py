from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, TaskResult, TaskSpec
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_claim_normalizer import ClaimNormalizerStep


class _FakeExecutor:
    def __init__(self, output: dict, ok: bool = True) -> None:
        self.output = output
        self.ok = ok

    def execute(self, spec: TaskSpec) -> TaskResult:
        return TaskResult(ok=self.ok, output=self.output)


def _ctx(tmp_path: Path) -> PipelineContext:
    paper = tmp_path / "paper.md"
    paper.write_text("# T\n\n## Abstract\nWe improve results.\n", encoding="utf-8")
    data = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "ICML", "year": 2026},
            "claims": ["Our method is better than baseline."],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(run_id="r1", run_dir=run_dir, input_data=data)
    ctx.artifacts["paper_structured"] = {"raw_text": "abstract we improve results."}
    return ctx


def test_claim_normalizer_uses_executor_when_valid(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    step = ClaimNormalizerStep(
        _FakeExecutor(
            {
                "claim_id": "C1",
                "text": "Our method is better than baseline.",
                "type": "baseline",
                "verifiable_claim": "Outperforms strong baselines under matched settings.",
                "success_criteria": "Improvement over all strong baselines.",
                "weakness_hint": "Baseline mismatch risk.",
            }
        )
    )
    step.run(ctx)
    first = ctx.artifacts["claims_normalized"]["claims"][0]
    assert first["claim_type"] == "baseline"
    assert first["verifiable_claim"].startswith("Outperforms")


def test_claim_normalizer_falls_back_when_executor_invalid(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    step = ClaimNormalizerStep(_FakeExecutor({"note": "bad"}))
    step.run(ctx)
    first = ctx.artifacts["claims_normalized"]["claims"][0]
    assert first["claim_id"] == "C1"
    assert "verifiable_claim" in first
    assert "success_criteria" in first

