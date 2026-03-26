from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, TaskResult, TaskSpec
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_risk_ranker import RiskRankerStep


class _FallbackRiskExecutor:
    def execute(self, spec: TaskSpec) -> TaskResult:
        if spec.task_type != "risk_ranking":
            return TaskResult(ok=True, output={})
        return TaskResult(
            ok=True,
            output={
                "risks": [
                    {
                        "id": "RISK-001",
                        "severity": "P1",
                        "score": 0.61,
                        "reason": "placeholder",
                        "evidence_refs": [],
                        "likely_reject_phrase": "placeholder",
                        "fix_hint": "placeholder",
                    }
                ],
                "scores": {
                    "novelty": 5.0,
                    "soundness": 5.0,
                    "experiment": 5.0,
                    "clarity": 5.0,
                    "overall": 5.0,
                },
            },
            warnings=["executor_api_key_missing_use_fallback"],
        )


def _ctx(tmp_path: Path) -> PipelineContext:
    paper = tmp_path / "paper.md"
    paper.write_text("# T\n\ntext", encoding="utf-8")
    data = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "ICLR", "year": 2026},
            "claims": ["c1"],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(run_id="r1", run_dir=run_dir, input_data=data)
    ctx.artifacts["claim_evidence_matrix"] = {
        "alignments": [
            {
                "claim_id": "C1",
                "claim_text": "c1",
                "claim_type": "novelty",
                "strength": "Weak",
                "score": 0.45,
                "evidence_refs": [],
            }
        ]
    }
    ctx.artifacts["gaps"] = {
        "gaps": [
            {
                "code": "missing_significance",
                "severity_hint": "P1",
                "description": "missing stats",
                "evidence_refs": [],
            }
        ]
    }
    ctx.artifacts["venue_profile"] = {"profile": {"weights": {"novelty": 0.25, "soundness": 0.25, "experiment": 0.25, "clarity": 0.25}}}
    return ctx


def test_risk_ranker_uses_rule_fallback_when_executor_fallback_detected(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    step = RiskRankerStep(executor=_FallbackRiskExecutor())
    step.run(ctx)

    payload = ctx.artifacts["risk_ranking"]
    assert payload["source"] == "rule_fallback"
    assert any("risk_ranker_executor_fallback_detected_use_rule_fallback" in x for x in ctx.qa_issues)

