from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, TaskResult, TaskSpec
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_remediation import RemediationPlannerStep
from agent_paper_reviewers.pipeline.step_risk_ranker import RiskRankerStep


class _FakeExecutor:
    def __init__(self, output: dict, ok: bool = True) -> None:
        self.output = output
        self.ok = ok

    def execute(self, spec: TaskSpec) -> TaskResult:
        return TaskResult(ok=self.ok, output=self.output)


def _ctx(tmp_path: Path) -> PipelineContext:
    paper = tmp_path / "paper.md"
    paper.write_text("# T\n\n## Method\nM\n\n## Experiments\nE\n", encoding="utf-8")
    data = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "ICML", "year": 2026},
            "claims": ["c1"],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return PipelineContext(run_id="r1", run_dir=run_dir, input_data=data)


def test_risk_ranker_prefers_executor_output(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["claim_evidence_matrix"] = {
        "alignments": [
            {
                "claim_id": "C1",
                "claim_text": "c1",
                "strength": "Weak",
                "score": 0.4,
                "evidence_refs": [],
            }
        ]
    }
    ctx.artifacts["gaps"] = {"gaps": []}

    step = RiskRankerStep(
        _FakeExecutor(
            {
                "risks": [
                    {
                        "id": "RISK-X",
                        "severity": "P1",
                        "score": 0.61,
                        "reason": "custom",
                        "evidence_refs": [],
                        "likely_reject_phrase": "phrase",
                        "fix_hint": "hint",
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

    assert ctx.artifacts["risk_ranking"]["risks"][0]["id"] == "RISK-X"
    assert ctx.artifacts["risk_ranking"]["scores"]["overall"] == 6.0


def test_remediation_planner_falls_back_on_invalid_executor_output(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["risk_ranking"] = {
        "risks": [
            {
                "id": "RISK-001",
                "severity": "P1",
                "score": 0.6,
                "reason": "r",
                "evidence_refs": [],
                "likely_reject_phrase": "p",
                "fix_hint": "f",
            }
        ]
    }

    step = RemediationPlannerStep(_FakeExecutor({"note": "invalid"}))
    step.run(ctx)

    tasks = ctx.artifacts["remediation_plan"]["tasks"]
    assert tasks
    assert tasks[0]["id"] == "EXP-001"
    assert tasks[0]["risk_id"] == "RISK-001"


def test_remediation_planner_enforces_constraints(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.input_data.constraints.gpu_budget_hours = 15
    ctx.input_data.constraints.time_days = 3
    ctx.input_data.constraints.max_new_experiments = 2
    ctx.artifacts["risk_ranking"] = {
        "risks": [
            {
                "id": "RISK-001",
                "severity": "P1",
                "score": 0.8,
                "reason": "r1",
                "evidence_refs": [],
                "likely_reject_phrase": "p1",
                "fix_hint": "f1",
            },
            {
                "id": "RISK-002",
                "severity": "P1",
                "score": 0.7,
                "reason": "r2",
                "evidence_refs": [],
                "likely_reject_phrase": "p2",
                "fix_hint": "f2",
            },
        ]
    }
    ctx.artifacts["gaps"] = {"gaps": []}

    executor_output = {
        "tasks": [
            {
                "id": "EXP-X",
                "risk_id": "RISK-001",
                "title": "too expensive",
                "priority": "high",
                "effort": "L",
                "est_time_days": 5,
                "est_gpu_hours": 40,
                "expected_gain": "x",
                "protocol": ["p"],
            },
            {
                "id": "EXP-Y",
                "risk_id": "RISK-001",
                "title": "affordable 1",
                "priority": "high",
                "effort": "M",
                "est_time_days": 2,
                "est_gpu_hours": 10,
                "expected_gain": "y",
                "protocol": ["p"],
            },
            {
                "id": "EXP-Z",
                "risk_id": "RISK-002",
                "title": "affordable 2",
                "priority": "medium",
                "effort": "S",
                "est_time_days": 1,
                "est_gpu_hours": 4,
                "expected_gain": "z",
                "protocol": ["p"],
            },
        ]
    }

    step = RemediationPlannerStep(_FakeExecutor(executor_output))
    step.run(ctx)
    payload = ctx.artifacts["remediation_plan"]
    tasks = payload["tasks"]

    assert tasks
    assert len(tasks) <= 2
    assert payload["total_est_gpu_hours"] <= 15
    assert payload["total_est_time_days"] <= 3
