from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput
from agent_paper_reviewers.models import TaskResult, TaskSpec
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_reviewer_questions import ReviewerQuestionSimulatorStep


class _FencedResponseExecutor:
    def execute(self, spec: TaskSpec) -> TaskResult:
        if spec.task_type == "reviewer_question_simulation":
            return TaskResult(
                ok=True,
                output={
                    "response": """```json
{
  "questions": [
    {
      "priority": "high",
      "reviewer_persona": "empirical",
      "question": "Are the gains statistically significant across seeds?",
      "why_this_will_be_asked": "Single-run gains are unstable.",
      "trigger_gap_codes": ["missing_significance"],
      "linked_risk_ids": ["RISK-001"],
      "evidence_to_prepare": ["Multi-seed mean/std and paired tests."],
      "suggested_response_strategy": "Report seed-level significance."
    }
  ]
}
```"""
                },
            )
        return TaskResult(ok=True, output={})


def _ctx(tmp_path: Path) -> PipelineContext:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Method\nM\n", encoding="utf-8")
    data = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "SIGMOD", "year": 2026},
            "claims": ["We build a system for SQL dialect translation."],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return PipelineContext(run_id="r1", run_dir=run_dir, input_data=data)


def test_reviewer_question_simulator_adds_db_followups(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "title": "Cracking SQL Barriers",
        "raw_text": (
            "We propose a system-level SQL dialect translation pipeline. "
            "The system targets cross-dialect query translation and execution."
        ),
    }
    ctx.artifacts["gaps"] = {
        "gaps": [
            {"code": "missing_baseline", "severity_hint": "P1", "description": "Need stronger baselines."},
            {"code": "weak_claim_alignment", "severity_hint": "P1", "description": "Claims weakly supported."},
        ]
    }
    ctx.artifacts["risk_ranking"] = {
        "risks": [
            {
                "id": "RISK-001",
                "severity": "P1",
                "score": 0.66,
                "reason": "Baseline comparisons are not strong or fair enough for this venue.",
                "likely_reject_phrase": "Baseline weak.",
                "fix_hint": "Add baseline.",
            }
        ]
    }
    ctx.artifacts["claim_evidence_matrix"] = {
        "alignments": [
            {
                "claim_id": "C1",
                "strength": "Weak",
                "score": 0.41,
                "claim_text": "System outperforms prior methods.",
                "evidence_refs": [],
            }
        ]
    }

    ReviewerQuestionSimulatorStep().run(ctx)
    payload = ctx.artifacts["reviewer_questions"]
    questions = payload.get("questions", [])

    assert payload.get("source") == "rule_fallback"
    assert questions
    assert any("SQLGlot" in str(q.get("question", "")) for q in questions)
    assert any("LLM" in str(q.get("question", "")) for q in questions)
    assert any("missing_baseline" in q.get("trigger_gap_codes", []) for q in questions if isinstance(q, dict))


def test_reviewer_question_simulator_parses_fenced_json_executor_output(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "title": "Test",
        "raw_text": "We report gains.",
    }
    ctx.artifacts["gaps"] = {
        "gaps": [{"code": "missing_significance", "severity_hint": "P1", "description": "Need stats."}]
    }
    ctx.artifacts["risk_ranking"] = {
        "risks": [{"id": "RISK-001", "severity": "P1", "score": 0.62, "reason": "Stats unclear."}]
    }
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}
    ctx.artifacts["venue_profile"] = {"profile": {"common_reject_reasons": []}}

    ReviewerQuestionSimulatorStep(_FencedResponseExecutor()).run(ctx)
    payload = ctx.artifacts["reviewer_questions"]
    assert payload["source"] == "executor"
    assert len(payload.get("questions", [])) == 1
    assert "significant" in payload["questions"][0]["question"].lower()
