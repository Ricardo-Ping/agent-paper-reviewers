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


def _ctx(
    tmp_path: Path,
    *,
    stage: str = "initial_submission",
    reviewer_comments: list[dict] | None = None,
) -> PipelineContext:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Method\nM\n", encoding="utf-8")
    reviewer_comments = reviewer_comments or []
    data = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "SIGMOD", "year": 2026},
            "claims": ["We build a system for SQL dialect translation."],
            "review_context": {
                "manuscript_stage": stage,
                "reviewer_comments": reviewer_comments,
            },
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
                "evidence_refs": [
                    {
                        "section": "experiments",
                        "passage_id": "S004_para0",
                        "excerpt": "We compare against two baselines.",
                        "page": 4,
                    }
                ],
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
    assert any("[see:" in str(q.get("evidence_anchor_hint", "")) for q in questions if isinstance(q, dict))
    assert any(isinstance(q.get("evidence_anchor_refs", []), list) and q.get("evidence_anchor_refs", []) for q in questions if isinstance(q, dict))


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
        "risks": [
            {
                "id": "RISK-001",
                "severity": "P1",
                "score": 0.62,
                "reason": "Stats unclear.",
                "evidence_refs": [
                    {
                        "section": "results",
                        "passage_id": "S008_para1",
                        "excerpt": "Single-run results only.",
                    }
                ],
            }
        ]
    }
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}
    ctx.artifacts["venue_profile"] = {"profile": {"common_reject_reasons": []}}

    ReviewerQuestionSimulatorStep(_FencedResponseExecutor()).run(ctx)
    payload = ctx.artifacts["reviewer_questions"]
    assert payload["source"] in {"executor", "executor_plus_rule_enrichment"}
    assert len(payload.get("questions", [])) >= 1
    assert any("significant" in str(x.get("question", "")).lower() for x in payload.get("questions", []))
    assert any(str(x.get("priority", "")).lower() == "medium" for x in payload.get("questions", []))
    assert any("[see:" in str(x.get("evidence_anchor_hint", "")) for x in payload.get("questions", []))


def test_reviewer_question_simulator_initial_stage_has_priority_diversity(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, stage="initial_submission")
    ctx.artifacts["paper_structured"] = {
        "title": "Pre-submission SQL Draft",
        "raw_text": "We report gains and discuss limits.",
    }
    ctx.artifacts["gaps"] = {
        "gaps": [{"code": "missing_significance", "severity_hint": "P1", "description": "Need stats."}]
    }
    ctx.artifacts["risk_ranking"] = {
        "risks": [{"id": "RISK-001", "severity": "P1", "score": 0.62, "reason": "Stats unclear."}]
    }
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}

    ReviewerQuestionSimulatorStep().run(ctx)
    payload = ctx.artifacts["reviewer_questions"]
    questions = payload.get("questions", [])
    priorities = {str(q.get("priority", "")).lower() for q in questions if isinstance(q, dict)}
    qtypes = {str(q.get("question_type", "")).lower() for q in questions if isinstance(q, dict)}

    assert payload["manuscript_stage"] == "initial_submission"
    assert payload["source"] == "rule_fallback"
    assert "high" in priorities
    assert "medium" in priorities
    assert "novelty_boundary" in qtypes


def test_reviewer_question_simulator_includes_role_based_personas(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, stage="initial_submission")
    ctx.artifacts["paper_structured"] = {
        "title": "Persona Coverage Draft",
        "raw_text": "We propose a novel method and provide preliminary experiments.",
    }
    ctx.artifacts["gaps"] = {
        "gaps": [
            {"code": "weak_claim_alignment", "severity_hint": "P1", "description": "Need direct evidence."},
            {"code": "missing_significance", "severity_hint": "P1", "description": "Need stronger statistics."},
        ]
    }
    ctx.artifacts["risk_ranking"] = {
        "risks": [
            {"id": "RISK-001", "severity": "P1", "score": 0.70, "reason": "Weak claim alignment."},
            {"id": "RISK-002", "severity": "P1", "score": 0.66, "reason": "Missing significance."},
        ]
    }
    ctx.artifacts["claim_evidence_matrix"] = {
        "alignments": [
            {"claim_id": "C1", "strength": "Weak", "score": 0.41, "claim_text": "Outperforms all baselines."}
        ]
    }

    ReviewerQuestionSimulatorStep().run(ctx)
    payload = ctx.artifacts["reviewer_questions"]
    questions = payload.get("questions", [])

    personas = {str(q.get("reviewer_persona", "")).strip() for q in questions if isinstance(q, dict)}
    qtypes = {str(q.get("question_type", "")).strip() for q in questions if isinstance(q, dict)}

    assert "methodology_reviewer" in personas
    assert "empirical_reviewer" in personas
    assert "theory_reviewer" in personas
    assert "methodology_rigor" in qtypes
    assert "empirical_validation" in qtypes
    assert "theory_soundness" in qtypes


def test_reviewer_question_simulator_meta_stage_targets_reviewer_comments(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        stage="meta_review_discussion",
        reviewer_comments=[
            {"review_id": "R1", "concern": "Statistical significance is weak."},
            {"review_id": "R2", "concern": "Need fairer baseline comparisons."},
        ],
    )
    ctx.artifacts["paper_structured"] = {
        "title": "Discussion-stage SQL Draft",
        "raw_text": "We report gains with SQL experiments.",
    }
    ctx.artifacts["gaps"] = {
        "gaps": [{"code": "weak_claim_alignment", "severity_hint": "P1", "description": "Need direct evidence."}]
    }
    ctx.artifacts["risk_ranking"] = {
        "risks": [
            {"id": "RISK-001", "severity": "P1", "score": 0.61, "reason": "Statistical significance is missing."},
            {"id": "RISK-002", "severity": "P1", "score": 0.59, "reason": "Baseline fairness is unclear."},
        ]
    }
    ctx.artifacts["claim_evidence_matrix"] = {"alignments": []}

    ReviewerQuestionSimulatorStep().run(ctx)
    payload = ctx.artifacts["reviewer_questions"]
    questions = payload.get("questions", [])
    highs = [q for q in questions if str(q.get("priority", "")).lower() == "high"]

    assert payload["manuscript_stage"] == "meta_review_discussion"
    assert any("Reviewer R1" in str(q.get("question", "")) for q in questions)
    assert any("Reviewer R2" in str(q.get("question", "")) for q in questions)
    assert any(str(q.get("question_type", "")).lower() == "stage_followup" for q in questions)
    assert len(highs) >= 2
