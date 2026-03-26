from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, RunStatus, TaskResult, TaskSpec
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_paper_qa_gate import PaperQAGateStep
from agent_paper_reviewers.services.translator import Translator


class _PaperQAGateExecutor:
    def __init__(self) -> None:
        self.self_review_calls = 0

    def execute(self, spec: TaskSpec) -> TaskResult:
        if spec.task_type == "paper_qa_self_review":
            self.self_review_calls += 1
            if self.self_review_calls == 1:
                return TaskResult(
                    ok=True,
                    output={
                        "accept": False,
                        "issues": ["template_like_response"],
                        "per_item": [{"review_id": "R1", "verdict": "fail", "issues": ["template_like_response"]}],
                        "rewrites": [
                            {
                                "review_id": "R1",
                                "response": "We now add direct numeric evidence (Table 2, +3.1 BLEU, p<0.05).",
                                "new_evidence": ["Table 2 reports +3.1 BLEU with paired test p<0.05."],
                                "paper_change": "Update Experiments Section 5.2 and add statistical appendix.",
                            }
                        ],
                    },
                )
            return TaskResult(
                ok=True,
                output={
                    "accept": True,
                    "issues": [],
                    "per_item": [{"review_id": "R1", "verdict": "pass", "issues": []}],
                    "rewrites": [],
                },
            )
        if spec.task_type == "translate_zh":
            return TaskResult(ok=True, output={"translated_text": f"中:{spec.context.get('text', '')}"})
        return TaskResult(ok=True, output={})


def _ctx(tmp_path: Path, language_mode: str = "en") -> PipelineContext:
    paper = tmp_path / "paper.md"
    paper.write_text("# Title\n\n## Experiments\nResults\n", encoding="utf-8")
    data = ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "NeurIPS", "year": 2026},
            "claims": ["c1"],
            "options": {
                "language_mode": language_mode,
                "executor_backend": "agent_api",
                "always_export_pdf": False,
            },
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return PipelineContext(run_id="r1", run_dir=run_dir, input_data=data)


def test_paper_qa_gate_rewrites_then_passes(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, language_mode="en")
    ctx.artifacts["risk_ranking"] = {
        "risks": [{"id": "RISK-001", "severity": "P1", "score": 0.63, "reason": "Evidence is generic."}]
    }
    ctx.artifacts["rebuttal"] = {
        "en": {
            "bundle": {
                "venue": "NeurIPS",
                "year": 2026,
                "manuscript_stage": "initial_submission",
                "mode": "per_review_only",
                "items": [
                    {
                        "review_id": "R1",
                        "concern": "Evidence is too generic.",
                        "response": "Thank you. We will improve this.",
                        "new_evidence": ["More experiments."],
                        "paper_change": "Update paper.",
                        "char_count": 120,
                        "char_limit": 1000,
                    }
                ],
                "global_response": None,
                "attachment_pdf": None,
            },
            "markdown": "stub",
        }
    }

    executor = _PaperQAGateExecutor()
    step = PaperQAGateStep(Translator(executor), executor)
    step.run(ctx)

    gate = ctx.artifacts["paper_qa_gate"]
    assert gate["initial_accept"] is False
    assert gate["accepted"] is True
    assert gate["rewrites_applied"] == 1
    assert gate["post_recheck_accept"] is True
    assert ctx.status == RunStatus.SUCCESS

    item = ctx.artifacts["rebuttal"]["en"]["bundle"]["items"][0]
    assert "Table 2" in item["response"]
    assert "p<0.05" in item["response"]


def test_paper_qa_gate_updates_zh_bundle_after_rewrite(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, language_mode="en_zh")
    ctx.artifacts["risk_ranking"] = {
        "risks": [{"id": "RISK-001", "severity": "P1", "score": 0.63, "reason": "Evidence is generic."}]
    }
    ctx.artifacts["rebuttal"] = {
        "en": {
            "bundle": {
                "venue": "NeurIPS",
                "year": 2026,
                "manuscript_stage": "initial_submission",
                "mode": "per_review_only",
                "items": [
                    {
                        "review_id": "R1",
                        "concern": "Evidence is too generic.",
                        "response": "Thank you. We will improve this.",
                        "new_evidence": ["More experiments."],
                        "paper_change": "Update paper.",
                        "char_count": 120,
                        "char_limit": 1000,
                    }
                ],
                "global_response": None,
                "attachment_pdf": None,
            },
            "markdown": "stub",
        }
    }

    executor = _PaperQAGateExecutor()
    step = PaperQAGateStep(Translator(executor), executor)
    step.run(ctx)

    assert "zh" in ctx.artifacts["rebuttal"]
    zh_item = ctx.artifacts["rebuttal"]["zh"]["bundle"]["items"][0]
    assert "Table 2" in str(zh_item["response"])
