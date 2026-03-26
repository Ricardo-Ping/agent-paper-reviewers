from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput, TaskResult, TaskSpec
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_rebuttal import RebuttalComposerStep
from agent_paper_reviewers.services.translator import Translator


class _WeakRebuttalExecutor:
    def execute(self, spec: TaskSpec) -> TaskResult:
        if spec.task_type == "rebuttal_compose":
            return TaskResult(
                ok=True,
                output={
                    "response": "Thanks. We will improve the paper.",
                    "new_evidence": ["More experiments."],
                    "paper_change": "Update paper.",
                },
            )
        if spec.task_type == "rebuttal_precheck":
            return TaskResult(
                ok=True,
                output={
                    "pass": False,
                    "issues": ["response too generic"],
                    "revised_response": "Thanks. We will add significance tests and multi-seed statistics to address this concern.",
                    "revised_new_evidence": ["Add paired significance tests.", "Report multi-seed mean/std."],
                    "revised_paper_change": "Update Experiments and Analysis sections.",
                },
            )
        if spec.task_type == "rebuttal_global":
            return TaskResult(ok=True, output={"global_response": "Global response."})
        return TaskResult(ok=True, output={})


class _HallucinatedReferenceExecutor:
    def execute(self, spec: TaskSpec) -> TaskResult:
        if spec.task_type == "rebuttal_compose":
            return TaskResult(
                ok=True,
                output={
                    "response": "We will reconcile Figure 99 and Table 88 in Experiments and Ablation.",
                    "new_evidence": [
                        "Add analysis around Figure 99 and Table 88 for stronger support."
                    ],
                    "paper_change": "Update Experiments, Ablation, and Limitations sections with Figure 99/Table 88 evidence.",
                },
            )
        if spec.task_type == "rebuttal_precheck":
            return TaskResult(ok=False, output={})
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


def test_rebuttal_precheck_repairs_generic_response(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
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
                "reason": "Statistical significance evidence appears missing.",
                "severity": "P1",
                "score": 0.62,
                "likely_reject_phrase": "Stats unclear.",
                "fix_hint": "Add significance analysis.",
            }
        ]
    }
    ctx.artifacts["remediation_plan"] = {"tasks": []}

    executor = _WeakRebuttalExecutor()
    step = RebuttalComposerStep(Translator(executor), executor)
    step.run(ctx)

    item = ctx.artifacts["rebuttal"]["en"]["bundle"]["items"][0]
    assert "significance tests" in item["response"].lower()
    precheck = ctx.artifacts["rebuttal_precheck"]["items"][0]
    assert precheck["repair_applied"] is True
    plan = ctx.artifacts.get("rebuttal_plan", {})
    assert isinstance(plan.get("plan_items", []), list)
    assert plan["plan_items"][0]["review_id"] == "R1"
    assert isinstance(plan.get("post_generation_audit", []), list)
    assert isinstance(plan.get("summary", {}), dict)


def test_rebuttal_precheck_detects_hallucinated_sections_and_anchors(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.artifacts["paper_structured"] = {
        "title": "Paper",
        "raw_text": "## Method\nDetails only.",
        "sections": [
            {"name": "method", "text": "Details only.", "section_id": "S001", "section_index": 1},
            {"name": "experiments", "text": "", "section_id": "S002", "section_index": 2},
        ],
    }
    ctx.artifacts["evidence_index"] = {
        "passages": [
            {"id": "S001_para0", "section": "method", "text": "Method details only.", "kind": "paragraph"},
        ]
    }
    ctx.artifacts["venue_profile"] = {
        "profile": {
            "rebuttal_policy": {
                "mode": "per_review_only",
                "per_review_char_limit": 1400,
                "global_char_limit": 0,
            }
        }
    }
    ctx.artifacts["risk_ranking"] = {
        "risks": [
            {
                "id": "RISK-001",
                "reason": "Claim-result contradiction in experiments.",
                "severity": "P1",
                "score": 0.73,
                "likely_reject_phrase": "Concern.",
                "fix_hint": "Fix.",
            }
        ]
    }
    ctx.artifacts["remediation_plan"] = {"tasks": []}

    executor = _HallucinatedReferenceExecutor()
    step = RebuttalComposerStep(Translator(executor), executor)
    step.run(ctx)

    item = ctx.artifacts["rebuttal"]["en"]["bundle"]["items"][0]
    assert "Figure 99" not in item["response"]
    assert "Table 88" not in item["paper_change"]

    precheck = ctx.artifacts["rebuttal_precheck"]["items"][0]
    issues = precheck.get("issues", [])
    assert any("hallucinated_anchor" in x for x in issues) or any("unverifiable_anchor" in x for x in issues)
    assert any("section_without_content:experiments" in x for x in issues)
    assert precheck.get("repair_applied") is True
    assert any("rebuttal_hallucination_warning:R1" in x for x in ctx.qa_issues)
