from __future__ import annotations

import re
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .executors.factory import get_executor
from .mcp.factory import get_mcp_provider
from .models import ReviewRunInput, RunStatus, RunSummary
from .pipeline import (
    ClaimEvidenceAlignerStep,
    ClaimNormalizerStep,
    EvidenceIndexerStep,
    ExporterAndQAGateStep,
    GapDetectorStep,
    IntakeStep,
    PaperParserStep,
    RebuttalComposerStep,
    RemediationPlannerStep,
    ReportBuilderStep,
    RiskRankerStep,
    VenueProfileResolverStep,
)
from .pipeline.base import PipelineContext, PipelineStep
from .services.skill_flow_loader import load_skill_flow
from .services.translator import Translator


class ReviewOrchestrator:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def run(self, input_data: ReviewRunInput, output_root: Path) -> RunSummary:
        run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        run_dir = self._build_output_dir(output_root, input_data)

        executor = get_executor(input_data.options.executor_backend)
        mcp_tools = get_mcp_provider(input_data.options.mcp_backend)
        translator = Translator(executor)
        flow_profile = load_skill_flow(self.repo_root)

        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        ctx = PipelineContext(
            run_id=run_id,
            run_dir=run_dir,
            input_data=input_data,
            mcp_tools=mcp_tools,
        )

        if flow_profile.warnings:
            ctx.qa_issues.extend(flow_profile.warnings)

        skill_flow_payload = {
            "source": flow_profile.source,
            "steps": flow_profile.steps,
            "warnings": flow_profile.warnings,
            "mcp_capabilities": flow_profile.mcp_capabilities,
        }
        mcp_runtime_payload = {
            "backend": input_data.options.mcp_backend.value,
            "provider": mcp_tools.name,
            "capabilities": mcp_tools.capabilities(),
        }
        ctx.artifacts["skill_flow"] = skill_flow_payload
        ctx.artifacts["mcp_runtime"] = mcp_runtime_payload
        ctx.dump_json("artifacts/skill_flow_used.json", skill_flow_payload)
        ctx.dump_json("artifacts/mcp_runtime.json", mcp_runtime_payload)

        step_registry = self._build_step_registry(translator)
        steps = self._build_step_sequence(flow_profile.steps, step_registry)

        for step in steps:
            try:
                step.run(ctx)
            except Exception as exc:  # noqa: BLE001
                ctx.status = RunStatus.FAILED
                err_msg = f"step_failed:{step.name}:{exc}"
                ctx.qa_issues.append(err_msg)
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "pipeline_exception.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                break

        summary = RunSummary(
            run_id=run_id,
            status=ctx.status,
            output_dir=str(run_dir),
            qa_issues=ctx.qa_issues,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_result.json").write_text(
            summary.model_dump_json(indent=2), encoding="utf-8"
        )
        return summary

    def _build_step_registry(self, translator: Translator) -> dict[str, PipelineStep]:
        return {
            "Intake": IntakeStep(),
            "VenueProfileResolver": VenueProfileResolverStep(self.repo_root),
            "PaperParser": PaperParserStep(),
            "ClaimNormalizer": ClaimNormalizerStep(),
            "EvidenceIndexer": EvidenceIndexerStep(),
            "ClaimEvidenceAligner": ClaimEvidenceAlignerStep(),
            "GapDetector": GapDetectorStep(),
            "RiskRanker": RiskRankerStep(),
            "RemediationPlanner": RemediationPlannerStep(),
            "RebuttalComposer": RebuttalComposerStep(translator),
            "ReportBuilder": ReportBuilderStep(translator),
            "ExporterAndQAGate": ExporterAndQAGateStep(),
        }

    @staticmethod
    def _build_step_sequence(
        ordered_names: list[str],
        step_registry: dict[str, PipelineStep],
    ) -> list[PipelineStep]:
        steps: list[PipelineStep] = []
        for name in ordered_names:
            step = step_registry.get(name)
            if not step:
                raise ValueError(f"unknown_step_in_skill_flow:{name}")
            steps.append(step)
        return steps

    @staticmethod
    def _build_output_dir(output_root: Path, input_data: ReviewRunInput) -> Path:
        paper_name = Path(input_data.paper.path).stem
        paper_name = re.sub(r'[<>:"/\\|?*]+', " ", paper_name).strip()
        paper_name = re.sub(r"\s+", " ", paper_name)
        if not paper_name:
            paper_name = "paper"
        return output_root / paper_name
