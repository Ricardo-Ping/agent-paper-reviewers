from __future__ import annotations

import re
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .executors.factory import get_executor
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
from .pipeline.base import PipelineContext
from .services.translator import Translator


class ReviewOrchestrator:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def run(self, input_data: ReviewRunInput, output_root: Path) -> RunSummary:
        run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        run_dir = self._build_output_dir(output_root, input_data)
        executor = get_executor(input_data.options.executor_backend)
        translator = Translator(executor)

        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        ctx = PipelineContext(
            run_id=run_id,
            run_dir=run_dir,
            input_data=input_data,
        )

        steps = [
            IntakeStep(),
            VenueProfileResolverStep(self.repo_root),
            PaperParserStep(),
            ClaimNormalizerStep(),
            EvidenceIndexerStep(),
            ClaimEvidenceAlignerStep(),
            GapDetectorStep(),
            RiskRankerStep(),
            RemediationPlannerStep(),
            RebuttalComposerStep(translator),
            ReportBuilderStep(translator),
            ExporterAndQAGateStep(),
        ]

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

    @staticmethod
    def _build_output_dir(output_root: Path, input_data: ReviewRunInput) -> Path:
        paper_name = Path(input_data.paper.path).stem
        paper_name = re.sub(r'[<>:"/\\|?*]+', " ", paper_name).strip()
        paper_name = re.sub(r"\s+", " ", paper_name)
        if not paper_name:
            paper_name = "paper"
        return output_root / paper_name
