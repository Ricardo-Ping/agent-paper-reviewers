from __future__ import annotations

import json
import re
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .executors.base import ExecutorAdapter
from .executors.factory import get_executor, validate_executor_readiness
from .models import ReviewRunInput, RunStatus, RunSummary
from .pipeline import (
    ClaimEvidenceAlignerStep,
    ClaimDiscovererStep,
    ClaimNormalizerStep,
    CitationGraphStep,
    EvidenceIndexerStep,
    ExporterAndQAGateStep,
    GapDetectorStep,
    IntakeStep,
    PaperParserStep,
    PaperQAGateStep,
    RebuttalComposerStep,
    RemediationPlannerStep,
    ReportBuilderStep,
    ReviewerQuestionSimulatorStep,
    RiskRankerStep,
    VenueRecommenderStep,
    VenueProfileResolverStep,
)
from .pipeline.base import PipelineContext, PipelineStep
from .services.historical_profile import (
    load_historical_profile_prior,
    update_historical_profiles,
)
from .services.skill_flow_loader import load_skill_flow
from .services.translator import Translator


class ReviewOrchestrator:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def run(self, input_data: ReviewRunInput, output_root: Path) -> RunSummary:
        run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        run_dir = self._build_output_dir(output_root, input_data)

        ready, reason = validate_executor_readiness(input_data.options.executor_backend)
        if not ready:
            backend = input_data.options.executor_backend.value
            raise RuntimeError(
                "FATAL: executor backend validation failed for "
                f"`{backend}`. {reason} "
                "Deterministic fallback is disabled for production-style analysis."
            )

        executor = get_executor(input_data.options.executor_backend)
        translator = Translator(executor)
        flow_profile = load_skill_flow(self.repo_root)

        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        ctx = PipelineContext(
            run_id=run_id,
            run_dir=run_dir,
            input_data=input_data,
            repo_root=self.repo_root,
        )
        historical_prior = load_historical_profile_prior(self.repo_root, input_data)
        ctx.artifacts["historical_profile_prior"] = historical_prior
        ctx.dump_json("artifacts/historical_profile_prior.json", historical_prior)

        if flow_profile.warnings:
            for warning in flow_profile.warnings:
                ctx.add_qa_issue(warning)

        skill_flow_payload = {
            "source": flow_profile.source,
            "steps": flow_profile.steps,
            "warnings": flow_profile.warnings,
        }
        runtime_context_payload = {
            "mode": "local_skill_tools_only",
            "rules_source": "local_venue_rules",
            "notes": ["Runtime uses local skill tools only."],
        }
        ctx.artifacts["skill_flow"] = skill_flow_payload
        ctx.artifacts["runtime_context"] = runtime_context_payload
        ctx.dump_json("artifacts/skill_flow_used.json", skill_flow_payload)
        ctx.dump_json("artifacts/runtime_context.json", runtime_context_payload)

        step_statuses = self._init_step_statuses(flow_profile.steps)
        ctx.step_statuses = step_statuses
        step_registry = self._build_step_registry(translator, executor)
        failed_step_name: str | None = None

        try:
            steps = self._build_step_sequence(flow_profile.steps, step_registry)
            for idx, step in enumerate(steps):
                self._mark_step_running(step_statuses, idx)
                try:
                    step.run(ctx)
                    self._mark_step_success(step_statuses, idx)
                except Exception as exc:  # noqa: BLE001
                    ctx.status = RunStatus.FAILED
                    failed_step_name = step.name
                    err_msg = f"step_failed:{step.name}:{exc}"
                    ctx.add_qa_issue(err_msg)
                    self._mark_step_failed(step_statuses, idx, str(exc))
                    self._mark_remaining_steps_skipped(step_statuses, idx + 1, step.name)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "pipeline_exception.log").write_text(
                        traceback.format_exc(), encoding="utf-8"
                    )
                    break
            else:
                self._mark_unset_steps_as_skipped(step_statuses, reason="not_executed")
        except Exception as exc:  # noqa: BLE001
            ctx.status = RunStatus.FAILED
            err_msg = f"orchestrator_failed:{exc}"
            ctx.add_qa_issue(err_msg)
            failed_name = self._extract_unknown_step_name(str(exc))
            if failed_name:
                self._mark_unknown_step_failed(step_statuses, failed_name, str(exc))
                self._mark_remaining_by_name_skipped(step_statuses, failed_name)
                failed_step_name = failed_name
            self._mark_unset_steps_as_skipped(step_statuses, reason="orchestrator_failed")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "pipeline_exception.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )

        ctx.dump_json("pipeline_steps.json", {"steps": step_statuses})
        historical_profile = update_historical_profiles(
            self.repo_root,
            run_id=run_id,
            input_data=input_data,
            risk_ranking=ctx.artifacts.get("risk_ranking"),
            gaps=ctx.artifacts.get("gaps"),
            alignments=ctx.artifacts.get("claim_evidence_matrix"),
        )
        ctx.artifacts["historical_profile"] = historical_profile
        ctx.dump_json("historical_profile.json", historical_profile)
        ctx.dump_json("artifacts/historical_profile.json", historical_profile)
        produced_artifacts = self._collect_produced_artifacts(run_dir)
        summary = RunSummary(
            run_id=run_id,
            status=ctx.status,
            output_dir=str(run_dir),
            qa_issues=ctx.qa_issues,
            step_statuses=step_statuses,
            produced_artifacts=produced_artifacts,
            historical_profile=historical_profile,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_result.json").write_text(
            json.dumps(
                {
                    **summary.model_dump(),
                    "failed_step": failed_step_name,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return summary

    def _build_step_registry(
        self,
        translator: Translator,
        executor: ExecutorAdapter,
    ) -> dict[str, PipelineStep]:
        return {
            "Intake": IntakeStep(),
            "VenueProfileResolver": VenueProfileResolverStep(self.repo_root, executor),
            "PaperParser": PaperParserStep(),
            "ClaimDiscoverer": ClaimDiscovererStep(executor),
            "ClaimNormalizer": ClaimNormalizerStep(executor),
            "EvidenceIndexer": EvidenceIndexerStep(),
            "ClaimEvidenceAligner": ClaimEvidenceAlignerStep(),
            "CitationGraph": CitationGraphStep(),
            "GapDetector": GapDetectorStep(executor),
            "VenueRecommender": VenueRecommenderStep(executor),
            "RiskRanker": RiskRankerStep(executor),
            "ReviewerQuestionSimulator": ReviewerQuestionSimulatorStep(executor),
            "RemediationPlanner": RemediationPlannerStep(executor),
            "RebuttalComposer": RebuttalComposerStep(translator, executor),
            "PaperQAGate": PaperQAGateStep(translator, executor),
            "ReportBuilder": ReportBuilderStep(translator, executor),
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
    def _now_iso() -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _init_step_statuses(self, step_names: list[str]) -> list[dict]:
        return [
            {
                "name": name,
                "status": "pending",
                "started_at": None,
                "ended_at": None,
                "error": None,
                "skip_reason": None,
            }
            for name in step_names
        ]

    def _mark_step_running(self, statuses: list[dict], idx: int) -> None:
        statuses[idx]["status"] = "running"
        statuses[idx]["started_at"] = self._now_iso()
        statuses[idx]["ended_at"] = None
        statuses[idx]["error"] = None
        statuses[idx]["skip_reason"] = None

    def _mark_step_success(self, statuses: list[dict], idx: int) -> None:
        statuses[idx]["status"] = "success"
        if statuses[idx].get("started_at") is None:
            statuses[idx]["started_at"] = self._now_iso()
        statuses[idx]["ended_at"] = self._now_iso()

    def _mark_step_failed(self, statuses: list[dict], idx: int, error: str) -> None:
        statuses[idx]["status"] = "failed"
        if statuses[idx].get("started_at") is None:
            statuses[idx]["started_at"] = self._now_iso()
        statuses[idx]["ended_at"] = self._now_iso()
        statuses[idx]["error"] = error

    def _mark_remaining_steps_skipped(self, statuses: list[dict], start_idx: int, blocker: str) -> None:
        for pos in range(start_idx, len(statuses)):
            if statuses[pos].get("status") in {"pending", "running"}:
                statuses[pos]["status"] = "skipped"
                statuses[pos]["started_at"] = None
                statuses[pos]["ended_at"] = self._now_iso()
                statuses[pos]["skip_reason"] = f"blocked_by:{blocker}"

    def _mark_unset_steps_as_skipped(self, statuses: list[dict], reason: str) -> None:
        for row in statuses:
            if row.get("status") in {"pending", "running"}:
                row["status"] = "skipped"
                row["ended_at"] = self._now_iso()
                row["skip_reason"] = reason

    @staticmethod
    def _extract_unknown_step_name(error: str) -> str | None:
        marker = "unknown_step_in_skill_flow:"
        if marker not in error:
            return None
        return error.split(marker, 1)[1].strip() or None

    def _mark_unknown_step_failed(self, statuses: list[dict], name: str, error: str) -> None:
        for row in statuses:
            if row.get("name") == name:
                row["status"] = "failed"
                row["started_at"] = row.get("started_at") or self._now_iso()
                row["ended_at"] = self._now_iso()
                row["error"] = error
                row["skip_reason"] = None
                return

    def _mark_remaining_by_name_skipped(self, statuses: list[dict], failed_name: str) -> None:
        seen_failed = False
        for row in statuses:
            if row.get("name") == failed_name:
                seen_failed = True
                continue
            if seen_failed and row.get("status") in {"pending", "running"}:
                row["status"] = "skipped"
                row["ended_at"] = self._now_iso()
                row["skip_reason"] = f"blocked_by:{failed_name}"

    @staticmethod
    def _collect_produced_artifacts(run_dir: Path) -> list[str]:
        if not run_dir.exists():
            return []
        out: list[str] = []
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(run_dir).as_posix()
            if rel == "run_result.json":
                continue
            out.append(rel)
        return out

    @staticmethod
    def _build_output_dir(output_root: Path, input_data: ReviewRunInput) -> Path:
        paper_name = Path(input_data.paper.path).stem
        paper_name = re.sub(r'[<>:"/\\|?*]+', " ", paper_name).strip()
        paper_name = re.sub(r"\s+", " ", paper_name)
        if not paper_name:
            paper_name = "paper"
        return output_root / paper_name
