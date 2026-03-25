from __future__ import annotations

from ..models import ExperimentTask
from .base import PipelineContext, PipelineStep


class RemediationPlannerStep(PipelineStep):
    name = "RemediationPlanner"

    def run(self, ctx: PipelineContext) -> None:
        risks = ctx.artifacts["risk_ranking"]["risks"]
        max_n = ctx.input_data.constraints.max_new_experiments

        tasks = []
        for idx, risk in enumerate(risks[:max_n], start=1):
            effort = "L" if risk["severity"] == "P0" else "M" if risk["severity"] == "P1" else "S"
            task = ExperimentTask(
                id=f"EXP-{idx:03d}",
                risk_id=risk["id"],
                title=f"Mitigate {risk['id']} - {risk['severity']} risk",
                priority="high" if risk["severity"] in {"P0", "P1"} else "medium",
                effort=effort,
                est_time_days=4.0 if effort == "L" else 2.0 if effort == "M" else 1.0,
                est_gpu_hours=36 if effort == "L" else 14 if effort == "M" else 4,
                expected_gain="Reduce rejection likelihood by strengthening claim-evidence linkage.",
                protocol=[
                    "Define exact hypothesis and target claim.",
                    "Run comparison against strong baselines with identical settings.",
                    "Report mean/std over multiple seeds and significance tests.",
                    "Add analysis of failures and limitations.",
                ],
            )
            tasks.append(task.model_dump())

        payload = {"tasks": tasks}
        ctx.artifacts["remediation_plan"] = payload
        ctx.dump_json("artifacts/remediation_plan.json", payload)
