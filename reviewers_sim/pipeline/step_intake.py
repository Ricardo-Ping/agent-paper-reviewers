from __future__ import annotations

from dataclasses import asdict

from .base import PipelineContext, PipelineStep


class IntakeStep(PipelineStep):
    name = "Intake"

    def run(self, ctx: PipelineContext) -> None:
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        ctx.dump_json("artifacts/input.normalized.json", ctx.input_data.model_dump())
        ctx.artifacts["run_metadata"] = {
            "run_id": ctx.run_id,
            "venue": ctx.input_data.venue.model_dump(),
            "paper": ctx.input_data.paper.model_dump(),
            "claims_count": len(ctx.input_data.claims),
        }
        ctx.dump_json("artifacts/run_metadata.json", ctx.artifacts["run_metadata"])
