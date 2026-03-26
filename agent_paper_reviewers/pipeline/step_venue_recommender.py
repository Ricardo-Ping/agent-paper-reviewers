from __future__ import annotations

from ..services.venue_recommender import recommend_venues
from .base import PipelineContext, PipelineStep


class VenueRecommenderStep(PipelineStep):
    name = "VenueRecommender"

    def run(self, ctx: PipelineContext) -> None:
        paper_structured = ctx.artifacts.get("paper_structured", {})
        claims_normalized = ctx.artifacts.get("claims_normalized", {})
        evidence_index = ctx.artifacts.get("evidence_index", {})
        target_year = int(ctx.input_data.venue.year or 0)

        payload = recommend_venues(
            ctx.repo_root or ctx.run_dir,
            target_year=target_year,
            paper_structured=paper_structured if isinstance(paper_structured, dict) else {},
            claims_normalized=claims_normalized if isinstance(claims_normalized, dict) else {},
            evidence_index=evidence_index if isinstance(evidence_index, dict) else {},
            top_k=5,
        )
        ctx.artifacts["venue_recommendations"] = payload
        ctx.dump_json("artifacts/venue_recommendations.json", payload)
