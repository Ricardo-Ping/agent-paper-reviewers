from __future__ import annotations

from pathlib import Path

from ..services.venue_loader import load_venue_profile
from .base import PipelineContext, PipelineStep


class VenueProfileResolverStep(PipelineStep):
    name = "VenueProfileResolver"

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def run(self, ctx: PipelineContext) -> None:
        venue = ctx.input_data.venue.name
        year = ctx.input_data.venue.year
        year_profile, used_fallback, source = load_venue_profile(self.repo_root, venue, year)

        policy = year_profile.rebuttal_policy
        policy_needs_manual_check = False

        if policy.dynamic_from_openreview:
            if ctx.mcp_tools is None:
                policy_needs_manual_check = True
                ctx.qa_issues.append("policy_resolver_warning:mcp_provider_missing")
            else:
                resolved = ctx.mcp_tools.resolve_openreview_policy(year_profile.openreview_group_id)
                if resolved.policy:
                    policy = resolved.policy
                else:
                    policy_needs_manual_check = True
                    if resolved.warning:
                        ctx.qa_issues.append(f"policy_resolver_warning:{resolved.warning}")

        profile_payload = {
            "venue": venue,
            "year": year,
            "used_fallback": used_fallback,
            "source": source,
            "policy_needs_manual_check": policy_needs_manual_check,
            "profile": {
                "scoring_axes": year_profile.scoring_axes,
                "weights": year_profile.weights,
                "common_reject_reasons": year_profile.common_reject_reasons,
                "required_checks": year_profile.required_checks,
                "rebuttal_policy": policy.model_dump(),
                "openreview_group_id": year_profile.openreview_group_id,
                "version_date": year_profile.version_date,
            },
        }
        ctx.artifacts["venue_profile"] = profile_payload
        ctx.dump_json("artifacts/venue_profile.json", profile_payload)
