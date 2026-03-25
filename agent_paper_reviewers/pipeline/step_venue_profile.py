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

        scoring_axes = list(year_profile.scoring_axes)
        weights = dict(year_profile.weights)
        common_reject_reasons = list(year_profile.common_reject_reasons)
        policy = year_profile.rebuttal_policy
        policy_needs_manual_check = False

        if policy.dynamic_from_openreview:
            if ctx.mcp_tools is None:
                policy_needs_manual_check = True
                ctx.add_qa_issue("policy_resolver_warning:mcp_provider_missing")
            else:
                resolved = ctx.mcp_tools.resolve_openreview_policy(year_profile.openreview_group_id)
                if resolved.warning:
                    ctx.add_qa_issue(f"policy_resolver_warning:{resolved.warning}")

                if resolved.policy:
                    policy = resolved.policy
                else:
                    policy_needs_manual_check = True

                if resolved.profile_overrides:
                    scoring_axes, weights, common_reject_reasons = self._merge_profile_overrides(
                        scoring_axes=scoring_axes,
                        weights=weights,
                        common_reject_reasons=common_reject_reasons,
                        overrides=resolved.profile_overrides,
                    )

        profile_payload = {
            "venue": venue,
            "year": year,
            "used_fallback": used_fallback,
            "source": source,
            "policy_needs_manual_check": policy_needs_manual_check,
            "profile": {
                "scoring_axes": scoring_axes,
                "weights": weights,
                "common_reject_reasons": common_reject_reasons,
                "required_checks": year_profile.required_checks,
                "rebuttal_policy": policy.model_dump(),
                "openreview_group_id": year_profile.openreview_group_id,
                "version_date": year_profile.version_date,
            },
        }
        ctx.artifacts["venue_profile"] = profile_payload
        ctx.dump_json("artifacts/venue_profile.json", profile_payload)

    @staticmethod
    def _merge_profile_overrides(
        *,
        scoring_axes: list[str],
        weights: dict[str, float],
        common_reject_reasons: list[str],
        overrides: dict,
    ) -> tuple[list[str], dict[str, float], list[str]]:
        merged_axes = list(scoring_axes)
        merged_weights = dict(weights)
        merged_reasons = list(common_reject_reasons)

        override_axes = overrides.get("scoring_axes")
        if isinstance(override_axes, list):
            cleaned_axes = [str(x).strip().lower() for x in override_axes if str(x).strip()]
            if cleaned_axes:
                merged_axes = list(dict.fromkeys(cleaned_axes))

        override_weights = overrides.get("weights")
        if isinstance(override_weights, dict):
            for axis, value in override_weights.items():
                axis_key = str(axis).strip().lower()
                try:
                    merged_weights[axis_key] = float(value)
                except (TypeError, ValueError):
                    continue
        merged_weights = VenueProfileResolverStep._normalize_weights(merged_weights, merged_axes)

        override_reasons = overrides.get("common_reject_reasons")
        if isinstance(override_reasons, list):
            cleaned_reasons = [str(x).strip() for x in override_reasons if str(x).strip()]
            if cleaned_reasons:
                merged_reasons = list(dict.fromkeys(cleaned_reasons + merged_reasons))[:10]

        return merged_axes, merged_weights, merged_reasons

    @staticmethod
    def _normalize_weights(raw_weights: dict[str, float], scoring_axes: list[str]) -> dict[str, float]:
        if not scoring_axes:
            return {}

        selected: dict[str, float] = {}
        for axis in scoring_axes:
            value = raw_weights.get(axis)
            try:
                if value is not None:
                    selected[axis] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue

        if not selected:
            equal = round(1.0 / len(scoring_axes), 4)
            return {axis: equal for axis in scoring_axes}

        total = sum(selected.values())
        if total <= 0:
            equal = round(1.0 / len(scoring_axes), 4)
            return {axis: equal for axis in scoring_axes}

        normalized = {axis: round(value / total, 4) for axis, value in selected.items()}
        missing = [axis for axis in scoring_axes if axis not in normalized]
        if missing:
            remainder = max(0.0, 1.0 - sum(normalized.values()))
            fill = round(remainder / len(missing), 4)
            for axis in missing:
                normalized[axis] = fill

        final_total = sum(normalized.values())
        if final_total > 0:
            normalized = {axis: round(value / final_total, 4) for axis, value in normalized.items()}
        return normalized
