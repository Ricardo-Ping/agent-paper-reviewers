from __future__ import annotations

from pathlib import Path

from ..executors.base import ExecutorAdapter
from ..models import TaskSpec
from ..mcp.base import PolicyResolveResult
from ..services.venue_loader import load_venue_profile
from .base import PipelineContext, PipelineStep


class VenueProfileResolverStep(PipelineStep):
    name = "VenueProfileResolver"

    def __init__(
        self,
        repo_root: Path,
        executor: ExecutorAdapter | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.executor = executor

    def run(self, ctx: PipelineContext) -> None:
        venue = ctx.input_data.venue.name
        year = ctx.input_data.venue.year
        year_profile, used_fallback, source = load_venue_profile(self.repo_root, venue, year)

        scoring_axes = list(year_profile.scoring_axes)
        weights = dict(year_profile.weights)
        common_reject_reasons = list(year_profile.common_reject_reasons)
        required_checks = list(year_profile.required_checks)
        required_check_specs = {
            key: value.model_dump() if hasattr(value, "model_dump") else dict(value)
            for key, value in year_profile.required_check_specs.items()
        }
        policy = year_profile.rebuttal_policy
        policy_needs_manual_check = False
        dynamic_focus_weaknesses: list[str] = []
        source_notes: list[str] = []
        openreview_group_id = year_profile.openreview_group_id

        if policy.dynamic_from_openreview and openreview_group_id:
            resolved = self._safe_resolve_policy(ctx, openreview_group_id)
            if resolved is not None and (resolved.policy or resolved.profile_overrides):
                if resolved.policy:
                    policy = resolved.policy
                if resolved.profile_overrides:
                    (
                        scoring_axes,
                        weights,
                        common_reject_reasons,
                        required_checks,
                        required_check_specs,
                        dynamic_focus_weaknesses,
                    ) = self._merge_profile_overrides(
                        scoring_axes=scoring_axes,
                        weights=weights,
                        common_reject_reasons=common_reject_reasons,
                        required_checks=required_checks,
                        required_check_specs=required_check_specs,
                        overrides=resolved.profile_overrides,
                    )
                if resolved.resolved_group_id:
                    openreview_group_id = resolved.resolved_group_id
                if resolved.warning:
                    source_notes.append(f"openreview_policy_warning:{resolved.warning}")
                    ctx.add_qa_issue(f"policy_resolver_warning:{resolved.warning}")

        # Unknown venue path:
        # 1) try MCP discovery by venue/year (silent fallback),
        # 2) if still nothing and executor exists, bootstrap a venue profile draft.
        if used_fallback and source == "fallback_global":
            discovered = self._safe_resolve_policy_by_venue(ctx, venue, year)
            if discovered is not None and (discovered.policy or discovered.profile_overrides):
                if discovered.policy:
                    policy = discovered.policy
                if discovered.profile_overrides:
                    (
                        scoring_axes,
                        weights,
                        common_reject_reasons,
                        required_checks,
                        required_check_specs,
                        dynamic_focus_weaknesses,
                    ) = self._merge_profile_overrides(
                        scoring_axes=scoring_axes,
                        weights=weights,
                        common_reject_reasons=common_reject_reasons,
                        required_checks=required_checks,
                        required_check_specs=required_check_specs,
                        overrides=discovered.profile_overrides,
                    )
                if discovered.resolved_group_id:
                    openreview_group_id = discovered.resolved_group_id
                    source = "fallback_global+openreview_discovered"
                if discovered.warning:
                    source_notes.append(f"openreview_discovery_warning:{discovered.warning}")
                    ctx.add_qa_issue(f"policy_resolver_warning:{discovered.warning}")
            else:
                bootstrapped = self._bootstrap_unknown_venue_with_executor(
                    ctx,
                    venue=venue,
                    year=year,
                    scoring_axes=scoring_axes,
                    weights=weights,
                    common_reject_reasons=common_reject_reasons,
                    required_checks=required_checks,
                    required_check_specs=required_check_specs,
                )
                if bootstrapped is not None:
                    scoring_axes = bootstrapped["scoring_axes"]
                    weights = bootstrapped["weights"]
                    common_reject_reasons = bootstrapped["common_reject_reasons"]
                    required_checks = bootstrapped["required_checks"]
                    required_check_specs = bootstrapped["required_check_specs"]
                    source = "fallback_global+executor_bootstrap"
                    source_notes.append("venue_profile_bootstrapped_by_executor")

        profile_payload = {
            "venue": venue,
            "year": year,
            "used_fallback": used_fallback,
            "source": source,
            "policy_needs_manual_check": policy_needs_manual_check,
            "source_notes": source_notes,
            "profile": {
                "scoring_axes": scoring_axes,
                "weights": weights,
                "common_reject_reasons": common_reject_reasons,
                "dynamic_focus_weaknesses": dynamic_focus_weaknesses,
                "required_checks": required_checks,
                "required_check_specs": required_check_specs,
                "rebuttal_policy": policy.model_dump(),
                "decision_policy": year_profile.decision_policy.model_dump(),
                "openreview_group_id": openreview_group_id,
                "version_date": year_profile.version_date,
            },
        }
        ctx.artifacts["venue_profile"] = profile_payload
        ctx.dump_json("artifacts/venue_profile.json", profile_payload)

    @staticmethod
    def _safe_resolve_policy(ctx: PipelineContext, openreview_group_id: str) -> PolicyResolveResult | None:
        if ctx.mcp_tools is None:
            return None
        try:
            return ctx.mcp_tools.resolve_openreview_policy(openreview_group_id)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _safe_resolve_policy_by_venue(
        ctx: PipelineContext,
        venue_name: str,
        year: int,
    ) -> PolicyResolveResult | None:
        if ctx.mcp_tools is None:
            return None
        try:
            return ctx.mcp_tools.resolve_openreview_policy_by_venue(venue_name, year)
        except Exception:  # noqa: BLE001
            return None

    def _bootstrap_unknown_venue_with_executor(
        self,
        ctx: PipelineContext,
        *,
        venue: str,
        year: int,
        scoring_axes: list[str],
        weights: dict[str, float],
        common_reject_reasons: list[str],
        required_checks: list[str],
        required_check_specs: dict[str, dict],
    ) -> dict | None:
        if self.executor is None:
            return None

        spec = TaskSpec(
            task_type="venue_profile_bootstrap",
            prompt=(
                "You are designing a strict reviewer profile for an unknown venue. "
                "Return JSON only, keep checks concrete and executable."
            ),
            context={
                "venue": venue,
                "year": year,
                "baseline_profile": {
                    "scoring_axes": scoring_axes,
                    "weights": weights,
                    "common_reject_reasons": common_reject_reasons,
                    "required_checks": required_checks,
                    "required_check_specs": required_check_specs,
                },
                "requirements": [
                    "Keep scoring axes from novelty/soundness/experiment/clarity unless venue clearly differs.",
                    "Provide 5-12 required checks.",
                    "Each check spec should include min_hits and concrete keywords.",
                    "Use practical reviewer language.",
                ],
            },
            output_schema={
                "scoring_axes": ["novelty", "soundness", "experiment", "clarity"],
                "weights": {"novelty": 0.25, "soundness": 0.3, "experiment": 0.3, "clarity": 0.15},
                "common_reject_reasons": ["string"],
                "required_checks": ["baseline_coverage", "statistical_significance"],
                "required_check_specs": {
                    "baseline_coverage": {
                        "keywords": ["baseline", "sota"],
                        "min_hits": 2,
                        "min_distinct_sections": 1,
                        "severity_hint": "P1",
                    }
                },
            },
            model_profile="judge",
        )

        result = self.executor.execute(spec)
        for warning in result.warnings:
            ctx.add_qa_issue(f"venue_bootstrap_executor_warning:{warning}")
        if not result.ok:
            return None

        payload = result.output
        if isinstance(payload.get("response"), dict):
            payload = payload["response"]
        if not isinstance(payload, dict):
            return None

        axes = payload.get("scoring_axes")
        if not isinstance(axes, list) or not axes:
            axes = scoring_axes
        axes = [str(x).strip().lower() for x in axes if str(x).strip()]
        axes = list(dict.fromkeys(axes)) or scoring_axes

        raw_weights = payload.get("weights")
        merged_weights = dict(weights)
        if isinstance(raw_weights, dict):
            for axis, value in raw_weights.items():
                axis_key = str(axis).strip().lower()
                try:
                    merged_weights[axis_key] = float(value)
                except (TypeError, ValueError):
                    continue
        merged_weights = self._normalize_weights(merged_weights, axes)

        raw_reasons = payload.get("common_reject_reasons")
        reasons = common_reject_reasons
        if isinstance(raw_reasons, list):
            cleaned = [str(x).strip() for x in raw_reasons if str(x).strip()]
            if cleaned:
                reasons = list(dict.fromkeys(cleaned + reasons))[:10]

        raw_checks = payload.get("required_checks")
        checks = required_checks
        if isinstance(raw_checks, list):
            cleaned = [str(x).strip() for x in raw_checks if str(x).strip()]
            if cleaned:
                checks = list(dict.fromkeys(cleaned))

        merged_specs = dict(required_check_specs)
        raw_specs = payload.get("required_check_specs")
        if isinstance(raw_specs, dict):
            for key, value in raw_specs.items():
                k = str(key).strip()
                if k and isinstance(value, dict):
                    merged_specs[k] = value

        return {
            "scoring_axes": axes,
            "weights": merged_weights,
            "common_reject_reasons": reasons,
            "required_checks": checks,
            "required_check_specs": merged_specs,
        }

    @staticmethod
    def _merge_profile_overrides(
        *,
        scoring_axes: list[str],
        weights: dict[str, float],
        common_reject_reasons: list[str],
        required_checks: list[str],
        required_check_specs: dict[str, dict],
        overrides: dict,
    ) -> tuple[list[str], dict[str, float], list[str], list[str], dict[str, dict], list[str]]:
        merged_axes = list(scoring_axes)
        merged_weights = dict(weights)
        merged_reasons = list(common_reject_reasons)
        merged_checks = list(required_checks)
        merged_specs = dict(required_check_specs)
        dynamic_focus_weaknesses: list[str] = []

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

        override_checks = overrides.get("required_checks")
        if isinstance(override_checks, list):
            cleaned_checks = [str(x).strip() for x in override_checks if str(x).strip()]
            if cleaned_checks:
                merged_checks = list(dict.fromkeys(cleaned_checks))

        override_specs = overrides.get("required_check_specs")
        if isinstance(override_specs, dict):
            for key, value in override_specs.items():
                k = str(key).strip()
                if k and isinstance(value, dict):
                    merged_specs[k] = value

        override_focus = overrides.get("dynamic_focus_weaknesses")
        if isinstance(override_focus, list):
            dynamic_focus_weaknesses = [str(x).strip() for x in override_focus if str(x).strip()]

        return (
            merged_axes,
            merged_weights,
            merged_reasons,
            merged_checks,
            merged_specs,
            dynamic_focus_weaknesses,
        )

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
