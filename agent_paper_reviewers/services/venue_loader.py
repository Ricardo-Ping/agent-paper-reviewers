from __future__ import annotations

from pathlib import Path

import yaml

from ..models import VenueProfile, VenueRuleSnapshot, VenueYearProfile


def normalize_venue_slug(venue_name: str) -> str:
    return venue_name.strip().lower().replace(" ", "-").replace("_", "-")


def rules_root(data_root: Path) -> Path:
    return data_root / "data" / "venue_rules"


def venue_dir_path(data_root: Path, venue_slug: str) -> Path:
    return rules_root(data_root) / venue_slug


def venue_year_path(data_root: Path, venue_slug: str, year: int) -> Path:
    return venue_dir_path(data_root, venue_slug) / f"{year}.yaml"


def fallback_path(data_root: Path) -> Path:
    return rules_root(data_root) / "_fallback.yaml"


def list_venues(data_root: Path) -> list[str]:
    root = rules_root(data_root)
    if not root.exists():
        return []
    venues = []
    for child in root.iterdir():
        if child.is_dir() and not child.name.startswith("_"):
            venues.append(child.name)
    return sorted(venues)


def list_years_for_venue(data_root: Path, venue_slug: str) -> list[int]:
    d = venue_dir_path(data_root, venue_slug)
    if not d.exists():
        return []
    years: list[int] = []
    for file in d.glob("*.yaml"):
        stem = file.stem
        if stem.isdigit():
            years.append(int(stem))
    return sorted(set(years))


def load_venue_profile(data_root: Path, venue_name: str, year: int) -> tuple[VenueYearProfile, bool, str]:
    venue_slug = normalize_venue_slug(venue_name)

    exact_file = venue_year_path(data_root, venue_slug, year)
    if exact_file.exists():
        snapshot = load_venue_snapshot(exact_file)
        return snapshot.profile, False, "exact_match"

    years = list_years_for_venue(data_root, venue_slug)
    if years:
        candidate_years = [y for y in years if y <= year]
        fallback_year = max(candidate_years) if candidate_years else max(years)
        snapshot = load_venue_snapshot(venue_year_path(data_root, venue_slug, fallback_year))
        return snapshot.profile, True, f"fallback_to_{fallback_year}"

    # Legacy compatibility path: old flat file format data/venue_rules/<venue>.yaml
    legacy = rules_root(data_root) / f"{venue_slug}.yaml"
    if legacy.exists():
        payload = yaml.safe_load(legacy.read_text(encoding="utf-8"))
        profile = VenueProfile.model_validate(payload)
        year_key = str(year)
        if year_key in profile.years:
            return profile.years[year_key], False, "legacy_exact_match"

        fallback_key = str(profile.default_year)
        if fallback_key not in profile.years:
            fallback_key = sorted(profile.years.keys())[-1]
        return profile.years[fallback_key], True, f"legacy_fallback_to_{fallback_key}"

    fb_file = fallback_path(data_root)
    if not fb_file.exists():
        raise FileNotFoundError(f"fallback venue profile not found: {fb_file}")
    fb_snapshot = load_venue_snapshot(fb_file)
    return fb_snapshot.profile, True, "fallback_global"


def load_venue_snapshot(path: Path) -> VenueRuleSnapshot:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return VenueRuleSnapshot.model_validate(payload)


def save_venue_snapshot(path: Path, snapshot: VenueRuleSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot.model_dump()
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def merge_profile_overrides(profile: VenueYearProfile, overrides: dict | None) -> VenueYearProfile:
    if not overrides:
        return profile

    scoring_axes = list(profile.scoring_axes)
    weights = dict(profile.weights)
    common_reject_reasons = list(profile.common_reject_reasons)
    required_checks = list(profile.required_checks)
    required_check_specs = dict(profile.required_check_specs)

    override_axes = overrides.get("scoring_axes")
    if isinstance(override_axes, list):
        cleaned_axes = [str(x).strip().lower() for x in override_axes if str(x).strip()]
        if cleaned_axes:
            scoring_axes = list(dict.fromkeys(cleaned_axes))

    override_weights = overrides.get("weights")
    if isinstance(override_weights, dict):
        for axis, value in override_weights.items():
            axis_key = str(axis).strip().lower()
            try:
                weights[axis_key] = float(value)
            except (TypeError, ValueError):
                continue
        weights = _normalize_weights(weights, scoring_axes)

    override_reasons = overrides.get("common_reject_reasons")
    if isinstance(override_reasons, list):
        cleaned_reasons = [str(x).strip() for x in override_reasons if str(x).strip()]
        if cleaned_reasons:
            common_reject_reasons = list(dict.fromkeys(cleaned_reasons + common_reject_reasons))[:10]

    override_required_checks = overrides.get("required_checks")
    if isinstance(override_required_checks, list):
        cleaned_checks = [str(x).strip() for x in override_required_checks if str(x).strip()]
        if cleaned_checks:
            required_checks = list(dict.fromkeys(cleaned_checks))

    override_required_specs = overrides.get("required_check_specs")
    if isinstance(override_required_specs, dict):
        merged_specs = dict(required_check_specs)
        for key, value in override_required_specs.items():
            k = str(key).strip()
            if not k:
                continue
            if isinstance(value, dict):
                merged_specs[k] = value
        required_check_specs = merged_specs

    return VenueYearProfile(
        scoring_axes=scoring_axes,
        weights=weights,
        common_reject_reasons=common_reject_reasons,
        required_checks=required_checks,
        required_check_specs=required_check_specs,
        rebuttal_policy=profile.rebuttal_policy,
        decision_policy=profile.decision_policy,
        openreview_group_id=profile.openreview_group_id,
        version_date=profile.version_date,
    )


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
