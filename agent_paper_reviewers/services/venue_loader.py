from __future__ import annotations

import yaml
from pathlib import Path

from ..models import VenueProfile, VenueYearProfile


def load_venue_profile(data_root: Path, venue_name: str, year: int) -> tuple[VenueYearProfile, bool, str]:
    normalized = venue_name.strip().lower().replace(" ", "-")
    path = data_root / "data" / "venue_rules" / f"{normalized}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Venue profile not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    profile = VenueProfile.model_validate(payload)
    year_key = str(year)
    if year_key in profile.years:
        return profile.years[year_key], False, "exact_match"

    # Fallback to default/latest known profile.
    fallback_key = str(profile.default_year)
    if fallback_key not in profile.years:
        fallback_key = sorted(profile.years.keys())[-1]
    return profile.years[fallback_key], True, f"fallback_to_{fallback_key}"
