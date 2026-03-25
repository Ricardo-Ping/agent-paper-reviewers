from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..mcp.http_provider import HttpMCPToolProvider
from ..models import VenueRuleSnapshot, VenueYearProfile
from .venue_loader import (
    fallback_path,
    list_venues,
    load_venue_profile,
    load_venue_snapshot,
    merge_profile_overrides,
    normalize_venue_slug,
    save_venue_snapshot,
    venue_year_path,
)


@dataclass
class VenueRefreshItem:
    venue: str
    year: int
    status: str
    file: str
    openreview_group_id: str
    warning: str | None = None
    source: str | None = None


def refresh_venue_rules(
    repo_root: Path,
    *,
    venue: str = "all",
    year: int,
    openreview_group: str = "",
    dry_run: bool = False,
) -> dict:
    provider = HttpMCPToolProvider()
    venues = _resolve_venues(repo_root, venue)
    today = date.today().isoformat()
    items: list[VenueRefreshItem] = []

    if openreview_group and len(venues) != 1:
        raise ValueError("--openreview-group can only be used when refreshing one venue.")

    for venue_slug in venues:
        target_file = venue_year_path(repo_root, venue_slug, year)
        display_name = _display_name(repo_root, venue_slug)

        try:
            base_profile, used_fallback, source = load_venue_profile(repo_root, venue_slug, year)
            profile = base_profile
            source_note = source
            if used_fallback:
                source_note = f"{source}_as_template"
        except Exception:  # noqa: BLE001
            fb_file = fallback_path(repo_root)
            if not fb_file.exists():
                items.append(
                    VenueRefreshItem(
                        venue=venue_slug,
                        year=year,
                        status="failed",
                        file=str(target_file),
                        openreview_group_id="",
                        warning="template_profile_not_found",
                    )
                )
                continue
            fb_snapshot = load_venue_snapshot(fb_file)
            profile = fb_snapshot.profile
            source_note = "fallback_template"

        group_id = openreview_group.strip() or profile.openreview_group_id
        if not group_id:
            updated = _apply_update(
                profile,
                openreview_group_id="",
                version_date=today,
            )
            snapshot = VenueRuleSnapshot(
                schema_version=1,
                venue=venue_slug,
                year=year,
                display_name=display_name,
                profile=updated,
            )
            if not dry_run:
                save_venue_snapshot(target_file, snapshot)
            items.append(
                VenueRefreshItem(
                    venue=venue_slug,
                    year=year,
                    status="saved_without_openreview_sync",
                    file=str(target_file),
                    openreview_group_id="",
                    warning="openreview_group_id_missing",
                    source=source_note,
                )
            )
            continue

        resolved = provider.resolve_openreview_policy(group_id)
        updated_profile = profile
        if resolved.profile_overrides:
            updated_profile = merge_profile_overrides(updated_profile, resolved.profile_overrides)
        if resolved.policy:
            updated_profile = updated_profile.model_copy(update={"rebuttal_policy": resolved.policy})

        updated_profile = _apply_update(
            updated_profile,
            openreview_group_id=group_id,
            version_date=today,
        )

        snapshot = VenueRuleSnapshot(
            schema_version=1,
            venue=venue_slug,
            year=year,
            display_name=display_name,
            profile=updated_profile,
        )
        if not dry_run:
            save_venue_snapshot(target_file, snapshot)

        status = "synced" if resolved.policy or resolved.profile_overrides else "saved_with_template_only"
        items.append(
            VenueRefreshItem(
                venue=venue_slug,
                year=year,
                status=status,
                file=str(target_file),
                openreview_group_id=group_id,
                warning=resolved.warning,
                source=source_note,
            )
        )

    changelog_file = repo_root / "data" / "venue_rules" / "changelog.md"
    _append_changelog(changelog_file, items, dry_run=dry_run)

    return {
        "year": year,
        "dry_run": dry_run,
        "items": [item.__dict__ for item in items],
        "updated_count": sum(1 for i in items if i.status in {"synced", "saved_without_openreview_sync", "saved_with_template_only"}),
        "failed_count": sum(1 for i in items if i.status == "failed"),
    }


def _resolve_venues(repo_root: Path, venue: str) -> list[str]:
    value = (venue or "all").strip().lower()
    if value == "all":
        venues = list_venues(repo_root)
        return venues
    return sorted(
        {
            normalize_venue_slug(v)
            for v in value.split(",")
            if v.strip()
        }
    )


def _display_name(repo_root: Path, venue_slug: str) -> str:
    years = []
    venue_dir = repo_root / "data" / "venue_rules" / venue_slug
    if venue_dir.exists():
        for file in venue_dir.glob("*.yaml"):
            if file.stem.isdigit():
                years.append(int(file.stem))
    if years:
        latest = max(years)
        snap = load_venue_snapshot(venue_dir / f"{latest}.yaml")
        if snap.display_name:
            return snap.display_name
    return venue_slug.upper().replace("-", "-")


def _apply_update(profile: VenueYearProfile, *, openreview_group_id: str, version_date: str) -> VenueYearProfile:
    return profile.model_copy(
        update={
            "openreview_group_id": openreview_group_id,
            "version_date": version_date,
        }
    )


def _append_changelog(changelog_file: Path, items: list[VenueRefreshItem], *, dry_run: bool) -> None:
    if dry_run:
        return
    changelog_file.parent.mkdir(parents=True, exist_ok=True)
    if not changelog_file.exists():
        changelog_file.write_text("# Venue Policy Changelog\n\n", encoding="utf-8")

    lines = [f"## Refresh {date.today().isoformat()}", ""]
    for item in items:
        warning_part = f" | warning={item.warning}" if item.warning else ""
        source_part = f" | source={item.source}" if item.source else ""
        lines.append(
            f"- {item.venue} {item.year}: {item.status}{source_part}{warning_part}"
        )
    lines.append("")

    with changelog_file.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
