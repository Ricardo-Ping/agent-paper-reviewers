from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests

from ..models import RebuttalPolicy, VenueRuleSnapshot, VenueYearProfile
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
class PolicyResolveResult:
    policy: RebuttalPolicy | None
    profile_overrides: dict | None = None
    warning: str | None = None
    resolved_group_id: str | None = None


@dataclass
class VenueRefreshItem:
    venue: str
    year: int
    status: str
    file: str
    openreview_group_id: str
    warning: str | None = None
    source: str | None = None


class OpenReviewPolicyResolver:
    """Direct OpenReview resolver used by refresh-venue.

    Runtime pipeline no longer depends on provider abstractions.
    This lightweight resolver is a service-level utility for optional rule refresh.
    """

    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = (
            base_url
            or os.getenv("AGENT_PAPER_REVIEWERS_OPENREVIEW_BASE_URL")
            or "https://api2.openreview.net"
        ).rstrip("/")
        self.token = token or os.getenv("OPENREVIEW_TOKEN")

    def resolve_policy(self, group_id: str) -> PolicyResolveResult:
        if not group_id:
            return PolicyResolveResult(policy=None, warning="openreview_group_id_missing", resolved_group_id=None)

        warnings: list[str] = []
        group_data, group_warn = self._fetch_json("/groups", {"id": group_id})
        if group_warn:
            warnings.append(group_warn)
        if not isinstance(group_data, dict):
            if not warnings and not self.token:
                return PolicyResolveResult(policy=None, warning=None, resolved_group_id=group_id)
            return PolicyResolveResult(
                policy=None,
                warning=";".join(warnings) if warnings else "openreview_group_fetch_failed",
                resolved_group_id=group_id,
            )

        groups = group_data.get("groups")
        if not isinstance(groups, list) or not groups:
            warnings.append("openreview_group_not_found")
            return PolicyResolveResult(policy=None, warning=";".join(warnings), resolved_group_id=group_id)

        group = groups[0] if isinstance(groups[0], dict) else {}
        details = group.get("details") if isinstance(group.get("details"), dict) else {}

        invitations: list[dict] = []
        inv_data, inv_warn = self._fetch_json("/invitations", {"prefix": group_id}, allow_fail=True)
        if inv_warn:
            warnings.append(inv_warn)
        if isinstance(inv_data, dict) and isinstance(inv_data.get("invitations"), list):
            invitations.extend(x for x in inv_data["invitations"] if isinstance(x, dict))

        if not invitations:
            inv_data2, inv_warn2 = self._fetch_json("/invitations", {"invitee": group_id}, allow_fail=True)
            if inv_warn2:
                warnings.append(inv_warn2)
            if isinstance(inv_data2, dict) and isinstance(inv_data2.get("invitations"), list):
                invitations.extend(x for x in inv_data2["invitations"] if isinstance(x, dict))

        scoring_axes = self._extract_scoring_axes(group, details, invitations)
        weights = self._extract_weights(group, details, invitations, scoring_axes)
        common_reject = self._extract_reject_reasons(group, details, invitations)
        trend_reasons = self._extract_recent_weakness_trends(group_id)
        if trend_reasons:
            common_reject = list(dict.fromkeys(common_reject + trend_reasons))[:10]

        policy, policy_warns = self._extract_rebuttal_policy(group, details, invitations)
        warnings.extend(policy_warns)

        overrides = {
            "scoring_axes": scoring_axes,
            "weights": weights,
            "common_reject_reasons": common_reject,
            "dynamic_focus_weaknesses": trend_reasons,
        }
        overrides = {k: v for k, v in overrides.items() if v}

        warning = ";".join(dict.fromkeys(warnings)) if warnings else None
        return PolicyResolveResult(
            policy=policy,
            profile_overrides=overrides or None,
            warning=warning,
            resolved_group_id=group_id,
        )

    def resolve_policy_by_venue(self, venue_name: str, year: int) -> PolicyResolveResult:
        for group_id in self._candidate_group_ids(venue_name, year):
            resolved = self.resolve_policy(group_id)
            if resolved.policy is not None or resolved.profile_overrides is not None:
                if not resolved.resolved_group_id:
                    resolved.resolved_group_id = group_id
                return resolved
        return PolicyResolveResult(policy=None, warning=None, resolved_group_id=None)

    def _fetch_json(
        self,
        path: str,
        params: dict[str, Any],
        *,
        allow_fail: bool = False,
    ) -> tuple[dict[str, Any] | None, str | None]:
        headers = {"User-Agent": "agent-paper-reviewers/0.1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = requests.get(
                f"{self.base_url}{path}",
                params=params,
                headers=headers,
                timeout=10,
            )
        except requests.RequestException as exc:
            return None, f"openreview_request_failed:{exc}"

        if response.status_code == 403:
            if not self.token:
                return None, None
            return None, "openreview_forbidden_with_token"
        if response.status_code != 200:
            warn = f"openreview_status_{response.status_code}:{path}"
            return (None, warn)
        try:
            return response.json(), None
        except ValueError:
            return None, f"openreview_invalid_json:{path}"

    @staticmethod
    def _collect_text_values(obj: Any) -> list[str]:
        out: list[str] = []

        def walk(x: Any) -> None:
            if isinstance(x, str):
                s = x.strip()
                if s:
                    out.append(s)
                return
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
                return
            if isinstance(x, list):
                for v in x:
                    walk(v)

        walk(obj)
        return out

    @staticmethod
    def _extract_scoring_axes(group: dict, details: dict, invitations: list[dict]) -> list[str]:
        text = "\n".join(
            OpenReviewPolicyResolver._collect_text_values(
                {"group": group, "details": details, "invitations": invitations}
            )
        ).lower()
        axes: list[str] = []
        rules = [
            ("novelty", ["novelty", "originality"]),
            ("soundness", ["soundness", "technical correctness", "correctness", "technical quality", "validity"]),
            ("experiment", ["experiment", "empirical", "evaluation", "results"]),
            ("clarity", ["clarity", "presentation", "writing", "readability"]),
        ]
        for axis, keys in rules:
            if any(k in text for k in keys):
                axes.append(axis)
        return axes

    @staticmethod
    def _find_weight_candidates(obj: Any) -> list[dict[str, float]]:
        candidates: list[dict[str, float]] = []

        def walk(x: Any) -> None:
            if isinstance(x, dict):
                converted: dict[str, float] = {}
                for k, v in x.items():
                    key = str(k).lower()
                    axis = None
                    if any(t in key for t in ["novelty", "originality"]):
                        axis = "novelty"
                    elif any(t in key for t in ["soundness", "correctness", "technical"]):
                        axis = "soundness"
                    elif any(t in key for t in ["experiment", "empirical", "evaluation"]):
                        axis = "experiment"
                    elif any(t in key for t in ["clarity", "presentation", "writing"]):
                        axis = "clarity"
                    if axis is not None:
                        try:
                            converted[axis] = float(v)
                        except (TypeError, ValueError):
                            pass
                if converted:
                    candidates.append(converted)
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(obj)
        return candidates

    @staticmethod
    def _normalize_weights(raw: dict[str, float], axes: list[str]) -> dict[str, float]:
        selected = {k: max(0.0, float(v)) for k, v in raw.items() if k in axes}
        if not selected:
            return {}
        total = sum(selected.values())
        if total <= 0:
            equal = round(1.0 / len(axes), 4)
            return {k: equal for k in axes}
        normalized = {k: round(v / total, 4) for k, v in selected.items()}
        missing = [a for a in axes if a not in normalized]
        if missing:
            remain = max(0.0, 1.0 - sum(normalized.values()))
            add = round(remain / len(missing), 4) if missing else 0.0
            for a in missing:
                normalized[a] = add
        s = sum(normalized.values())
        if s > 0:
            normalized = {k: round(v / s, 4) for k, v in normalized.items()}
        return normalized

    @staticmethod
    def _extract_weights(group: dict, details: dict, invitations: list[dict], axes: list[str]) -> dict[str, float]:
        if not axes:
            return {}
        candidates = OpenReviewPolicyResolver._find_weight_candidates(
            {"group": group, "details": details, "invitations": invitations}
        )
        if candidates:
            best = max(candidates, key=lambda d: sum(d.values()))
            return OpenReviewPolicyResolver._normalize_weights(best, axes)
        text = "\n".join(
            OpenReviewPolicyResolver._collect_text_values(
                {"group": group, "details": details, "invitations": invitations}
            )
        )
        extracted: dict[str, float] = {}
        patterns = {
            "novelty": r"(?:novelty|originality)\s*[:=]\s*(\d{1,3})\s*%?",
            "soundness": r"(?:soundness|technical\s+correctness|correctness)\s*[:=]\s*(\d{1,3})\s*%?",
            "experiment": r"(?:experiment|empirical|evaluation)\s*[:=]\s*(\d{1,3})\s*%?",
            "clarity": r"(?:clarity|presentation|writing)\s*[:=]\s*(\d{1,3})\s*%?",
        }
        for axis, pattern in patterns.items():
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                extracted[axis] = float(m.group(1))
        if extracted:
            return OpenReviewPolicyResolver._normalize_weights(extracted, axes)
        return {}

    @staticmethod
    def _extract_reject_reasons(group: dict, details: dict, invitations: list[dict]) -> list[str]:
        text_values = OpenReviewPolicyResolver._collect_text_values(
            {"group": group, "details": details, "invitations": invitations}
        )
        reasons: list[str] = []
        keyword_map = [
            ("baseline", "Insufficient comparison with strong baselines."),
            ("significance", "Statistical significance analysis is missing or unclear."),
            ("ablation", "Ablation study is incomplete for key components."),
            ("reproduc", "Reproducibility details are insufficient."),
            ("clarity", "Writing or presentation clarity is below venue expectation."),
            ("novelty", "Novelty and positioning versus prior work are not convincing."),
        ]
        all_text = "\n".join(text_values).lower()
        for key, reason in keyword_map:
            if key in all_text and reason not in reasons:
                reasons.append(reason)
        for line in text_values:
            l = line.strip()
            ll = l.lower()
            if len(l) < 25:
                continue
            if any(k in ll for k in ["reject", "weakness", "missing", "insufficient", "not clear"]):
                if l not in reasons:
                    reasons.append(l)
            if len(reasons) >= 6:
                break
        return reasons[:6]

    @staticmethod
    def _collect_numeric_limits(obj: Any) -> list[tuple[str, float]]:
        limits: list[tuple[str, float]] = []

        def walk(x: Any, path: str) -> None:
            if isinstance(x, dict):
                for k, v in x.items():
                    key = str(k)
                    new_path = f"{path}.{key}" if path else key
                    lk = key.lower()
                    if any(t in lk for t in ["maxlength", "max_length", "char", "word_limit", "wordlimit"]):
                        try:
                            num = float(v)
                            if 1 <= num <= 200000:
                                limits.append((new_path.lower(), num))
                        except (TypeError, ValueError):
                            pass
                    walk(v, new_path)
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{path}[{i}]")

        walk(obj, "")
        return limits

    @staticmethod
    def _choose_limit(limits: list[tuple[str, float]], keywords: list[str]) -> int | None:
        cands = [int(v) for p, v in limits if any(k in p for k in keywords)]
        if not cands:
            return None
        cands = [x for x in cands if x <= 50000]
        if not cands:
            return None
        return max(cands)

    @staticmethod
    def _detect_attachment_allowed(obj: Any) -> bool:
        text = "\n".join(OpenReviewPolicyResolver._collect_text_values(obj)).lower()
        return any(k in text for k in ["attachment", "attach", "supplementary", "pdf", "file upload", "upload file"])

    @staticmethod
    def _detect_links_allowed(obj: Any) -> bool:
        text = "\n".join(OpenReviewPolicyResolver._collect_text_values(obj)).lower()
        return any(k in text for k in ["url", "link", "http://", "https://", "external link"])

    def _extract_rebuttal_policy(
        self,
        group: dict,
        details: dict,
        invitations: list[dict],
    ) -> tuple[RebuttalPolicy, list[str]]:
        warnings: list[str] = []
        combined_sources: list[dict | list] = [group, details]
        if invitations:
            combined_sources.append(invitations)
        else:
            warnings.append("openreview_invitations_missing_use_partial_policy")

        limits = self._collect_numeric_limits(combined_sources)
        per_review_limit = self._choose_limit(limits, keywords=["rebuttal", "response", "comment", "author"])
        global_limit = self._choose_limit(limits, keywords=["global", "overall", "meta"])
        if per_review_limit is None:
            per_review_limit = 2500
            warnings.append("openreview_char_limit_not_found_use_default_2500")

        allow_attachment_pdf = self._detect_attachment_allowed(combined_sources)
        allow_links = self._detect_links_allowed(combined_sources)
        mode = "per_review_only"
        if global_limit and global_limit > 0:
            mode = "global+per_review"
        return (
            RebuttalPolicy(
                mode=mode,
                per_review_char_limit=int(per_review_limit),
                global_char_limit=int(global_limit or 0),
                allow_attachment_pdf=allow_attachment_pdf,
                attachment_page_limit=2 if allow_attachment_pdf else 0,
                allow_links=allow_links,
                dynamic_from_openreview=True,
            ),
            warnings,
        )

    def _extract_recent_weakness_trends(self, group_id: str) -> list[str]:
        m = re.match(r"^(?P<prefix>.+?)/(?P<year>20\d{2})/Conference$", group_id)
        if not m:
            return []
        prefix = m.group("prefix")
        year = int(m.group("year"))
        years = [year, year - 1, year - 2]
        review_invitation_suffixes = [
            "Official_Review",
            "Meta_Review",
            "Ethics_Review",
            "Senior_Area_Chair_Comment",
        ]
        text_blobs: list[str] = []
        for y in years:
            for suffix in review_invitation_suffixes:
                invitation = f"{prefix}/{y}/Conference/-/{suffix}"
                notes_data, _ = self._fetch_json("/notes", {"invitation": invitation, "limit": 200}, allow_fail=True)
                if not isinstance(notes_data, dict):
                    continue
                notes = notes_data.get("notes")
                if not isinstance(notes, list):
                    continue
                for note in notes:
                    if not isinstance(note, dict):
                        continue
                    content = note.get("content")
                    if isinstance(content, dict):
                        text_blobs.extend(self._collect_text_values(content))
        if not text_blobs:
            return []
        trend_map = {
            "baseline": "Recent OpenReview trend: baseline comparisons are often considered insufficient.",
            "significance": "Recent OpenReview trend: significance reporting and confidence intervals are frequently missing.",
            "ablation": "Recent OpenReview trend: ablation coverage is often questioned.",
            "reproduc": "Recent OpenReview trend: reproducibility details are a recurrent rejection trigger.",
            "clarity": "Recent OpenReview trend: writing clarity and paper organization are repeatedly criticized.",
            "novelty": "Recent OpenReview trend: novelty positioning versus prior work is frequently challenged.",
            "limitation": "Recent OpenReview trend: limitations and failure modes are expected to be explicit.",
            "robust": "Recent OpenReview trend: robustness checks are increasingly expected.",
            "workload": "Recent OpenReview trend: workload diversity and benchmark representativeness are closely reviewed.",
            "scalability": "Recent OpenReview trend: scalability evidence is often scrutinized.",
        }
        counts: Counter[str] = Counter()
        joined = "\n".join(text_blobs).lower()
        for key in trend_map:
            counts[key] = len(re.findall(rf"\b{re.escape(key)}\w*\b", joined))
        reasons: list[str] = []
        for key, _ in counts.most_common(4):
            if counts[key] <= 0:
                continue
            reasons.append(trend_map[key])
        return reasons

    @staticmethod
    def _candidate_group_ids(venue_name: str, year: int) -> list[str]:
        venue = venue_name.strip().lower().replace("_", "-").replace(" ", "-")
        token = re.sub(r"[^a-z0-9]", "", venue).upper()
        now_year = datetime.now().year
        years = [year]
        if year < now_year:
            years.extend([year + 1, year - 1])
        else:
            years.extend([year - 1, year - 2])
        years = [y for y in years if 2000 <= y <= now_year + 1]
        alias = {
            "acl-arr": "ACL",
            "acl": "ACL",
            "emnlp": "EMNLP",
            "neurips": "NeurIPS",
            "iclr": "ICLR",
            "icml": "ICML",
            "kdd": "KDD",
            "aaai": "AAAI",
            "cvpr": "CVPR",
            "eccv": "ECCV",
            "sigmod": "SIGMOD",
            "vldb": "VLDB",
            "icde": "ICDE",
        }
        pretty = alias.get(venue, token)
        cands: list[str] = []
        for y in years:
            cands.extend(
                [
                    f"{pretty}.cc/{y}/Conference",
                    f"{pretty}/{y}/Conference",
                    f"{token}.cc/{y}/Conference",
                    f"{token}/{y}/Conference",
                ]
            )
        return list(dict.fromkeys(cands))


def refresh_venue_rules(
    repo_root: Path,
    *,
    venue: str = "all",
    year: int,
    openreview_group: str = "",
    dry_run: bool = False,
) -> dict:
    resolver = OpenReviewPolicyResolver()
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
            source_note = f"{source}_as_template" if used_fallback else source
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
            updated = _apply_update(profile, openreview_group_id="", version_date=today)
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

        resolved = resolver.resolve_policy(group_id)
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
        return list_venues(repo_root)
    return sorted({normalize_venue_slug(v) for v in value.split(",") if v.strip()})


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
        lines.append(f"- {item.venue} {item.year}: {item.status}{source_part}{warning_part}")
    lines.append("")
    with changelog_file.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
