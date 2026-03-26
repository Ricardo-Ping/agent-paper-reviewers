from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime
from typing import Any

import requests

from ..models import RebuttalPolicy
from .base import MCPToolProvider, PolicyResolveResult


class HttpMCPToolProvider(MCPToolProvider):
    name = "http_mcp"

    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = (
            base_url
            or os.getenv("AGENT_PAPER_REVIEWERS_OPENREVIEW_BASE_URL")
            or "https://api2.openreview.net"
        ).rstrip("/")
        self.token = token or os.getenv("OPENREVIEW_TOKEN")

    def resolve_openreview_policy(self, group_id: str) -> PolicyResolveResult:
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

        # Keep only non-empty overrides.
        overrides = {k: v for k, v in overrides.items() if v}

        warning = ";".join(dict.fromkeys(warnings)) if warnings else None
        return PolicyResolveResult(
            policy=policy,
            profile_overrides=overrides or None,
            warning=warning,
            resolved_group_id=group_id,
        )

    def resolve_openreview_policy_by_venue(self, venue_name: str, year: int) -> PolicyResolveResult:
        for group_id in self._candidate_group_ids(venue_name, year):
            resolved = self.resolve_openreview_policy(group_id)
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
            warn = f"openreview_request_failed:{exc}"
            return (None, warn)

        if response.status_code == 403:
            # Public OpenReview routes may deny anonymous API calls for some venues.
            # When token is missing, silently fall back to local venue profile so the
            # pipeline can continue without noisy QA warnings.
            if not self.token:
                return (None, None)
            return (None, "openreview_forbidden_with_token")

        if response.status_code != 200:
            warn = f"openreview_status_{response.status_code}:{path}"
            if allow_fail:
                return (None, warn)
            return (None, warn)

        try:
            return response.json(), None
        except ValueError:
            return None, f"openreview_invalid_json:{path}"

    @staticmethod
    def _extract_scoring_axes(group: dict, details: dict, invitations: list[dict]) -> list[str]:
        text = "\n".join(
            HttpMCPToolProvider._collect_text_values(
                {"group": group, "details": details, "invitations": invitations}
            )
        ).lower()
        axes = []
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
    def _extract_weights(group: dict, details: dict, invitations: list[dict], axes: list[str]) -> dict[str, float]:
        if not axes:
            return {}

        candidates = HttpMCPToolProvider._find_weight_candidates(
            {"group": group, "details": details, "invitations": invitations}
        )
        if candidates:
            best = max(candidates, key=lambda d: sum(d.values()))
            return HttpMCPToolProvider._normalize_weights(best, axes)

        text = "\n".join(
            HttpMCPToolProvider._collect_text_values(
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
            return HttpMCPToolProvider._normalize_weights(extracted, axes)
        return {}

    @staticmethod
    def _extract_reject_reasons(group: dict, details: dict, invitations: list[dict]) -> list[str]:
        text_values = HttpMCPToolProvider._collect_text_values(
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

        policy = RebuttalPolicy(
            mode=mode,
            per_review_char_limit=int(per_review_limit),
            global_char_limit=int(global_limit or 0),
            allow_attachment_pdf=allow_attachment_pdf,
            attachment_page_limit=2 if allow_attachment_pdf else 0,
            allow_links=allow_links,
            dynamic_from_openreview=True,
        )
        return policy, warnings

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
        # Fill missing axes with equal tiny split if needed.
        missing = [a for a in axes if a not in normalized]
        if missing:
            remain = max(0.0, 1.0 - sum(normalized.values()))
            add = round(remain / len(missing), 4) if missing else 0.0
            for a in missing:
                normalized[a] = add

        # Final re-normalization.
        s = sum(normalized.values())
        if s > 0:
            normalized = {k: round(v / s, 4) for k, v in normalized.items()}
        return normalized

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
        # use the largest candidate under a reasonable upper bound to avoid accidental huge values
        cands = [x for x in cands if x <= 50000]
        if not cands:
            return None
        return max(cands)

    @staticmethod
    def _detect_attachment_allowed(obj: Any) -> bool:
        text = "\n".join(HttpMCPToolProvider._collect_text_values(obj)).lower()
        return any(k in text for k in ["attachment", "attach", "supplementary", "pdf", "file upload", "upload file"])

    @staticmethod
    def _detect_links_allowed(obj: Any) -> bool:
        text = "\n".join(HttpMCPToolProvider._collect_text_values(obj)).lower()
        return any(k in text for k in ["url", "link", "http://", "https://", "external link"])

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

    def _candidate_group_ids(self, venue_name: str, year: int) -> list[str]:
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
        # Keep order and unique.
        return list(dict.fromkeys(cands))

    def capabilities(self) -> dict[str, bool]:
        return {
            "openreview_policy_resolver": True,
            "openreview_group_discovery": True,
        }

