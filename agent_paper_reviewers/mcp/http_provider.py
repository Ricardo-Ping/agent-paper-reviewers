from __future__ import annotations

import os
import re
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
            return PolicyResolveResult(policy=None, warning="openreview_group_id_missing")

        warnings: list[str] = []
        group_data, group_warn = self._fetch_json("/groups", {"id": group_id})
        if group_warn:
            warnings.append(group_warn)
        if not isinstance(group_data, dict):
            if not warnings and not self.token:
                return PolicyResolveResult(policy=None, warning=None)
            return PolicyResolveResult(policy=None, warning=";".join(warnings) if warnings else "openreview_group_fetch_failed")

        groups = group_data.get("groups")
        if not isinstance(groups, list) or not groups:
            warnings.append("openreview_group_not_found")
            return PolicyResolveResult(policy=None, warning=";".join(warnings))

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

        policy, policy_warns = self._extract_rebuttal_policy(group, details, invitations)
        warnings.extend(policy_warns)

        overrides = {
            "scoring_axes": scoring_axes,
            "weights": weights,
            "common_reject_reasons": common_reject,
        }

        # Keep only non-empty overrides.
        overrides = {k: v for k, v in overrides.items() if v}

        warning = ";".join(dict.fromkeys(warnings)) if warnings else None
        return PolicyResolveResult(
            policy=policy,
            profile_overrides=overrides or None,
            warning=warning,
        )

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

    def capabilities(self) -> dict[str, bool]:
        return {
            "openreview_policy_resolver": True,
        }

