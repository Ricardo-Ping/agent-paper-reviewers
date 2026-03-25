from __future__ import annotations

from dataclasses import dataclass

import requests

from ..models import RebuttalPolicy


@dataclass
class PolicyResolveResult:
    policy: RebuttalPolicy | None
    warning: str | None = None


def resolve_openreview_policy(group_id: str) -> PolicyResolveResult:
    if not group_id:
        return PolicyResolveResult(policy=None, warning="openreview_group_id_missing")

    # Lightweight availability check.
    url = "https://api2.openreview.net/groups"
    try:
        response = requests.get(url, params={"id": group_id}, timeout=8)
    except requests.RequestException as exc:
        return PolicyResolveResult(policy=None, warning=f"openreview_request_failed:{exc}")

    if response.status_code != 200:
        return PolicyResolveResult(policy=None, warning=f"openreview_status_{response.status_code}")

    # OpenReview policy extraction is conference-specific; we keep this resolver conservative.
    return PolicyResolveResult(policy=None, warning="policy_not_extracted_use_fallback")
