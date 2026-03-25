from __future__ import annotations

import requests

from .base import MCPToolProvider, PolicyResolveResult


class HttpMCPToolProvider(MCPToolProvider):
    name = "http_mcp"

    def resolve_openreview_policy(self, group_id: str) -> PolicyResolveResult:
        if not group_id:
            return PolicyResolveResult(policy=None, warning="openreview_group_id_missing")

        url = "https://api2.openreview.net/groups"
        try:
            response = requests.get(url, params={"id": group_id}, timeout=8)
        except requests.RequestException as exc:
            return PolicyResolveResult(policy=None, warning=f"openreview_request_failed:{exc}")

        if response.status_code != 200:
            return PolicyResolveResult(policy=None, warning=f"openreview_status_{response.status_code}")

        # Conference-specific extraction varies; keep conservative behavior.
        return PolicyResolveResult(policy=None, warning="policy_not_extracted_use_fallback")

    def capabilities(self) -> dict[str, bool]:
        return {
            "openreview_policy_resolver": True,
        }

