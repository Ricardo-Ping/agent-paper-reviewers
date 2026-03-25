from __future__ import annotations

from .base import MCPToolProvider, PolicyResolveResult


class NoopMCPToolProvider(MCPToolProvider):
    name = "noop_mcp"

    def resolve_openreview_policy(self, group_id: str) -> PolicyResolveResult:
        return PolicyResolveResult(policy=None, warning="mcp_backend_disabled")

    def capabilities(self) -> dict[str, bool]:
        return {
            "openreview_policy_resolver": False,
        }

