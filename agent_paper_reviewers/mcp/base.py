from __future__ import annotations

from dataclasses import dataclass

from ..models import RebuttalPolicy


@dataclass
class PolicyResolveResult:
    policy: RebuttalPolicy | None
    profile_overrides: dict | None = None
    warning: str | None = None


class MCPToolProvider:
    """Capability provider used by the pipeline.

    The Skill (workflow) decides *when* to call a capability.
    MCP provider decides *how* to execute a concrete capability.
    """

    name = "base"

    def resolve_openreview_policy(self, group_id: str) -> PolicyResolveResult:
        raise NotImplementedError

    def capabilities(self) -> dict[str, bool]:
        return {
            "openreview_policy_resolver": False,
        }
