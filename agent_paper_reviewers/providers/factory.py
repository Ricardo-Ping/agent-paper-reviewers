from __future__ import annotations

from ..models import MCPBackend
from .base import MCPToolProvider
from .http_provider import HttpMCPToolProvider
from .noop_provider import NoopMCPToolProvider


def get_mcp_provider(backend: MCPBackend) -> MCPToolProvider:
    if backend == MCPBackend.DISABLED:
        return NoopMCPToolProvider()
    return HttpMCPToolProvider()

