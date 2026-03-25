from __future__ import annotations

from ..models import ExecutorBackend
from .base import ExecutorAdapter
from .deterministic import DeterministicExecutor


class ExternalApiExecutor(DeterministicExecutor):
    """Placeholder executor; currently falls back to deterministic behavior."""


class CodexExecutor(DeterministicExecutor):
    """Placeholder executor for Codex backend."""


def get_executor(backend: ExecutorBackend) -> ExecutorAdapter:
    if backend == ExecutorBackend.CODEX:
        return CodexExecutor()
    if backend in {
        ExecutorBackend.AGENT_API,
        ExecutorBackend.OPENAI,
        ExecutorBackend.ANTHROPIC,
        ExecutorBackend.QWEN,
        ExecutorBackend.LOCAL_VLLM,
    }:
        return ExternalApiExecutor()
    return DeterministicExecutor()
