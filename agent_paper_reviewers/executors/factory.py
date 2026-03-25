from __future__ import annotations

from ..models import ExecutorBackend
from .anthropic_executor import AnthropicExecutor
from .base import ExecutorAdapter
from .deterministic import DeterministicExecutor
from .openai_compatible_executor import (
    build_codex_executor,
    build_local_vllm_executor,
    build_openai_executor,
    build_qwen_executor,
)
from .openclawnode_executor import OpenClawNodeExecutor


def get_executor(backend: ExecutorBackend) -> ExecutorAdapter:
    if backend == ExecutorBackend.CODEX:
        return build_codex_executor()
    if backend == ExecutorBackend.AGENT_API:
        return OpenClawNodeExecutor()
    if backend == ExecutorBackend.OPENAI:
        return build_openai_executor()
    if backend == ExecutorBackend.ANTHROPIC:
        return AnthropicExecutor()
    if backend == ExecutorBackend.QWEN:
        return build_qwen_executor()
    if backend == ExecutorBackend.LOCAL_VLLM:
        return build_local_vllm_executor()
    return DeterministicExecutor()
