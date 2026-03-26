from __future__ import annotations

import os

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


def validate_executor_readiness(backend: ExecutorBackend) -> tuple[bool, str]:
    if backend in {ExecutorBackend.CODEX, ExecutorBackend.OPENAI}:
        if os.getenv("OPENAI_API_KEY"):
            return True, "OPENAI_API_KEY detected."
        if os.getenv("AGENT_PAPER_REVIEWERS_CODEX_API_KEY"):
            return True, "AGENT_PAPER_REVIEWERS_CODEX_API_KEY detected."
        return (
            False,
            "OPENAI_API_KEY or AGENT_PAPER_REVIEWERS_CODEX_API_KEY is required for this backend.",
        )

    if backend == ExecutorBackend.ANTHROPIC:
        if os.getenv("ANTHROPIC_API_KEY"):
            return True, "ANTHROPIC_API_KEY detected."
        return False, "ANTHROPIC_API_KEY is required for anthropic backend."

    if backend == ExecutorBackend.QWEN:
        if os.getenv("QWEN_API_KEY"):
            return True, "QWEN_API_KEY detected."
        return False, "QWEN_API_KEY is required for qwen backend."

    if backend == ExecutorBackend.AGENT_API:
        return (
            True,
            "Using OpenClaw Agent API backend. Ensure OpenClaw service endpoint is reachable.",
        )

    if backend == ExecutorBackend.LOCAL_VLLM:
        return (
            True,
            "Using local_vllm backend. Ensure LOCAL_VLLM_BASE_URL endpoint is reachable and model is loaded.",
        )

    return False, "DeterministicExecutor is placeholder-only and should not be used for production analysis."
