from __future__ import annotations

import os
from typing import Any

import httpx

from ..models import TaskResult, TaskSpec
from .base import ExecutorAdapter
from .deterministic import DeterministicExecutor
from .utils import build_llm_prompt, normalize_output


class OpenAICompatibleExecutor(ExecutorAdapter):
    """Generic executor for OpenAI-compatible chat completion endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        default_model: str,
        key_header: str = "Authorization",
        key_prefix: str = "Bearer ",
        silent_fallback: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.key_header = key_header
        self.key_prefix = key_prefix
        self.silent_fallback = silent_fallback
        self._fallback = DeterministicExecutor()

    def execute(self, spec: TaskSpec) -> TaskResult:
        if not self.api_key:
            fallback = self._fallback.execute(spec)
            if not self.silent_fallback:
                fallback.warnings.append("executor_api_key_missing_use_fallback")
            return fallback

        timeout = int(spec.context.get("timeout", 120))
        model = str(spec.context.get("model", self.default_model))
        prompt = build_llm_prompt(spec)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if self.key_header.lower() == "authorization":
            headers[self.key_header] = f"{self.key_prefix}{self.api_key}"
        else:
            headers[self.key_header] = self.api_key

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout + 10,
            )
            response.raise_for_status()
            data = response.json()
            text = self._extract_text(data)
            return TaskResult(ok=True, output=normalize_output(spec, text, raw=data))
        except Exception as exc:  # noqa: BLE001
            fallback = self._fallback.execute(spec)
            if not self.silent_fallback:
                fallback.warnings.append(f"openai_compatible_executor_failed:{exc}")
            return fallback

    @staticmethod
    def _extract_text(data: Any) -> str:
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
        return ""


def build_openai_executor() -> OpenAICompatibleExecutor:
    return OpenAICompatibleExecutor(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        default_model=os.getenv("AGENT_PAPER_REVIEWERS_OPENAI_MODEL", "gpt-4.1-mini"),
    )


def build_qwen_executor() -> OpenAICompatibleExecutor:
    return OpenAICompatibleExecutor(
        base_url=os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        api_key=os.getenv("QWEN_API_KEY"),
        default_model=os.getenv("AGENT_PAPER_REVIEWERS_QWEN_MODEL", "qwen-plus"),
    )


def build_local_vllm_executor() -> OpenAICompatibleExecutor:
    return OpenAICompatibleExecutor(
        base_url=os.getenv("LOCAL_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("LOCAL_VLLM_API_KEY", "EMPTY"),
        default_model=os.getenv("AGENT_PAPER_REVIEWERS_VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        silent_fallback=True,
    )


def build_codex_executor() -> OpenAICompatibleExecutor:
    return OpenAICompatibleExecutor(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        default_model=os.getenv("AGENT_PAPER_REVIEWERS_CODEX_MODEL", "gpt-5.4-mini"),
    )
