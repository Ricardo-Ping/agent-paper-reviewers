from __future__ import annotations

import os
from typing import Any

import httpx

from ..models import TaskResult, TaskSpec
from .base import ExecutorAdapter
from .deterministic import DeterministicExecutor
from .utils import build_llm_prompt, normalize_output


class AnthropicExecutor(ExecutorAdapter):
    MODEL_PROFILE_MAP = {
        "judge": "claude-3-5-sonnet-20241022",
        "extract": "claude-3-5-sonnet-20241022",
        "generate": "claude-3-5-sonnet-20241022",
        "translate": "claude-3-5-sonnet-20241022",
    }

    def __init__(self) -> None:
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip(
            "/"
        )
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.default_model = os.getenv(
            "AGENT_PAPER_REVIEWERS_ANTHROPIC_MODEL",
            "claude-3-5-sonnet-20241022",
        )
        self._fallback = DeterministicExecutor()

    def execute(self, spec: TaskSpec) -> TaskResult:
        strict_real_llm = bool(spec.context.get("require_real_llm", False))
        if not self.api_key:
            if strict_real_llm:
                return TaskResult(
                    ok=False,
                    warnings=["anthropic_api_key_missing_real_llm_required"],
                )
            fallback = self._fallback.execute(spec)
            fallback.warnings.append("anthropic_api_key_missing_use_fallback")
            return fallback

        timeout = int(spec.context.get("timeout", 120))
        prompt = build_llm_prompt(spec)
        model = self._resolve_model(spec)
        payload = {
            "model": model,
            "max_tokens": int(spec.context.get("max_tokens", 1200)),
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        try:
            response = httpx.post(
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=payload,
                timeout=timeout + 10,
            )
            response.raise_for_status()
            data = response.json()
            text = self._extract_text(data)
            return TaskResult(ok=True, output=normalize_output(spec, text, raw=data))
        except Exception as exc:  # noqa: BLE001
            if strict_real_llm:
                return TaskResult(
                    ok=False,
                    warnings=[
                        f"anthropic_executor_failed:{exc}",
                        "real_llm_required_no_fallback",
                    ],
                )
            fallback = self._fallback.execute(spec)
            fallback.warnings.append(f"anthropic_executor_failed:{exc}")
            return fallback

    def _resolve_model(self, spec: TaskSpec) -> str:
        explicit_model = str(spec.context.get("model", "")).strip()
        if explicit_model:
            return explicit_model
        model_profile = str(spec.model_profile or "").strip().lower()
        if model_profile:
            env_key = f"AGENT_PAPER_REVIEWERS_MODEL_PROFILE_{model_profile.upper()}"
            env_value = str(os.getenv(env_key, "")).strip()
            if env_value:
                return env_value
            mapped = str(self.MODEL_PROFILE_MAP.get(model_profile, "")).strip()
            if mapped:
                return mapped
        return self.default_model

    @staticmethod
    def _extract_text(data: Any) -> str:
        if not isinstance(data, dict):
            return ""
        content = data.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        return text
        return ""
