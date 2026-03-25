from __future__ import annotations

import os
from typing import Any

import httpx

from ..models import TaskResult, TaskSpec
from .base import ExecutorAdapter
from .deterministic import DeterministicExecutor
from .utils import build_llm_prompt, normalize_output


class OpenClawNodeExecutor(ExecutorAdapter):
    """Execute tasks through OpenClaw sessions spawn API."""

    def __init__(
        self,
        base_url: str | None = None,
        runtime: str = "agent_api",
        default_model: str = "minimax-m2.7",
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("AGENT_PAPER_REVIEWERS_OPENCLAW_URL")
            or "http://localhost:18789"
        ).rstrip("/")
        self.runtime = runtime
        self.default_model = os.getenv(
            "AGENT_PAPER_REVIEWERS_OPENCLAW_MODEL", default_model
        )
        self._fallback = DeterministicExecutor()

    def execute(self, spec: TaskSpec) -> TaskResult:
        timeout = int(spec.context.get("timeout", 120))
        payload = {
            "runtime": self.runtime,
            "task": build_llm_prompt(spec),
            "model": spec.context.get("model", self.default_model),
            "runTimeoutSeconds": timeout,
        }

        try:
            resp = httpx.post(
                f"{self.base_url}/api/sessions/spawn",
                json=payload,
                timeout=timeout + 10,
            )
            resp.raise_for_status()
            result = resp.json()
            text = self._extract_text(result)
            output = normalize_output(spec, text, raw=result)
            return TaskResult(ok=True, output=output)
        except Exception as exc:  # noqa: BLE001
            fallback = self._fallback.execute(spec)
            fallback.warnings.append(f"openclaw_executor_failed:{exc}")
            return fallback

    @staticmethod
    def _extract_text(data: Any) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in [
                "text",
                "content",
                "response",
                "output_text",
                "result",
                "message",
            ]:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            if isinstance(data.get("output"), list):
                for item in data["output"]:
                    text = OpenClawNodeExecutor._extract_text(item)
                    if text.strip():
                        return text
        if isinstance(data, list):
            for item in data:
                text = OpenClawNodeExecutor._extract_text(item)
                if text.strip():
                    return text
        return ""

