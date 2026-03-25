from __future__ import annotations

from agent_paper_reviewers.executors.anthropic_executor import AnthropicExecutor
from agent_paper_reviewers.executors.factory import get_executor
from agent_paper_reviewers.executors.openai_compatible_executor import OpenAICompatibleExecutor
from agent_paper_reviewers.executors.openclawnode_executor import OpenClawNodeExecutor
from agent_paper_reviewers.models import ExecutorBackend, TaskSpec


def test_factory_returns_real_executor_types() -> None:
    assert isinstance(get_executor(ExecutorBackend.AGENT_API), OpenClawNodeExecutor)
    assert isinstance(get_executor(ExecutorBackend.OPENAI), OpenAICompatibleExecutor)
    assert isinstance(get_executor(ExecutorBackend.QWEN), OpenAICompatibleExecutor)
    assert isinstance(get_executor(ExecutorBackend.LOCAL_VLLM), OpenAICompatibleExecutor)
    assert isinstance(get_executor(ExecutorBackend.CODEX), OpenAICompatibleExecutor)
    assert isinstance(get_executor(ExecutorBackend.ANTHROPIC), AnthropicExecutor)


def test_remote_executor_falls_back_when_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    executor = get_executor(ExecutorBackend.OPENAI)
    spec = TaskSpec(
        task_type="translate_zh",
        prompt="Translate",
        context={"text": "Novelty"},
        output_schema={"translated_text": "string"},
        model_profile="translate",
    )
    result = executor.execute(spec)
    assert result.ok
    assert "translated_text" in result.output
    assert any("api_key_missing" in w for w in result.warnings)

