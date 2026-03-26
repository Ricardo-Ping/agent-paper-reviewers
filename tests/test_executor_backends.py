from __future__ import annotations

from agent_paper_reviewers.executors.anthropic_executor import AnthropicExecutor
from agent_paper_reviewers.executors.deterministic import DeterministicExecutor
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


def test_deterministic_executor_supports_reviewer_and_paper_qa_tasks() -> None:
    executor = DeterministicExecutor()
    q_result = executor.execute(
        TaskSpec(
            task_type="reviewer_question_simulation",
            prompt="simulate",
            context={"gaps": [{"code": "missing_significance"}], "top_risks": [{"id": "RISK-001"}]},
            output_schema={"questions": []},
            model_profile="judge",
        )
    )
    assert q_result.ok
    assert isinstance(q_result.output.get("questions"), list)
    assert q_result.output["questions"]

    qa_result = executor.execute(
        TaskSpec(
            task_type="paper_qa_self_review",
            prompt="qa",
            context={
                "rebuttal_bundle_en": {
                    "items": [
                        {
                            "review_id": "R1",
                            "concern": "Need significance tests",
                            "response": "Thanks.",
                            "new_evidence": [],
                            "paper_change": "Update.",
                        }
                    ]
                }
            },
            output_schema={"accept": True},
            model_profile="judge",
        )
    )
    assert qa_result.ok
    assert "accept" in qa_result.output
    assert "per_item" in qa_result.output
