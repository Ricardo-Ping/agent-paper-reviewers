from __future__ import annotations

from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency fallback
    fuzz = None

from ..models import TaskResult, TaskSpec
from .base import ExecutorAdapter


class DeterministicExecutor(ExecutorAdapter):
    """Offline-safe executor used when no external model backend is configured."""

    def execute(self, spec: TaskSpec) -> TaskResult:
        if spec.task_type == "translate_zh":
            text = spec.context.get("text", "")
            return TaskResult(ok=True, output={"translated_text": self._pseudo_translate(text)})

        if spec.task_type == "summarize":
            text = spec.context.get("text", "")
            summary = text[: min(len(text), 600)]
            return TaskResult(ok=True, output={"summary": summary})

        if spec.task_type == "score_similarity":
            a = spec.context.get("a", "")
            b = spec.context.get("b", "")
            if fuzz is not None:
                score = fuzz.token_set_ratio(a, b) / 100.0
            else:
                score = SequenceMatcher(None, a, b).ratio()
            return TaskResult(ok=True, output={"score": score})

        return TaskResult(ok=True, output={"note": "No-op deterministic result."})

    @staticmethod
    def _pseudo_translate(text: str) -> str:
        glossary = {
            "Novelty": "新颖性",
            "Soundness": "技术正确性",
            "Experiment": "实验充分性",
            "Clarity": "写作清晰度",
            "Rebuttal": "答辩回复",
            "Risk": "风险",
            "Decision": "决策",
            "Not Ready": "不建议投稿",
            "Borderline": "边界状态",
            "Ready": "可投稿",
        }
        translated = text
        for en, zh in glossary.items():
            translated = translated.replace(en, zh)
        return translated
