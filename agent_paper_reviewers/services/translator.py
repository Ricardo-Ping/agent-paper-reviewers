from __future__ import annotations

import re
from typing import Any

try:
    from transformers import MarianMTModel, MarianTokenizer
except Exception:  # pragma: no cover - optional dependency
    MarianMTModel = None
    MarianTokenizer = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from deep_translator import GoogleTranslator
except Exception:  # pragma: no cover - optional dependency
    GoogleTranslator = None

from ..executors.base import ExecutorAdapter
from ..models import TaskSpec


class Translator:
    _model_name = "Helsinki-NLP/opus-mt-en-zh"
    _marian_model: Any = None
    _marian_tokenizer: Any = None
    _marian_load_failed = False

    def __init__(self, executor: ExecutorAdapter) -> None:
        self.executor = executor
        self._cache: dict[str, str] = {}

    def to_zh(self, text: str) -> str:
        if not text:
            return text
        if text in self._cache:
            return self._cache[text]

        translated = self._translate_with_google(text)
        if not translated:
            translated = self._translate_with_marian(text)
        if not translated:
            translated = self._translate_with_executor(text)

        self._cache[text] = translated
        return translated

    def _translate_with_executor(self, text: str) -> str:
        spec = TaskSpec(
            task_type="translate_zh",
            prompt="Translate to Simplified Chinese while preserving IDs and numeric values.",
            context={"text": text},
            output_schema={"translated_text": "string"},
            model_profile="translate",
        )
        result = self.executor.execute(spec)
        if result.ok and result.output.get("translated_text"):
            return str(result.output["translated_text"])
        return text

    def _translate_with_marian(self, text: str) -> str | None:
        if MarianMTModel is None or MarianTokenizer is None or torch is None:
            return None
        if self.__class__._marian_load_failed:
            return None

        if self.__class__._marian_model is None or self.__class__._marian_tokenizer is None:
            try:
                self.__class__._marian_tokenizer = MarianTokenizer.from_pretrained(
                    self._model_name
                )
                self.__class__._marian_model = MarianMTModel.from_pretrained(self._model_name)
                self.__class__._marian_model.eval()
            except Exception:
                self.__class__._marian_load_failed = True
                return None

        tokenizer = self.__class__._marian_tokenizer
        model = self.__class__._marian_model

        lines = text.splitlines()
        out: list[str] = []

        for line in lines:
            if not line.strip():
                out.append(line)
                continue
            if line.strip().startswith("```") or line.strip().startswith("`"):
                out.append(line)
                continue

            prefix_match = re.match(r"^(\s*(?:#+\s+|- |\d+\.\s+)?)(.*)$", line)
            if not prefix_match:
                out.append(line)
                continue

            prefix, content = prefix_match.groups()
            translated_content = self._translate_chunk_marian(tokenizer, model, content)
            out.append(prefix + translated_content)

        return "\n".join(out)

    @staticmethod
    def _translate_chunk_marian(tokenizer: Any, model: Any, text: str, max_len: int = 350) -> str:
        if not text.strip():
            return text

        placeholders: dict[str, str] = {}

        def hold(match: re.Match[str]) -> str:
            key = f"__PH{len(placeholders)}__"
            placeholders[key] = match.group(0)
            return key

        protected = re.sub(r"\b(?:RISK|EXP)-?\d+\b", hold, text)
        protected = re.sub(
            r"\b(?:NeurIPS|ICLR|ICML|ACL|ARR|EMNLP|KDD|AAAI|CVPR|ECCV)\b",
            hold,
            protected,
        )

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for token in protected.split(" "):
            token_len = len(token) + 1
            if current_len + token_len > max_len and current:
                chunks.append(" ".join(current))
                current = [token]
                current_len = token_len
            else:
                current.append(token)
                current_len += token_len
        if current:
            chunks.append(" ".join(current))

        translated_chunks: list[str] = []
        for chunk in chunks:
            try:
                encoded = tokenizer([chunk], return_tensors="pt", truncation=True)
                with torch.no_grad():
                    generated = model.generate(**encoded, max_length=256)
                translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            except Exception:
                translated = chunk
            translated_chunks.append(translated)

        translated_text = " ".join(translated_chunks)
        for idx, (key, value) in enumerate(placeholders.items()):
            translated_text = translated_text.replace(key, value)
            translated_text = re.sub(rf"[A-Z_]*PH{idx}[A-Z_]*", value, translated_text)
        return translated_text

    def _translate_with_google(self, text: str) -> str | None:
        if GoogleTranslator is None:
            return None

        translator = GoogleTranslator(source="auto", target="zh-CN")
        lines = text.splitlines()
        out: list[str] = []

        for line in lines:
            if not line.strip():
                out.append(line)
                continue
            if line.strip().startswith("```") or line.strip().startswith("`"):
                out.append(line)
                continue

            prefix_match = re.match(r"^(\s*(?:#+\s+|- |\d+\.\s+)?)(.*)$", line)
            if not prefix_match:
                out.append(line)
                continue

            prefix, content = prefix_match.groups()
            out.append(prefix + self._translate_chunk_google(translator, content))

        return "\n".join(out)

    @staticmethod
    def _translate_chunk_google(translator: GoogleTranslator, text: str, max_len: int = 1800) -> str:
        if not text.strip():
            return text

        placeholders: dict[str, str] = {}

        def hold(match: re.Match[str]) -> str:
            key = f"__PH{len(placeholders)}__"
            placeholders[key] = match.group(0)
            return key

        protected = re.sub(r"\b(?:RISK|EXP)-?\d+\b", hold, text)
        protected = re.sub(
            r"\b(?:NeurIPS|ICLR|ICML|ACL|ARR|EMNLP|KDD|AAAI|CVPR|ECCV)\b",
            hold,
            protected,
        )

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for token in protected.split(" "):
            token_len = len(token) + 1
            if current_len + token_len > max_len and current:
                chunks.append(" ".join(current))
                current = [token]
                current_len = token_len
            else:
                current.append(token)
                current_len += token_len
        if current:
            chunks.append(" ".join(current))

        translated_chunks: list[str] = []
        for chunk in chunks:
            try:
                translated_chunks.append(translator.translate(chunk))
            except Exception:
                translated_chunks.append(chunk)

        translated = " ".join(translated_chunks)
        for idx, (key, value) in enumerate(placeholders.items()):
            translated = translated.replace(key, value)
            translated = re.sub(rf"[A-Z_]*PH{idx}[A-Z_]*", value, translated)
        return translated
