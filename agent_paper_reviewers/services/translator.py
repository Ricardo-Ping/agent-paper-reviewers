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

    _stable_glossary = {
        "Novelty": "新颖性",
        "Soundness": "技术正确性",
        "Experiment": "实验充分性",
        "Clarity": "写作清晰度",
        "Decision": "决策",
        "Risk": "风险",
        "Rebuttal": "答辩回复",
        "Not Ready": "不建议投稿",
        "Borderline": "边界状态",
        "Ready": "可投稿",
        "Statistical significance evidence appears missing.": "统计显著性证据可能缺失。",
        "Reproducibility details are likely incomplete.": "可复现性细节可能不完整。",
        "Experimental evidence does not yet meet venue expectations.": "实验性证据尚未达到目标会议预期。",
        "Core claims are not sufficiently supported by rigorous evidence.": "核心主张尚未得到严格证据的充分支撑。",
        "Address this with a focused experiment or analysis update.": "通过补充针对性实验或分析更新来解决该问题。",
        "Add direct experiments and statistical validation tied to this claim.": "补充与该主张直接对应的实验和统计验证。",
        "Specifically, we will address this concern with direct evidence, stronger analysis, and explicit paper updates in revision.": "具体来说，我们将用直接证据、更强分析以及明确的论文修订来回应这条意见。",
        "Citation coverage appears shallow; references may be insufficient to position novelty and baselines.": "引用覆盖度偏浅，参考文献可能不足以支撑新颖性与基线定位。",
        "Related work appears to under-cover recent top-venue papers relevant to this topic.": "相关工作对近期顶会论文覆盖不足，可能影响定位说服力。",
        "One or more key claims have weak evidence alignment and need direct support.": "一项或多项关键主张的证据对齐偏弱，需要直接支撑。",
        "Reviewers typically reduce confidence when claim, evidence, and reporting are not tightly linked.": "当主张、证据与报告链条不够紧密时，审稿人通常会下调置信度。",
        "Add one targeted experiment, one statistical/significance validation block, and explicit section-level paper changes for this concern.": "针对该问题补充一组定向实验、一组统计显著性验证，并在论文中明确标注对应修改位置。",
        "Planned experiment:": "计划实验：",
        "Protocol update:": "方案更新：",
    }

    def __init__(self, executor: ExecutorAdapter) -> None:
        self.executor = executor
        self._cache: dict[str, str] = {}

    def to_zh(self, text: str) -> str:
        if not text:
            return text
        if text in self._cache:
            return self._cache[text]

        # 1) deterministic glossary first, for key report/rebuttal phrases.
        by_glossary = self._translate_with_glossary(text)
        if self._looks_translated(by_glossary, text):
            self._cache[text] = by_glossary
            return by_glossary

        translated: str | None = None

        # 2) online translator route (typically more fluent than local tiny MT).
        try:
            translated = self._translate_with_google(text)
        except Exception:  # noqa: BLE001
            translated = None

        # 3) local model route (no external network once model cached).
        if not translated:
            try:
                translated = self._translate_with_marian(text)
            except Exception:  # noqa: BLE001
                translated = None

        # 4) executor route.
        if not translated:
            try:
                translated = self._translate_with_executor(text)
            except Exception:  # noqa: BLE001
                translated = None

        # 5) final stable fallback: glossary/pattern conversion (never raise).
        if not translated or not self._looks_translated(translated, text):
            translated = self._stable_fallback(text)

        self._cache[text] = translated
        return translated

    def _translate_with_executor(self, text: str) -> str | None:
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
        return None

    def _translate_with_marian(self, text: str) -> str | None:
        if MarianMTModel is None or MarianTokenizer is None or torch is None:
            return None
        if self.__class__._marian_load_failed:
            return None

        if self.__class__._marian_model is None or self.__class__._marian_tokenizer is None:
            try:
                self.__class__._marian_tokenizer = MarianTokenizer.from_pretrained(self._model_name)
                self.__class__._marian_model = MarianMTModel.from_pretrained(self._model_name)
                self.__class__._marian_model.eval()
            except Exception:  # noqa: BLE001
                self.__class__._marian_load_failed = True
                return None

        tokenizer = self.__class__._marian_tokenizer
        model = self.__class__._marian_model

        lines = text.splitlines()
        out: list[str] = []
        for line in lines:
            if not line.strip() or line.strip().startswith("```") or line.strip().startswith("`"):
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
        protected = re.sub(r"\b(?:NeurIPS|ICLR|ICML|ACL|ARR|EMNLP|KDD|AAAI|CVPR|ECCV|SIGMOD|VLDB|ICDE)\b", hold, protected)

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
            except Exception:  # noqa: BLE001
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
        try:
            translator = GoogleTranslator(source="auto", target="zh-CN")
        except Exception:  # noqa: BLE001
            return None

        lines = text.splitlines()
        out: list[str] = []
        for line in lines:
            if not line.strip() or line.strip().startswith("```") or line.strip().startswith("`"):
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
        protected = re.sub(r"\b(?:NeurIPS|ICLR|ICML|ACL|ARR|EMNLP|KDD|AAAI|CVPR|ECCV|SIGMOD|VLDB|ICDE)\b", hold, protected)

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
            except Exception:  # noqa: BLE001
                translated_chunks.append(chunk)

        translated = " ".join(translated_chunks)
        for idx, (key, value) in enumerate(placeholders.items()):
            translated = translated.replace(key, value)
            translated = re.sub(rf"[A-Z_]*PH{idx}[A-Z_]*", value, translated)
        return translated

    def _translate_with_glossary(self, text: str) -> str:
        out = text
        for en, zh in self._stable_glossary.items():
            out = out.replace(en, zh)
        return out

    def _stable_fallback(self, text: str) -> str:
        out = self._translate_with_glossary(text)
        out = re.sub(
            r"^Claim (C\d+) has Weak evidence support\.?$",
            lambda m: f"主张 {m.group(1)} 的证据支持较弱。",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            r"^Mitigate (RISK-\d+) - (P\d) risk$",
            lambda m: f"缓解 {m.group(1)} - {m.group(2)} 风险",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            r"^Planned experiment:\s*(.+?)\.\s*$",
            lambda m: f"计划实验：{m.group(1).strip()}。",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            r"^Protocol update:\s*(.+?)\s*$",
            lambda m: f"方案更新：{m.group(1).strip()}",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            r"^The current draft likely lacks a direct experiment-or-analysis block mapping one-to-one to this concern:\s*(.+)$",
            lambda m: "当前稿件很可能缺少与该问题一一对应的实验或分析支撑："
            + m.group(1).strip(),
            out,
            flags=re.IGNORECASE,
        )
        return out

    @staticmethod
    def _looks_translated(candidate: str | None, src: str) -> bool:
        if not candidate or not candidate.strip():
            return False
        if candidate == src:
            return False
        cjk_count = sum(1 for ch in candidate if "\u4e00" <= ch <= "\u9fff")
        ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in candidate)
        total_letters = sum(ch.isalpha() for ch in candidate)
        total_chars = len(candidate)
        if total_chars == 0 or total_letters == 0:
            return False
        cjk_ratio = cjk_count / max(1, total_chars)
        ascii_ratio = ascii_letters / max(1, total_letters)
        if cjk_ratio >= 0.18 and ascii_ratio <= 0.7:
            return True
        if total_chars <= 24 and cjk_count >= 2:
            return True
        return False
