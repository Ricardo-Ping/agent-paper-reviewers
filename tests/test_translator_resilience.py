from __future__ import annotations

from agent_paper_reviewers.services import translator as translator_module
from agent_paper_reviewers.services.translator import Translator


class _FailExecutor:
    def execute(self, spec):  # noqa: ANN001
        raise RuntimeError("executor unavailable")


def test_translator_never_raises_when_backends_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(translator_module, "GoogleTranslator", None)
    monkeypatch.setattr(translator_module, "MarianMTModel", None)
    monkeypatch.setattr(translator_module, "MarianTokenizer", None)
    monkeypatch.setattr(translator_module, "torch", None)

    tr = Translator(_FailExecutor())
    out = tr.to_zh("Novelty, Soundness, and statistical significance evidence appears missing.")

    assert isinstance(out, str)
    assert out.strip()
    assert "新颖性" in out
