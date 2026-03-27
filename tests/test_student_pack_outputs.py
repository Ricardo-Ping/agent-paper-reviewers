from __future__ import annotations

from pathlib import Path

import pytest

from agent_paper_reviewers.models import ExecutorBackend, ReviewRunInput
from agent_paper_reviewers.orchestrator import ReviewOrchestrator


def _build_input(tmp_path: Path, language_mode: str) -> ReviewRunInput:
    paper = tmp_path / "paper.md"
    paper.write_text(
        "# Title\n\n## Abstract\nWe improve SQL translation.\n\n## Method\nMethod details.\n\n## Experiments\nResults are shown in Table 1.\n",
        encoding="utf-8",
    )
    return ReviewRunInput.model_validate(
        {
            "paper": {"format": "md", "path": str(paper)},
            "venue": {"name": "ICLR", "year": 2026},
            "claims": ["Our method improves SQL dialect translation."],
            "options": {
                "language_mode": language_mode,
                "executor_backend": "local_vllm",
                "always_export_pdf": False,
            },
        }
    )


def test_student_pack_en_outputs_exist(tmp_path: Path) -> None:
    data = _build_input(tmp_path, "en")
    orch = ReviewOrchestrator(Path(__file__).resolve().parents[1])
    summary = orch.run(data, tmp_path / "runs")

    run_dir = Path(summary.output_dir)
    assert (run_dir / "START_HERE.md").exists()
    assert (run_dir / "START_HERE.en.md").exists()
    assert (run_dir / "RUN_GUIDE.md").exists()
    assert (run_dir / "RUN_GUIDE.en.md").exists()
    assert (run_dir / "STUDENT_BRIEF.md").exists()
    assert (run_dir / "STUDENT_BRIEF.en.md").exists()
    assert (run_dir / "PERSONA_PLAYBOOK.md").exists()
    assert (run_dir / "PERSONA_PLAYBOOK.en.md").exists()
    assert (run_dir / "AGENT_HANDOFF.json").exists()
    assert (run_dir / "student_pack" / "en" / "001-submission-decision.md").exists()
    assert (run_dir / "student_pack" / "en" / "002-action-items.md").exists()
    assert (run_dir / "student_pack" / "en" / "003-rebuttal-draft.md").exists()


def test_student_pack_zh_outputs_exist_when_bilingual(tmp_path: Path) -> None:
    data = _build_input(tmp_path, "en_zh")
    orch = ReviewOrchestrator(Path(__file__).resolve().parents[1])
    summary = orch.run(data, tmp_path / "runs")

    run_dir = Path(summary.output_dir)
    assert (run_dir / "START_HERE.md").exists()
    assert (run_dir / "START_HERE.zh.md").exists()
    assert (run_dir / "RUN_GUIDE.md").exists()
    assert (run_dir / "RUN_GUIDE.zh.md").exists()
    assert (run_dir / "STUDENT_BRIEF.md").exists()
    assert (run_dir / "STUDENT_BRIEF.zh.md").exists()
    assert (run_dir / "PERSONA_PLAYBOOK.md").exists()
    assert (run_dir / "PERSONA_PLAYBOOK.zh.md").exists()
    assert (run_dir / "AGENT_HANDOFF.json").exists()
    assert (run_dir / "student_pack" / "zh" / "001-submission-decision.md").exists()
    assert (run_dir / "student_pack" / "zh" / "002-action-items.md").exists()
    assert (run_dir / "student_pack" / "zh" / "003-rebuttal-draft.md").exists()
    assert "从这里开始" in (run_dir / "START_HERE.zh.md").read_text(encoding="utf-8-sig")


def test_student_pack_blocks_fallback_when_real_agent_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_PAPER_REVIEWERS_CODEX_API_KEY", raising=False)
    data = _build_input(tmp_path, "en")
    data.options.executor_backend = ExecutorBackend.OPENAI
    orch = ReviewOrchestrator(Path(__file__).resolve().parents[1])
    with pytest.raises(RuntimeError) as excinfo:
        orch.run(data, tmp_path / "runs")
    text = str(excinfo.value)
    assert "executor backend validation failed" in text
    assert "`openai`" in text

