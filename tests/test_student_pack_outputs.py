from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput
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
                "mcp_backend": "disabled",
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
    assert (run_dir / "student_pack" / "zh" / "001-submission-decision.md").exists()
    assert (run_dir / "student_pack" / "zh" / "002-action-items.md").exists()
    assert (run_dir / "student_pack" / "zh" / "003-rebuttal-draft.md").exists()
