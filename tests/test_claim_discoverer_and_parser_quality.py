from __future__ import annotations

from pathlib import Path

from agent_paper_reviewers.models import ReviewRunInput
from agent_paper_reviewers.pipeline.base import PipelineContext
from agent_paper_reviewers.pipeline.step_claim_discoverer import ClaimDiscovererStep
from agent_paper_reviewers.pipeline.step_paper_parser import PaperParserStep


def _ctx(tmp_path: Path, *, claims: list[str], paper_format: str = "md") -> PipelineContext:
    paper = tmp_path / ("paper.pdf" if paper_format == "pdf" else "paper.md")
    if paper_format == "pdf":
        paper.write_bytes(b"dummy")
    else:
        paper.write_text("# Title\n\n## Abstract\nA.\n", encoding="utf-8")

    data = ReviewRunInput.model_validate(
        {
            "paper": {"format": paper_format, "path": str(paper)},
            "venue": {"name": "ICLR", "year": 2026},
            "claims": claims,
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return PipelineContext(run_id="r1", run_dir=run_dir, input_data=data)


def test_claim_discoverer_auto_selects_when_user_claims_empty(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, claims=[])
    ctx.artifacts["paper_structured"] = {
        "title": "Demo Paper",
        "sections": [
            {
                "name": "abstract",
                "text": (
                    "We propose a new method for SQL translation. "
                    "Our method improves execution success rate by 18%."
                ),
            },
            {
                "name": "conclusion",
                "text": "We achieve strong robustness across multiple dialect pairs.",
            },
        ],
        "raw_text": "",
    }

    ClaimDiscovererStep().run(ctx)
    payload = ctx.artifacts["claim_discovery"]
    assert payload["selected_claims"]
    assert payload["confirmation"]["required"] is True
    assert any("propose" in c.lower() or "improve" in c.lower() for c in payload["selected_claims"])


def test_claim_discoverer_keeps_user_claims_and_provides_suggestions(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, claims=["We outperform strong baselines on SQL translation."])
    ctx.artifacts["paper_structured"] = {
        "title": "Demo Paper",
        "sections": [
            {
                "name": "abstract",
                "text": (
                    "We propose a new method for SQL translation. "
                    "Our method improves execution success rate by 18%."
                ),
            }
        ],
        "raw_text": "",
    }

    ClaimDiscovererStep().run(ctx)
    payload = ctx.artifacts["claim_discovery"]
    assert payload["selected_claims"][0].startswith("We outperform strong baselines")
    assert payload["suggested_candidates"]
    assert payload["confirmation"]["required"] is True


def test_paper_parser_adds_pdf_quality_warnings(tmp_path: Path, monkeypatch) -> None:
    ctx = _ctx(tmp_path, claims=[], paper_format="pdf")

    def fake_parse_pdf(_path: Path) -> dict:
        return {
            "title": "Bad Parse",
            "sections": [{"name": "body", "text": "tiny"}],
            "raw_text": "tiny �������� noisy parse",
            "pages": [{"page": 1, "text": "tiny"}],
            "parse_backend": "pypdf",
        }

    monkeypatch.setattr("agent_paper_reviewers.pipeline.step_paper_parser.parse_pdf", fake_parse_pdf)

    PaperParserStep().run(ctx)

    assert any("pdf_parse_quality_low_word_count" in x for x in ctx.qa_issues)
    assert any("pdf_parse_quality_low_section_count" in x for x in ctx.qa_issues)
    assert any("pdf_parse_quality_encoding_noise_detected" in x for x in ctx.qa_issues)


def test_claim_discoverer_emits_input_guidance_for_missing_categories(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, claims=["We propose a SQL translation framework with a new architecture."])
    ctx.artifacts["paper_structured"] = {
        "title": "Demo Paper",
        "sections": [
            {"name": "abstract", "text": "We propose a new framework and improve robustness."},
            {"name": "conclusion", "text": "Future work is discussed."},
        ],
        "raw_text": "",
    }

    ClaimDiscovererStep().run(ctx)
    payload = ctx.artifacts["claim_discovery"]
    guidance = payload.get("input_guidance", {})

    assert isinstance(guidance, dict)
    missing = guidance.get("selected_claim_missing_categories", [])
    assert "baseline" in missing
    assert "statistical" in missing
    assert any("claim_discovery_guidance:missing_claim_categories" in x for x in ctx.qa_issues)
