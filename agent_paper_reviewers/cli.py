from __future__ import annotations

from dataclasses import asdict
import json
import os
import re
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .executors.factory import validate_executor_readiness
from .models import ExecutorBackend, ReviewRunInput
from .services.feedback_store import submit_feedback
from .services.paper_parser import parse_markdown, parse_pdf
from .services.pdf_export import detect_pdf_export_capability
from .services.toolkit_formatter import (
    render_student_pack_markdown,
    student_pack_analysis_template,
)
from .services.venue_loader import load_venue_profile, normalize_venue_slug
from .services.venue_sync import refresh_venue_rules

app = typer.Typer(help="agent-paper-reviewers pipeline CLI")
console = Console()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_executor_or_die(review_input: ReviewRunInput) -> None:
    backend = review_input.options.executor_backend
    ok, reason = validate_executor_readiness(backend)
    if ok:
        return
    backend_name = backend.value
    raise typer.BadParameter(
        "FATAL: No real model backend is ready for "
        f"`{backend_name}`. {reason} "
        "Set required API key (or use `agent_api`/`local_vllm` with a live endpoint)."
    )


def _resolve_paper_path(input_path: Path, raw: dict) -> dict:
    paper = raw.get("paper", {})
    paper_path = paper.get("path")
    if isinstance(paper_path, str) and paper_path.strip():
        path_obj = Path(paper_path)
        if not path_obj.is_absolute():
            raw["paper"]["path"] = str((input_path.parent / path_obj).resolve())
    return raw


def _paper_format_from_path(path: Path) -> str:
    suffix = path.suffix.lower().strip()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".md", ".markdown"}:
        return "md"
    raise typer.BadParameter(
        "Unable to infer paper format from extension. "
        "Use --paper-format md|pdf explicitly."
    )


def _attach_section_ids(raw_sections: object) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw_sections, list):
        return out
    for idx, sec in enumerate(raw_sections, start=1):
        if not isinstance(sec, dict):
            continue
        item = dict(sec)
        section_id = str(item.get("section_id") or "").strip() or f"S{idx:03d}"
        item["section_id"] = section_id
        item["section_index"] = idx
        item["name"] = str(item.get("name") or f"section_{idx}").strip().lower()
        item["text"] = str(item.get("text") or "")
        out.append(item)
    return out


def _build_section_locator(sections: list[dict]) -> list[dict]:
    locator: list[dict] = []
    for sec in sections:
        section_id = str(sec.get("section_id") or "").strip()
        if not section_id:
            continue
        text = str(sec.get("text") or "").strip()
        locator.append(
            {
                "section_id": section_id,
                "section_index": int(sec.get("section_index") or 0),
                "name": str(sec.get("name") or ""),
                "preview": text[:180],
            }
        )
    return locator


def _pdf_parse_quality_warnings(
    *,
    raw_text: str,
    word_count: int,
    section_count: int,
    parse_backend: str,
) -> list[str]:
    warnings: list[str] = []
    if word_count < 350:
        warnings.append(
            "pdf_parse_quality_low_word_count:"
            f"{word_count}:verify_text_layer_or_use_ocr_before_submission_review"
        )
    if section_count < 3:
        warnings.append(
            "pdf_parse_quality_low_section_count:"
            f"{section_count}:verify_pdf_structure_or_convert_to_clean_markdown"
        )

    mojibake_patterns = [
        r"\uFFFD",
        r"(?:Ã.|Â.){2,}",
        r"[锟閳楗椤绱閿]",
    ]
    noise_hits = sum(len(re.findall(pattern, raw_text)) for pattern in mojibake_patterns)
    if noise_hits >= 6:
        warnings.append(
            "pdf_parse_quality_encoding_noise_detected:"
            f"{noise_hits}:{parse_backend}:verify_pdf_encoding_or_use_ocr_export"
        )
    return warnings


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def _build_ai_summary_payload(summary, run_dir: Path) -> dict:
    decision = _safe_read_json(run_dir / "decision_brief.en.json")
    risk = _safe_read_json(run_dir / "artifacts" / "risk_ranking.json")
    rebuttal_path = run_dir / "rebuttal.en.md"

    top_risks: list[dict] = []
    must_fix: list[str] = []
    risks = risk.get("risks", [])
    if isinstance(risks, list):
        for row in risks[:5]:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id", "")).strip() or "RISK-?"
            severity = str(row.get("severity", "P2")).strip().upper()
            score = float(row.get("score", 0.0) or 0.0)
            blocking = severity in {"P0", "P1"}
            top_risks.append(
                {
                    "id": rid,
                    "severity": severity,
                    "score": round(score, 3),
                    "concern": str(row.get("reason", "")).strip(),
                    "blocking": blocking,
                }
            )
        must_fix = [x["id"] for x in top_risks if x["blocking"]]

    decision_text = str(decision.get("decision", summary.status.value)).strip() or summary.status.value
    interp = str(decision.get("decision_interpretation", "")).strip()
    verdict = f"{decision_text}: {interp}" if interp else decision_text

    fallback_count = 0
    if isinstance(risks, list):
        for row in risks:
            if not isinstance(row, dict):
                continue
            source = str(row.get("generation_source", "")).lower()
            if "fallback" in source:
                fallback_count += 1

    qa_penalty = 0.0
    for q in summary.qa_issues:
        ql = str(q).lower()
        if "pdf_parse_quality" in ql:
            qa_penalty += 0.08
        elif "warning" in ql:
            qa_penalty += 0.02
        elif "error" in ql:
            qa_penalty += 0.04

    confidence = 0.92 - (0.06 * fallback_count) - qa_penalty
    confidence = max(0.2, min(0.95, round(confidence, 3)))

    return {
        "run_id": summary.run_id,
        "status": summary.status.value,
        "verdict": verdict,
        "top_risks": top_risks,
        "must_fix_before_submit": must_fix,
        "rebuttal_ready": str(rebuttal_path) if rebuttal_path.exists() else None,
        "confidence": confidence,
        "qa_issue_count": len(summary.qa_issues),
    }


@app.command("doctor")
def doctor(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON for automation checks.",
    ),
    backend: ExecutorBackend = typer.Option(
        ExecutorBackend.CODEX,
        "--backend",
        help="Executor backend to validate for production runs.",
    ),
) -> None:
    """Check runtime dependencies and external tools."""
    checks = {
        "python": shutil.which("python") is not None,
        "pandoc (optional: PDF export)": shutil.which("pandoc") is not None,
        "xelatex (optional: PDF export)": shutil.which("xelatex") is not None,
        "lualatex (optional: PDF export)": shutil.which("lualatex") is not None,
        "tectonic (optional: PDF export)": shutil.which("tectonic") is not None,
        "conda": shutil.which("conda") is not None,
        "OPENAI_API_KEY (for codex/openai backends)": bool(os.getenv("OPENAI_API_KEY")),
        "AGENT_PAPER_REVIEWERS_CODEX_API_KEY (optional codex key)": bool(os.getenv("AGENT_PAPER_REVIEWERS_CODEX_API_KEY")),
        "ANTHROPIC_API_KEY (for anthropic backend)": bool(os.getenv("ANTHROPIC_API_KEY")),
        "QWEN_API_KEY (for qwen backend)": bool(os.getenv("QWEN_API_KEY")),
    }
    executor_ok, executor_reason = validate_executor_readiness(backend)
    llm_step_checks = [
        {
            "step": "GapDetector",
            "ok": executor_ok,
            "detail": executor_reason if executor_ok else "DeterministicExecutor active (no real model route).",
        },
        {
            "step": "RiskRanker",
            "ok": executor_ok,
            "detail": executor_reason if executor_ok else "DeterministicExecutor active (no real model route).",
        },
        {
            "step": "RebuttalComposer",
            "ok": executor_ok,
            "detail": executor_reason if executor_ok else "DeterministicExecutor active (no real model route).",
        },
    ]

    pdf = detect_pdf_export_capability()
    pdf_status = {
        "ready": pdf.ready,
        "detail": (
            f"ready (pandoc + {pdf.preferred_engine})"
            if pdf.ready
            else (
                "missing pandoc"
                if not pdf.pandoc_available
                else "no LaTeX engine (need xelatex/lualatex/tectonic)"
            )
        ),
        "toolchain": asdict(pdf),
    }

    if as_json:
        console.print_json(
            data={
                "checks": checks,
                "pdf_export": pdf_status,
                "executor_backend": backend.value,
                "executor_ready": executor_ok,
                "executor_detail": executor_reason,
                "llm_pipeline_steps": llm_step_checks,
            }
        )
        return

    table = Table(title="agent-paper-reviewers doctor")
    table.add_column("Dependency")
    table.add_column("Available")
    table.add_column("Details")
    for dep, ok in checks.items():
        table.add_row(dep, "yes" if ok else "no", "-")
    table.add_row(
        f"executor backend readiness ({backend.value})",
        "yes" if executor_ok else "no",
        executor_reason,
    )
    table.add_row(
        "pdf export capability (optional)",
        "yes" if pdf.ready else "no",
        str(pdf_status["detail"]),
    )
    console.print(table)
    console.print("Checking executors...")
    for row in llm_step_checks:
        prefix = "[OK]" if row["ok"] else "[FAIL]"
        console.print(f"  {prefix} {row['step']}: {row['detail']}")
    if not pdf.ready:
        console.print(
            "hint: install PDF toolchain with "
            "`conda install -c conda-forge pandoc tectonic` "
            "(or install `xelatex` / `lualatex`)."
        )


@app.command("run")
def run_pipeline(
    input: Path = typer.Option(..., "--input", help="Path to run input json"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", help="Output root directory"),
    ai_summary: bool = typer.Option(
        False,
        "--ai-summary",
        help="Write ai_summary.json (machine-readable concise verdict) into run output dir.",
    ),
) -> None:
    from .orchestrator import ReviewOrchestrator

    if not input.exists():
        raise typer.BadParameter(f"Input file not found: {input}")

    raw = json.loads(input.read_text(encoding="utf-8-sig"))
    raw = _resolve_paper_path(input, raw)

    review_input = ReviewRunInput.model_validate(raw)
    _validate_executor_or_die(review_input)

    orchestrator = ReviewOrchestrator(_repo_root())
    summary = orchestrator.run(review_input, output_dir)

    console.print(f"run_id: {summary.run_id}")
    console.print(f"status: {summary.status.value}")
    console.print(f"output: {summary.output_dir}")
    if summary.qa_issues:
        console.print("qa_issues:")
        for item in summary.qa_issues:
            console.print(f"- {item}")
    if ai_summary:
        run_dir = Path(summary.output_dir)
        payload = _build_ai_summary_payload(summary, run_dir)
        summary_path = run_dir / "ai_summary.json"
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"ai_summary: {summary_path}")


@app.command("tool-venue-profile")
def tool_venue_profile(
    venue: str = typer.Option(..., "--venue", help="Venue name or slug, e.g. ICLR"),
    year: int = typer.Option(..., "--year", help="Venue year, e.g. 2026"),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Optional output JSON path.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print JSON payload to stdout.",
    ),
) -> None:
    """Tool mode: resolve venue rubric/profile without running the review pipeline."""
    root = _repo_root()
    profile, used_fallback, source = load_venue_profile(root, venue, year)
    payload = {
        "venue": normalize_venue_slug(venue),
        "year": year,
        "used_fallback": used_fallback,
        "source": source,
        "profile": profile.model_dump(mode="json"),
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"saved: {output}")

    if as_json or output is None:
        console.print_json(data=payload)


@app.command("tool-parse-paper")
def tool_parse_paper(
    paper_path: Path = typer.Option(..., "--paper-path", help="Path to paper (.pdf or .md)."),
    paper_format: str = typer.Option(
        "auto",
        "--paper-format",
        help="auto|pdf|md",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Optional output JSON path.",
    ),
    include_raw_text: bool = typer.Option(
        True,
        "--include-raw-text/--no-include-raw-text",
        help="Whether to include full raw_text in JSON output.",
    ),
) -> None:
    """Tool mode: parse paper and emit structured sections for agent analysis."""
    if not paper_path.exists():
        raise typer.BadParameter(f"Paper file not found: {paper_path}")

    fmt = paper_format.strip().lower()
    if fmt == "auto":
        fmt = _paper_format_from_path(paper_path)
    if fmt not in {"pdf", "md"}:
        raise typer.BadParameter("paper-format must be one of: auto|pdf|md")

    structured = parse_markdown(paper_path) if fmt == "md" else parse_pdf(paper_path)
    raw_text = str(structured.get("raw_text", ""))
    sections = _attach_section_ids(structured.get("sections", []))
    nonempty_sections = [
        s
        for s in sections
        if isinstance(s, dict) and str(s.get("text", "")).strip()
    ]
    payload = {
        "paper_path": str(paper_path.resolve()),
        "paper_format": fmt,
        "title": structured.get("title", "Untitled"),
        "parse_backend": str(structured.get("parse_backend", "markdown")),
        "parse_quality": {
            "word_count": len(raw_text.split()),
            "section_count": len(nonempty_sections),
        },
        "warnings": [],
        "sections": sections,
        "section_locator": _build_section_locator(sections),
        "pages": structured.get("pages", []) if fmt == "pdf" else [],
    }

    if fmt == "pdf":
        warnings = _pdf_parse_quality_warnings(
            raw_text=raw_text,
            word_count=int(payload["parse_quality"]["word_count"]),
            section_count=int(payload["parse_quality"]["section_count"]),
            parse_backend=payload["parse_backend"],
        )
        payload["warnings"] = warnings

    if include_raw_text:
        payload["raw_text"] = raw_text

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"saved: {output}")
    else:
        console.print_json(data=payload)


@app.command("tool-format-template")
def tool_format_template(
    template: str = typer.Option(
        "student_pack_analysis",
        "--template",
        help="Template name. Currently: student_pack_analysis",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Optional path to save the template JSON.",
    ),
) -> None:
    """Tool mode: emit JSON template contract for agent-authored analysis payload."""
    if template.strip().lower() != "student_pack_analysis":
        raise typer.BadParameter("Unsupported template. Use --template student_pack_analysis")

    payload = student_pack_analysis_template()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"saved: {output}")
    else:
        console.print_json(data=payload)


@app.command("tool-format-student-pack")
def tool_format_student_pack(
    analysis_json: Path = typer.Option(
        ...,
        "--analysis-json",
        help="Agent-authored analysis JSON following tool-format-template contract.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        help="Directory to write 001/002/003 markdown files.",
    ),
    language: str = typer.Option("en", "--language", help="en|zh"),
) -> None:
    """Tool mode: format agent analysis payload into student-facing markdown pack."""
    if not analysis_json.exists():
        raise typer.BadParameter(f"analysis-json not found: {analysis_json}")
    payload = json.loads(analysis_json.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise typer.BadParameter("analysis-json must be a JSON object.")

    pack = render_student_pack_markdown(payload, language=language)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for filename, content in pack.items():
        target = output_dir / filename
        target.write_text(content, encoding="utf-8")
        saved.append(str(target))

    manifest = {
        "source_analysis": str(analysis_json),
        "language": language,
        "files": saved,
    }
    manifest_path = output_dir / "student_pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"saved: {manifest_path}")
    for item in saved:
        console.print(f"- {item}")


@app.command("refresh-venue")
def refresh_venue(
    venue: str = typer.Option(
        "all",
        "--venue",
        help="Venue slug(s), comma separated, or 'all'. Example: iclr,icml",
    ),
    year: int = typer.Option(..., "--year", help="Target year snapshot to write."),
    openreview_group: str = typer.Option(
        "",
        "--openreview-group",
        help="Optional OpenReview group id override for single venue sync.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only, do not write files."),
) -> None:
    """Refresh venue/year policy snapshots from OpenReview policy extraction."""
    root = _repo_root()
    summary = refresh_venue_rules(
        root,
        venue=venue,
        year=year,
        openreview_group=openreview_group,
        dry_run=dry_run,
    )

    table = Table(title=f"refresh-venue ({year})")
    table.add_column("Venue")
    table.add_column("Status")
    table.add_column("Group")
    table.add_column("Warning")
    for item in summary["items"]:
        table.add_row(
            item["venue"],
            item["status"],
            item.get("openreview_group_id") or "-",
            item.get("warning") or "-",
        )
    console.print(table)
    console.print(f"updated_count={summary['updated_count']}, failed_count={summary['failed_count']}")


@app.command("submit-feedback")
def submit_feedback_command(
    input: Path = typer.Option(..., "--input", help="Path to feedback template json"),
) -> None:
    """Submit user risk feedback and persist to feedback/<venue>/<year>/."""
    if not input.exists():
        raise typer.BadParameter(f"Input file not found: {input}")

    payload = json.loads(input.read_text(encoding="utf-8-sig"))
    result = submit_feedback(_repo_root(), payload)
    console.print(f"saved_to: {result['saved_to']}")
    console.print(f"accepted_items: {result['accepted_items']}")


if __name__ == "__main__":
    app()
