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
from .models import ExecutorBackend, ManuscriptStage, ReviewRunInput, RunStatus
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
    run_dir = run_dir.resolve()
    decision = _safe_read_json(run_dir / "decision_brief.en.json")
    risk = _safe_read_json(run_dir / "artifacts" / "risk_ranking.json")
    rebuttal_path = run_dir / "rebuttal.en.md"
    student_pack_decision = run_dir / "student_pack" / "en" / "001-submission-decision.md"
    run_guide_path = run_dir / "RUN_GUIDE.en.md"
    student_brief_path = run_dir / "STUDENT_BRIEF.en.md"
    persona_playbook_path = run_dir / "PERSONA_PLAYBOOK.en.md"
    agent_handoff_path = run_dir / "AGENT_HANDOFF.json"

    def _rel(path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        try:
            return str(path.resolve().relative_to(run_dir)).replace("\\", "/")
        except Exception:  # noqa: BLE001
            return str(path.resolve())

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

    step_statuses = summary.step_statuses if isinstance(summary.step_statuses, list) else []
    step_overview = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "pending": 0,
        "running": 0,
    }
    for row in step_statuses:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "")).strip().lower()
        if status in step_overview:
            step_overview[status] += 1

    qa_strings = [str(x) for x in summary.qa_issues]
    student_pack_blocked = any("student_pack_generation_failed" in x for x in qa_strings)
    student_pack_ready = student_pack_decision.exists() and not student_pack_blocked
    degraded_reasons: list[str] = []
    if summary.status != RunStatus.SUCCESS:
        degraded_reasons.append(f"run_status={summary.status.value}")
    if fallback_count > 0:
        degraded_reasons.append(f"executor_fallback_count={fallback_count}")
    degraded_reasons.extend(qa_strings[:6])
    degraded = bool(degraded_reasons)

    if student_pack_blocked:
        recommended_next_action = (
            "Student pack is blocked. Configure a live executor backend and rerun "
            "(OpenAI/Anthropic/OpenClaw/local vLLM endpoint)."
        )
    elif must_fix:
        recommended_next_action = (
            "Open student_pack/en/002-action-items.md and complete all P0/P1 tasks before submission."
        )
    elif summary.status == RunStatus.FAILED:
        recommended_next_action = "Open pipeline_exception.log and fix the failed pipeline step first."
    else:
        recommended_next_action = (
            "Start from START_HERE.en.md and verify rebuttal anchors in student_pack/en/003-rebuttal-draft.md."
        )

    return {
        "run_id": summary.run_id,
        "run_dir": str(run_dir),
        "status": summary.status.value,
        "verdict": verdict,
        "top_risks": top_risks,
        "must_fix_before_submit": must_fix,
        "rebuttal_ready": _rel(rebuttal_path),
        "confidence": confidence,
        "qa_issue_count": len(summary.qa_issues),
        "degraded": degraded,
        "degraded_reasons": degraded_reasons,
        "student_pack_ready": student_pack_ready,
        "recommended_next_action": recommended_next_action,
        "step_overview": step_overview,
        "key_files": {
            "start_here": "START_HERE.en.md",
            "run_guide": _rel(run_guide_path),
            "student_brief": _rel(student_brief_path),
            "persona_playbook": _rel(persona_playbook_path),
            "student_pack_decision": _rel(student_pack_decision),
            "student_pack_actions": (
                _rel(run_dir / "student_pack" / "en" / "002-action-items.md")
            ),
            "student_pack_rebuttal": (
                _rel(run_dir / "student_pack" / "en" / "003-rebuttal-draft.md")
            ),
            "full_review": _rel(run_dir / "full_review.en.md"),
            "agent_handoff": _rel(agent_handoff_path),
        },
        "persona_routes": {
            "agent_first": [
                "AGENT_HANDOFF.json",
                "ai_summary.json",
                "RUN_GUIDE.en.md",
                "PERSONA_PLAYBOOK.en.md",
            ],
            "student_first": [
                "STUDENT_BRIEF.en.md",
                "PERSONA_PLAYBOOK.en.md",
                "student_pack/en/001-submission-decision.md",
                "student_pack/en/002-action-items.md",
                "student_pack/en/003-rebuttal-draft.md",
            ],
            "minimal_checks": [
                "run_result.json",
                "pipeline_steps.json",
                "AGENT_HANDOFF.json",
            ],
        },
    }


def _parse_reviewer_comment_flags(items: list[str]) -> list[dict]:
    parsed: list[dict] = []
    for idx, raw in enumerate(items, start=1):
        text = str(raw or "").strip()
        if not text:
            continue
        if "::" in text:
            review_id, concern = text.split("::", 1)
            rid = review_id.strip() or f"R{idx}"
            concern_text = concern.strip()
        else:
            rid = f"R{idx}"
            concern_text = text
        if not concern_text:
            continue
        parsed.append({"review_id": rid, "concern": concern_text})
    return parsed


def _strict_quality_fail_reasons(summary) -> list[str]:
    reasons: list[str] = []
    qa_issues = [str(x) for x in summary.qa_issues]
    if summary.status == RunStatus.FAILED:
        reasons.append("run_status=failed")
    if any("student_pack_generation_failed" in x for x in qa_issues):
        reasons.append("student_pack_not_ready")
    if any("executor_warning" in x for x in qa_issues):
        reasons.append("executor_warning_detected")
    if any("fallback" in x.lower() for x in qa_issues):
        reasons.append("fallback_detected")
    if any("pdf_parse_quality" in x for x in qa_issues):
        reasons.append("pdf_parse_quality_warning")
    return reasons


def _apply_strict_quality_gate(summary, run_dir: Path) -> None:
    reasons = _strict_quality_fail_reasons(summary)
    if not reasons:
        return
    console.print("[strict-quality] blocked due to:")
    for item in reasons:
        console.print(f"- {item}")
    console.print(f"see: {run_dir / 'RUN_GUIDE.en.md'}")
    raise typer.Exit(code=2)


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
    strict_quality: bool = typer.Option(
        False,
        "--strict-quality",
        help="Fail with non-zero code when quality blockers are detected.",
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
    if strict_quality:
        _apply_strict_quality_gate(summary, Path(summary.output_dir))


@app.command("review-pdf")
def review_pdf(
    paper_path: Path = typer.Option(..., "--paper-path", help="Path to paper PDF."),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", help="Output root directory"),
    venue: str = typer.Option(
        "unknownconf",
        "--venue",
        help="Target venue name. Use unknownconf to trigger recommendation-first behavior.",
    ),
    year: int = typer.Option(2026, "--year", help="Target venue year."),
    language_mode: str = typer.Option("en", "--language-mode", help="en|en_zh"),
    executor_backend: ExecutorBackend = typer.Option(
        ExecutorBackend.CODEX,
        "--executor-backend",
        help="codex|agent_api|openai|anthropic|qwen|local_vllm",
    ),
    always_export_pdf: bool = typer.Option(
        False,
        "--always-export-pdf/--no-always-export-pdf",
        help="Whether to export markdown reports to PDF.",
    ),
    claim: list[str] | None = typer.Option(
        None,
        "--claim",
        help="Optional repeatable core claims. If omitted, ClaimDiscoverer auto-discovers from paper.",
    ),
    manuscript_stage: ManuscriptStage = typer.Option(
        ManuscriptStage.INITIAL_SUBMISSION,
        "--manuscript-stage",
        help="initial_submission|rejected_after_reviews|meta_review_discussion",
    ),
    reviewer_comment: list[str] | None = typer.Option(
        None,
        "--reviewer-comment",
        help="Optional repeatable reviewer concern. Format: R1::text or plain text.",
    ),
    time_days: int = typer.Option(10, "--time-days", help="Resource constraint: available days."),
    gpu_budget_hours: int = typer.Option(
        200,
        "--gpu-budget-hours",
        help="Resource constraint: total GPU hours.",
    ),
    max_new_experiments: int = typer.Option(
        6,
        "--max-new-experiments",
        help="Resource constraint: max number of new experiments.",
    ),
    author_hash: str = typer.Option("", "--author-hash", help="Optional profile author hash."),
    ai_summary: bool = typer.Option(
        True,
        "--ai-summary/--no-ai-summary",
        help="Write ai_summary.json into run output dir.",
    ),
    save_generated_input: bool = typer.Option(
        True,
        "--save-generated-input/--no-save-generated-input",
        help="Save generated input payload to generated_input.json in run dir.",
    ),
    strict_quality: bool = typer.Option(
        False,
        "--strict-quality",
        help="Fail with non-zero code when quality blockers are detected.",
    ),
) -> None:
    """One-command PDF review for agent + graduate-student workflows."""
    from .orchestrator import ReviewOrchestrator

    if not paper_path.exists():
        raise typer.BadParameter(f"Paper file not found: {paper_path}")
    if paper_path.suffix.lower().strip() != ".pdf":
        raise typer.BadParameter("review-pdf currently expects a .pdf file.")
    lang_mode = language_mode.strip().lower()
    if lang_mode not in {"en", "en_zh"}:
        raise typer.BadParameter("language-mode must be en or en_zh.")

    reviewer_comment = reviewer_comment or []
    claim = claim or []
    generated_input = {
        "paper": {"format": "pdf", "path": str(paper_path.resolve())},
        "venue": {"name": venue, "year": year},
        "claims": [x for x in claim if str(x).strip()],
        "constraints": {
            "time_days": int(time_days),
            "gpu_budget_hours": int(gpu_budget_hours),
            "max_new_experiments": int(max_new_experiments),
        },
        "options": {
            "language_mode": lang_mode,
            "executor_backend": executor_backend.value,
            "always_export_pdf": bool(always_export_pdf),
        },
        "profile": {"author_hash": author_hash.strip()},
        "review_context": {
            "manuscript_stage": manuscript_stage.value,
            "reviewer_comments": _parse_reviewer_comment_flags(reviewer_comment),
        },
    }
    review_input = ReviewRunInput.model_validate(generated_input)
    _validate_executor_or_die(review_input)

    orchestrator = ReviewOrchestrator(_repo_root())
    summary = orchestrator.run(review_input, output_dir)
    run_dir = Path(summary.output_dir)

    if save_generated_input:
        (run_dir / "generated_input.json").write_text(
            json.dumps(generated_input, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    console.print(f"run_id: {summary.run_id}")
    console.print(f"status: {summary.status.value}")
    console.print(f"output: {summary.output_dir}")
    if summary.qa_issues:
        console.print("qa_issues:")
        for item in summary.qa_issues:
            console.print(f"- {item}")
    if ai_summary:
        payload = _build_ai_summary_payload(summary, run_dir)
        summary_path = run_dir / "ai_summary.json"
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"ai_summary: {summary_path}")
    if strict_quality:
        _apply_strict_quality_gate(summary, run_dir)


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
