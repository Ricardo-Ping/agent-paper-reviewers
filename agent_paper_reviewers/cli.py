from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .models import ReviewRunInput
from .services.venue_sync import refresh_venue_rules

app = typer.Typer(help="agent-paper-reviewers pipeline CLI")
console = Console()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@app.command("doctor")
def doctor() -> None:
    """Check runtime dependencies and external tools."""
    checks = {
        "python": shutil.which("python") is not None,
        "pandoc": shutil.which("pandoc") is not None,
        "xelatex": shutil.which("xelatex") is not None,
        "lualatex": shutil.which("lualatex") is not None,
        "tectonic": shutil.which("tectonic") is not None,
        "conda": shutil.which("conda") is not None,
    }

    table = Table(title="agent-paper-reviewers doctor")
    table.add_column("Dependency")
    table.add_column("Available")
    for dep, ok in checks.items():
        table.add_row(dep, "yes" if ok else "no")
    console.print(table)


@app.command("run")
def run_pipeline(
    input: Path = typer.Option(..., "--input", help="Path to run input json"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", help="Output root directory"),
) -> None:
    from .orchestrator import ReviewOrchestrator

    if not input.exists():
        raise typer.BadParameter(f"Input file not found: {input}")

    raw = json.loads(input.read_text(encoding="utf-8-sig"))

    # Resolve relative paper paths against the input json directory.
    paper = raw.get("paper", {})
    paper_path = paper.get("path")
    if isinstance(paper_path, str) and paper_path.strip():
        path_obj = Path(paper_path)
        if not path_obj.is_absolute():
            raw["paper"]["path"] = str((input.parent / path_obj).resolve())

    review_input = ReviewRunInput.model_validate(raw)

    orchestrator = ReviewOrchestrator(_repo_root())
    summary = orchestrator.run(review_input, output_dir)

    console.print(f"run_id: {summary.run_id}")
    console.print(f"status: {summary.status.value}")
    console.print(f"output: {summary.output_dir}")
    if summary.qa_issues:
        console.print("qa_issues:")
        for item in summary.qa_issues:
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


if __name__ == "__main__":
    app()

