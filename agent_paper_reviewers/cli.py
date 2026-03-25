from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .models import ReviewRunInput
from .orchestrator import ReviewOrchestrator

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
    if not input.exists():
        raise typer.BadParameter(f"Input file not found: {input}")

    raw = json.loads(input.read_text(encoding="utf-8-sig"))
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
def refresh_venue() -> None:
    """Generate a changelog stub for monthly venue policy refresh."""
    root = _repo_root()
    changelog = root / "data" / "venue_rules" / "changelog.md"
    changelog.parent.mkdir(parents=True, exist_ok=True)
    if not changelog.exists():
        changelog.write_text("# Venue Policy Changelog\n\n", encoding="utf-8")

    existing = changelog.read_text(encoding="utf-8")
    snippet = "- TODO: Refresh venue policy snapshots and rerun regression suite.\n"
    if snippet not in existing:
        changelog.write_text(existing + snippet, encoding="utf-8")
    console.print(f"Updated {changelog}")


if __name__ == "__main__":
    app()

