from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models import ReviewRunInput, RunStatus

if TYPE_CHECKING:
    from ..providers.base import MCPToolProvider


@dataclass
class PipelineContext:
    run_id: str
    run_dir: Path
    input_data: ReviewRunInput
    repo_root: Path | None = None
    mcp_tools: "MCPToolProvider | None" = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    step_statuses: list[dict[str, Any]] = field(default_factory=list)
    qa_issues: list[str] = field(default_factory=list)
    _qa_issue_seen: set[str] = field(default_factory=set)
    status: RunStatus = RunStatus.SUCCESS

    def artifact_path(self, name: str) -> Path:
        return self.run_dir / name

    def dump_json(self, name: str, data: Any) -> Path:
        path = self.artifact_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def add_qa_issue(self, issue: str) -> None:
        clean = issue.strip()
        if not clean:
            return
        if clean in self._qa_issue_seen:
            return
        self._qa_issue_seen.add(clean)
        self.qa_issues.append(clean)


class PipelineStep:
    name = "base"

    def run(self, ctx: PipelineContext) -> None:
        raise NotImplementedError
