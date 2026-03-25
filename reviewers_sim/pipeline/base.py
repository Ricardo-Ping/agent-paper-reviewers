from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models import ReviewRunInput, RunStatus

if TYPE_CHECKING:
    from ..mcp.base import MCPToolProvider


@dataclass
class PipelineContext:
    run_id: str
    run_dir: Path
    input_data: ReviewRunInput
    mcp_tools: "MCPToolProvider | None" = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    qa_issues: list[str] = field(default_factory=list)
    status: RunStatus = RunStatus.SUCCESS

    def artifact_path(self, name: str) -> Path:
        return self.run_dir / name

    def dump_json(self, name: str, data: Any) -> Path:
        path = self.artifact_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class PipelineStep:
    name = "base"

    def run(self, ctx: PipelineContext) -> None:
        raise NotImplementedError
