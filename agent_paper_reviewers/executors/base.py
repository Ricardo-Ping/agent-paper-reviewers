from __future__ import annotations

from ..models import TaskResult, TaskSpec


class ExecutorAdapter:
    def execute(self, spec: TaskSpec) -> TaskResult:
        raise NotImplementedError
