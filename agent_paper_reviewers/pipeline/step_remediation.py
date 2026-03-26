from __future__ import annotations

import json
import re

from ..executors.base import ExecutorAdapter
from ..models import Constraints, ExperimentTask, TaskSpec
from .base import PipelineContext, PipelineStep


class RemediationPlannerStep(PipelineStep):
    name = "RemediationPlanner"

    def __init__(self, executor: ExecutorAdapter | None = None) -> None:
        self.executor = executor

    def run(self, ctx: PipelineContext) -> None:
        risks = ctx.artifacts["risk_ranking"]["risks"]
        gaps = ctx.artifacts.get("gaps", {}).get("gaps", [])
        constraints = ctx.input_data.constraints

        tasks = self._plan_with_executor(ctx, risks, gaps, constraints)
        source = "executor"
        if tasks is None:
            tasks = self._plan_rule_based(risks, gaps, constraints.max_new_experiments)
            source = "rule_fallback"

        tasks = self._enforce_constraints(tasks, constraints, risks)
        tasks = self._renumber_tasks(tasks)

        payload = {
            "tasks": tasks,
            "source": source,
            "total_est_gpu_hours": sum(int(t["est_gpu_hours"]) for t in tasks),
            "total_est_time_days": round(sum(float(t["est_time_days"]) for t in tasks), 2),
            "max_new_experiments": constraints.max_new_experiments,
            "gpu_budget_hours": constraints.gpu_budget_hours,
            "time_days": constraints.time_days,
        }
        ctx.artifacts["remediation_plan"] = payload
        ctx.dump_json("artifacts/remediation_plan.json", payload)

    def _plan_with_executor(
        self,
        ctx: PipelineContext,
        risks: list[dict],
        gaps: list[dict],
        constraints: Constraints,
    ) -> list[dict] | None:
        if self.executor is None:
            return None

        prompt = (
            "Generate a remediation experiment plan based on paper risks and constraints.\n"
            "Output MUST be JSON only. Prefer high-severity/high-score risks first.\n\n"
            f"Risks:\n{json.dumps(risks, ensure_ascii=False, indent=2)}\n\n"
            f"Gaps:\n{json.dumps(gaps, ensure_ascii=False, indent=2)}\n\n"
            "Resource constraints:\n"
            f"- Time days: {constraints.time_days}\n"
            f"- GPU budget hours: {constraints.gpu_budget_hours}\n"
            f"- Max experiments: {constraints.max_new_experiments}\n"
            f"- Cannot run: {constraints.cannot_run}\n\n"
            "For each experiment include:\n"
            "title, risk_id, priority(high|medium|low), effort(S|M|L), est_time_days, est_gpu_hours, expected_gain, protocol(list).\n"
            "Ensure your final plan is feasible under all constraints."
        )

        spec = TaskSpec(
            task_type="remediation_plan",
            prompt=prompt,
            context={
                "risks": risks,
                "gaps": gaps,
                "constraints": constraints.model_dump(),
            },
            output_schema={
                "tasks": [
                    {
                        "id": "EXP-001",
                        "risk_id": "RISK-001",
                        "title": "Mitigate RISK-001 with focused experiment",
                        "priority": "high",
                        "effort": "M",
                        "est_time_days": 2.0,
                        "est_gpu_hours": 12,
                        "expected_gain": "Reduce rejection risk on statistical validity.",
                        "protocol": ["step1", "step2"],
                    }
                ]
            },
            model_profile="judge",
        )

        result = self.executor.execute(spec)
        for w in result.warnings:
            ctx.add_qa_issue(f"remediation_executor_warning:{w}")

        if not result.ok:
            ctx.add_qa_issue("remediation_executor_not_ok_use_rule_fallback")
            return None

        tasks = self._normalize_executor_tasks(result.output, risks, gaps, constraints.max_new_experiments)
        if tasks is None:
            ctx.add_qa_issue("remediation_executor_output_invalid_use_rule_fallback")
            return None

        return tasks

    def _normalize_executor_tasks(
        self,
        raw_output: object,
        risks: list[dict],
        gaps: list[dict],
        max_n: int,
    ) -> list[dict] | None:
        raw_tasks = self._extract_task_list(raw_output)
        if raw_tasks is None:
            return None

        risk_ids = [r["id"] for r in risks]
        risk_id_set = set(risk_ids)
        risk_default = risk_ids[0] if risk_ids else "RISK-001"

        tasks: list[dict] = []
        limit = max(max_n * 3, max_n)
        for idx, item in enumerate(raw_tasks[:limit], start=1):
            if not isinstance(item, dict):
                continue

            risk_id = str(item.get("risk_id") or "").strip()
            if risk_id not in risk_id_set:
                risk_id = self._guess_risk_id(item, risk_ids) or risk_default

            priority = str(item.get("priority") or "").strip().lower()
            if priority not in {"high", "medium", "low"}:
                priority = self._priority_from_risk(risk_id, risks)

            effort = str(item.get("effort") or "").strip().upper()
            if effort not in {"S", "M", "L"}:
                effort = "L" if priority == "high" else "M" if priority == "medium" else "S"

            default_time = 4.0 if effort == "L" else 2.0 if effort == "M" else 1.0
            default_gpu = 32 if effort == "L" else 12 if effort == "M" else 4

            est_time_days = self._to_float(item.get("est_time_days"), default_time)
            est_gpu_hours = self._to_int(item.get("est_gpu_hours"), default_gpu)

            protocol = item.get("protocol")
            if not isinstance(protocol, list) or not protocol:
                protocol = self._default_protocol(risk_id, gaps, risk_reason="")

            task = ExperimentTask(
                id=str(item.get("id") or f"EXP-{idx:03d}"),
                risk_id=risk_id,
                title=str(item.get("title") or f"Mitigate {risk_id} - planned experiment"),
                priority=priority,
                effort=effort,
                est_time_days=max(0.5, est_time_days),
                est_gpu_hours=max(0, est_gpu_hours),
                expected_gain=str(
                    item.get("expected_gain")
                    or "Reduce rejection likelihood by adding direct claim-grounded evidence."
                ),
                protocol=[str(x) for x in protocol if str(x).strip()],
            )
            tasks.append(task.model_dump())

        if not tasks:
            return None

        return tasks

    @staticmethod
    def _extract_task_list(raw_output: object) -> list[dict] | None:
        if isinstance(raw_output, list):
            return [x for x in raw_output if isinstance(x, dict)]

        if isinstance(raw_output, dict):
            if isinstance(raw_output.get("tasks"), list):
                return [x for x in raw_output["tasks"] if isinstance(x, dict)]
            response = raw_output.get("response")
            if isinstance(response, list):
                return [x for x in response if isinstance(x, dict)]
            if isinstance(response, dict) and isinstance(response.get("tasks"), list):
                return [x for x in response["tasks"] if isinstance(x, dict)]
            if isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    return RemediationPlannerStep._extract_task_list(parsed)
                except Exception:  # noqa: BLE001
                    return None
        return None

    @staticmethod
    def _guess_risk_id(item: dict, risk_ids: list[str]) -> str | None:
        if not risk_ids:
            return None
        blob = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("expected_gain") or ""),
                " ".join(str(x) for x in item.get("protocol", []) if isinstance(x, str)),
            ]
        ).upper()
        for rid in risk_ids:
            if rid.upper() in blob:
                return rid
        return None

    @staticmethod
    def _priority_from_risk(risk_id: str, risks: list[dict]) -> str:
        for risk in risks:
            if risk.get("id") != risk_id:
                continue
            severity = str(risk.get("severity", "")).upper()
            if severity == "P0":
                return "high"
            if severity == "P1":
                return "high"
            return "medium"
        return "medium"

    @staticmethod
    def _default_protocol(risk_id: str, gaps: list[dict], risk_reason: str = "") -> list[str]:
        reason = risk_reason.lower()
        if any(k in reason for k in ["contradiction", "conflict", "opposite result", "inconsistent with claim"]):
            return [
                f"Build contradiction-resolution checklist for {risk_id}.",
                "Locate conflicting figure/table anchors and verify metric direction (higher/lower is better).",
                "Split claim into scoped sub-claims that exactly match reported settings and datasets.",
                "Add reconciled result table with explicit why/when we lose to baseline analysis.",
            ]
        if "baseline" in reason:
            return [
                f"Define fair-comparison protocol for {risk_id} (matched training budget/hardware).",
                "Add at least two strong baselines and report full metric table.",
                "Document baseline tuning strategy and fairness constraints.",
                "Explain wins/losses by scenario with failure cases.",
            ]
        if any(k in reason for k in ["significance", "statistical", "confidence interval", "p-value"]):
            return [
                f"Set statistical validation protocol for {risk_id}.",
                "Run >=3 seeds for each compared method and report mean/std.",
                "Apply paired significance tests on primary metrics and report p-values.",
                "Add confidence intervals and variance-aware interpretation.",
            ]
        if "ablation" in reason:
            return [
                f"Define component attribution targets for {risk_id}.",
                "Add one-by-one component ablation table.",
                "Add interaction ablations for key component pairs.",
                "Discuss causal interpretation and failure patterns.",
            ]
        if any(k in reason for k in ["reproduc", "implementation", "environment"]):
            return [
                f"Create reproducibility checklist for {risk_id}.",
                "Release full hyperparameters, data preprocessing, and environment settings.",
                "Provide deterministic rerun commands and seed policy.",
                "Validate one independent rerun and report drift tolerance.",
            ]
        if any(k in reason for k in ["citation", "related work", "top-venue"]):
            return [
                f"Build related-work expansion checklist for {risk_id}.",
                "Add recent top-venue papers from last 2-3 years and closest baselines.",
                "Add explicit novelty positioning table (ours vs prior).",
                "Tie each core claim to at least one directly comparable prior method.",
            ]
        gap_hints = [g.get("description", "") for g in gaps[:2]]
        protocol = [
            f"Define hypothesis and acceptance metric for {risk_id}.",
            "Design one targeted experiment that directly validates the questioned claim.",
            "Report both aggregate metrics and error/failure breakdown.",
            "Add explicit paper-change mapping from new evidence to claim statement.",
        ]
        if gap_hints:
            protocol.append(f"Address identified gap explicitly: {gap_hints[0]}")
        return protocol

    def _plan_rule_based(self, risks: list[dict], gaps: list[dict], max_n: int) -> list[dict]:
        tasks: list[dict] = []
        for idx, risk in enumerate(risks[: max(max_n * 2, max_n)], start=1):
            severity = str(risk.get("severity", "P2"))
            effort = "L" if severity == "P0" else "M" if severity == "P1" else "S"
            priority = "high" if severity in {"P0", "P1"} else "medium"
            reason = str(risk.get("reason", ""))
            title = self._default_task_title(str(risk.get("id") or f"RISK-{idx:03d}"), reason)
            task = ExperimentTask(
                id=f"EXP-{idx:03d}",
                risk_id=str(risk.get("id") or f"RISK-{idx:03d}"),
                title=title,
                priority=priority,
                effort=effort,
                est_time_days=4.0 if effort == "L" else 2.0 if effort == "M" else 1.0,
                est_gpu_hours=36 if effort == "L" else 14 if effort == "M" else 4,
                expected_gain="Reduce rejection likelihood by strengthening claim-evidence linkage.",
                protocol=self._default_protocol(
                    str(risk.get("id") or f"RISK-{idx:03d}"),
                    gaps,
                    risk_reason=reason,
                ),
            )
            tasks.append(task.model_dump())

        return tasks

    @staticmethod
    def _default_task_title(risk_id: str, reason: str) -> str:
        lower = reason.lower()
        if any(k in lower for k in ["contradiction", "conflict", "opposite result", "inconsistent with claim"]):
            return f"Resolve Claim-Result Contradictions for {risk_id}"
        if "baseline" in lower:
            return f"Strengthen Baseline Fairness for {risk_id}"
        if any(k in lower for k in ["significance", "statistical", "p-value"]):
            return f"Add Statistical Validation Suite for {risk_id}"
        if "ablation" in lower:
            return f"Complete Component Ablation Matrix for {risk_id}"
        if "reproduc" in lower:
            return f"Build Reproducibility Package for {risk_id}"
        if any(k in lower for k in ["citation", "related work", "top-venue"]):
            return f"Expand Related Work Positioning for {risk_id}"
        return f"Targeted Claim Validation for {risk_id}"

    def _enforce_constraints(self, tasks: list[dict], constraints: Constraints, risks: list[dict]) -> list[dict]:
        max_n = max(0, int(constraints.max_new_experiments))
        gpu_budget = max(0, int(constraints.gpu_budget_hours))
        time_budget = max(0.0, float(constraints.time_days))

        if max_n == 0 or not tasks:
            return []

        risk_map = {r.get("id"): r for r in risks}

        def utility(task: dict) -> float:
            priority = str(task.get("priority", "medium")).lower()
            p_weight = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(priority, 2.0)

            risk = risk_map.get(task.get("risk_id"), {})
            severity = str(risk.get("severity", "P2")).upper()
            s_weight = {"P0": 3.0, "P1": 2.0, "P2": 1.0}.get(severity, 1.0)
            r_score = float(risk.get("score", 0.45))

            gpu = max(1, int(task.get("est_gpu_hours", 0)))
            days = max(0.5, float(task.get("est_time_days", 0.5)))
            gpu_norm = gpu / max(1, gpu_budget)
            day_norm = days / max(1.0, time_budget)
            cost = max(0.2, gpu_norm + day_norm)
            return (p_weight + s_weight + r_score) / cost

        ranked = sorted(tasks, key=utility, reverse=True)

        selected: list[dict] = []
        used_gpu = 0
        used_days = 0.0
        seen_risk_ids: set[str] = set()

        for task in ranked:
            if len(selected) >= max_n:
                break

            risk_id = str(task.get("risk_id") or "")
            if risk_id in seen_risk_ids:
                continue

            task_gpu = max(0, int(task.get("est_gpu_hours", 0)))
            task_days = max(0.5, float(task.get("est_time_days", 0.5)))

            if used_gpu + task_gpu > gpu_budget:
                continue
            if used_days + task_days > time_budget:
                continue

            selected.append(task)
            used_gpu += task_gpu
            used_days += task_days
            if risk_id:
                seen_risk_ids.add(risk_id)

        if not selected:
            # Graceful fallback: choose the cheapest feasible task if any.
            feasible = [
                t
                for t in ranked
                if int(t.get("est_gpu_hours", 0)) <= gpu_budget
                and float(t.get("est_time_days", 0.5)) <= time_budget
            ]
            if feasible and max_n > 0:
                selected = [feasible[0]]

        return selected

    @staticmethod
    def _renumber_tasks(tasks: list[dict]) -> list[dict]:
        out = []
        for idx, task in enumerate(tasks, start=1):
            row = dict(task)
            row["id"] = f"EXP-{idx:03d}"
            out.append(row)
        return out

    @staticmethod
    def _to_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: object, default: int) -> int:
        try:
            if isinstance(value, str):
                m = re.search(r"\d+", value)
                if m:
                    return int(m.group(0))
            return int(value)
        except (TypeError, ValueError):
            return default
