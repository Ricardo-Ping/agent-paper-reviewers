from __future__ import annotations

from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency fallback
    fuzz = None

from ..models import TaskResult, TaskSpec
from .base import ExecutorAdapter


class DeterministicExecutor(ExecutorAdapter):
    """Offline-safe executor used when no external model backend is configured."""

    def execute(self, spec: TaskSpec) -> TaskResult:
        if spec.task_type == "claim_normalize":
            return TaskResult(ok=True, output=self._claim_normalize(spec))

        if spec.task_type == "risk_ranking":
            return TaskResult(ok=True, output=self._risk_ranking(spec))

        if spec.task_type == "remediation_plan":
            return TaskResult(ok=True, output=self._remediation_plan(spec))

        if spec.task_type == "translate_zh":
            text = spec.context.get("text", "")
            return TaskResult(ok=True, output={"translated_text": self._pseudo_translate(text)})

        if spec.task_type == "summarize":
            text = spec.context.get("text", "")
            summary = text[: min(len(text), 600)]
            return TaskResult(ok=True, output={"summary": summary})

        if spec.task_type == "score_similarity":
            a = spec.context.get("a", "")
            b = spec.context.get("b", "")
            if fuzz is not None:
                score = fuzz.token_set_ratio(a, b) / 100.0
            else:
                score = SequenceMatcher(None, a, b).ratio()
            return TaskResult(ok=True, output={"score": score})

        return TaskResult(ok=True, output={"note": "No-op deterministic result."})

    @staticmethod
    def _claim_normalize(spec: TaskSpec) -> dict:
        claim = str(spec.context.get("raw_claim", "")).strip()
        claim_id = str(spec.context.get("claim_id", "C1"))
        claim_type = DeterministicExecutor._infer_claim_type(claim)
        return {
            "claim_id": claim_id,
            "text": claim,
            "type": claim_type,
            "verifiable_claim": DeterministicExecutor._default_verifiable_claim(claim_type, claim),
            "success_criteria": DeterministicExecutor._default_success_criteria(claim_type),
            "weakness_hint": DeterministicExecutor._default_weakness_hint(claim_type),
        }

    @staticmethod
    def _risk_ranking(spec: TaskSpec) -> dict:
        alignments = spec.context.get("alignments", [])
        gaps = spec.context.get("gaps", [])
        risks: list[dict] = []
        idx = 1

        if isinstance(alignments, list):
            for item in alignments:
                if not isinstance(item, dict):
                    continue
                strength = str(item.get("strength", "")).lower()
                if strength not in {"weak", "none"}:
                    continue
                score = 0.82 if strength == "none" else 0.56
                risks.append(
                    {
                        "id": f"RISK-{idx:03d}",
                        "severity": "P0" if strength == "none" else "P1",
                        "score": round(score, 3),
                        "reason": f"Claim {item.get('claim_id', f'C{idx}')} has {strength} evidence support.",
                        "evidence_refs": item.get("evidence_refs", []) if isinstance(item.get("evidence_refs"), list) else [],
                        "likely_reject_phrase": "Core claims are not sufficiently supported by rigorous evidence.",
                        "fix_hint": "Add direct experiments and statistical validation tied to this claim.",
                    }
                )
                idx += 1

        gap_score_map = {
            "missing_significance": 0.62,
            "missing_baseline": 0.66,
            "missing_ablation": 0.58,
            "missing_reproducibility": 0.52,
            "missing_reference_coverage": 0.57,
            "weak_novelty_signal_from_citations": 0.48,
        }
        if isinstance(gaps, list):
            for gap in gaps:
                if not isinstance(gap, dict):
                    continue
                score = float(gap_score_map.get(str(gap.get("code", "")), 0.45))
                severity = "P0" if score >= 0.75 else "P1" if score >= 0.45 else "P2"
                risks.append(
                    {
                        "id": f"RISK-{idx:03d}",
                        "severity": severity,
                        "score": round(score, 3),
                        "reason": str(gap.get("description") or "Detected venue compliance gap."),
                        "evidence_refs": gap.get("evidence_refs", []) if isinstance(gap.get("evidence_refs"), list) else [],
                        "likely_reject_phrase": "Experimental evidence does not yet meet venue expectations.",
                        "fix_hint": "Address this with a focused experiment or analysis update.",
                    }
                )
                idx += 1

        if not risks:
            risks = [
                {
                    "id": "RISK-001",
                    "severity": "P2",
                    "score": 0.35,
                    "reason": "No explicit high-risk signal detected from current evidence.",
                    "evidence_refs": [],
                    "likely_reject_phrase": "Current draft still leaves reviewer concerns about contribution quality.",
                    "fix_hint": "Strengthen claim-to-evidence mapping and clarify contribution scope.",
                }
            ]

        p0 = sum(1 for r in risks if r["severity"] == "P0")
        p1 = sum(1 for r in risks if r["severity"] == "P1")
        p2 = sum(1 for r in risks if r["severity"] == "P2")

        novelty = max(0.0, 8.5 - 0.8 * p1 - 1.2 * p0)
        soundness = max(0.0, 8.0 - 1.0 * p1 - 1.5 * p0)
        experiment = max(0.0, 8.2 - 1.1 * p1 - 1.4 * p0)
        clarity = max(0.0, 8.8 - 0.5 * p2 - 0.6 * p1)
        overall = round((novelty + soundness + experiment + clarity) / 4.0, 2)

        return {
            "risks": sorted(risks, key=lambda x: float(x.get("score", 0.0)), reverse=True),
            "scores": {
                "novelty": round(novelty, 2),
                "soundness": round(soundness, 2),
                "experiment": round(experiment, 2),
                "clarity": round(clarity, 2),
                "overall": overall,
            },
        }

    @staticmethod
    def _remediation_plan(spec: TaskSpec) -> dict:
        risks = spec.context.get("risks", [])
        constraints = spec.context.get("constraints", {})

        max_n = int(constraints.get("max_new_experiments", 6) or 6)
        gpu_budget = int(constraints.get("gpu_budget_hours", 120) or 120)
        time_days = float(constraints.get("time_days", 10) or 10)

        tasks: list[dict] = []
        used_gpu = 0
        used_days = 0.0
        if not isinstance(risks, list):
            risks = []

        for idx, risk in enumerate(risks, start=1):
            if len(tasks) >= max_n:
                break
            if not isinstance(risk, dict):
                continue

            severity = str(risk.get("severity", "P2")).upper()
            effort = "L" if severity == "P0" else "M" if severity == "P1" else "S"
            priority = "high" if severity in {"P0", "P1"} else "medium"
            est_gpu = 32 if effort == "L" else 12 if effort == "M" else 4
            est_days = 4.0 if effort == "L" else 2.0 if effort == "M" else 1.0

            if used_gpu + est_gpu > gpu_budget:
                continue
            if used_days + est_days > time_days:
                continue

            risk_id = str(risk.get("id") or f"RISK-{idx:03d}")
            tasks.append(
                {
                    "id": f"EXP-{len(tasks)+1:03d}",
                    "risk_id": risk_id,
                    "title": f"Mitigate {risk_id} with focused evidence update",
                    "priority": priority,
                    "effort": effort,
                    "est_time_days": est_days,
                    "est_gpu_hours": est_gpu,
                    "expected_gain": "Reduce rejection risk by adding direct, claim-grounded evidence.",
                    "protocol": [
                        "Define hypothesis and acceptance metric.",
                        "Run matched-baseline experiments with controlled settings.",
                        "Report multi-seed mean/std and significance tests.",
                    ],
                }
            )
            used_gpu += est_gpu
            used_days += est_days

        return {"tasks": tasks}

    @staticmethod
    def _pseudo_translate(text: str) -> str:
        glossary = {
            "Novelty": "新颖性",
            "Soundness": "技术正确性",
            "Experiment": "实验充分性",
            "Clarity": "写作清晰度",
            "Rebuttal": "答辩回复",
            "Risk": "风险",
            "Decision": "决策",
            "Not Ready": "不建议投稿",
            "Borderline": "边界状态",
            "Ready": "可投稿",
        }
        translated = text
        for en, zh in glossary.items():
            translated = translated.replace(en, zh)
        return translated

    @staticmethod
    def _infer_claim_type(text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["baseline", "compare", "compared with", "outperform"]):
            return "baseline"
        if any(k in t for k in ["ablation", "component", "remove", "without"]):
            return "ablation"
        if any(k in t for k in ["significant", "p-value", "std", "variance", "seed"]):
            return "statistical"
        if any(k in t for k in ["reproduce", "reproduc", "code", "implementation", "deterministic"]):
            return "reproducibility"
        return "novelty"

    @staticmethod
    def _default_verifiable_claim(claim_type: str, claim: str) -> str:
        if claim_type == "baseline":
            return f"Compared with strong baselines under matched settings, '{claim}' should show consistent improvement."
        if claim_type == "ablation":
            return f"Ablation experiments should isolate key components supporting '{claim}'."
        if claim_type == "statistical":
            return f"'{claim}' should hold with multi-seed statistics and significance testing."
        if claim_type == "reproducibility":
            return f"'{claim}' should be reproducible with complete implementation and configuration details."
        return f"The novelty claim '{claim}' should be supported by direct empirical or analytical evidence."

    @staticmethod
    def _default_success_criteria(claim_type: str) -> str:
        if claim_type == "baseline":
            return "Report consistent gains against strong baselines under identical settings."
        if claim_type == "ablation":
            return "Show each key component contributes measurably in ablation tables."
        if claim_type == "statistical":
            return "Provide mean/std over multiple seeds and significance tests."
        if claim_type == "reproducibility":
            return "Provide enough details so independent reruns reproduce main results."
        return "Provide direct evidence linking contribution claims to measurable outcomes."

    @staticmethod
    def _default_weakness_hint(claim_type: str) -> str:
        if claim_type == "baseline":
            return "Weak baseline setup can invalidate comparative conclusions."
        if claim_type == "ablation":
            return "Missing controlled ablations can make attribution unclear."
        if claim_type == "statistical":
            return "Single-run metrics without significance testing may be unstable."
        if claim_type == "reproducibility":
            return "Insufficient implementation details may block independent verification."
        return "Novelty claims can be rejected if evidence is indirect or positioning is unclear."
