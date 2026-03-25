from __future__ import annotations

import re

from ..services.translator import Translator
from .base import PipelineContext, PipelineStep


class ReportBuilderStep(PipelineStep):
    name = "ReportBuilder"

    def __init__(self, translator: Translator) -> None:
        self.translator = translator

    def run(self, ctx: PipelineContext) -> None:
        ranking = ctx.artifacts["risk_ranking"]
        risks = ranking["risks"]
        scores = ranking["scores"]
        remediation = ctx.artifacts["remediation_plan"]["tasks"]
        alignments = ctx.artifacts["claim_evidence_matrix"]["alignments"]

        decision = self._decision(risks)
        top_risks = risks[:5]
        top_tasks = remediation[:5]

        decision_json = {
            "decision": decision,
            "scores": scores,
            "top_risks": top_risks,
            "top_remediation_tasks": top_tasks,
        }
        full_json = {
            "decision": decision,
            "scores": scores,
            "all_risks": risks,
            "claim_evidence_matrix": alignments,
            "remediation_tasks": remediation,
            "rebuttal": ctx.artifacts["rebuttal"]["en"]["bundle"],
        }

        decision_md = self._decision_md(decision_json)
        full_md = self._full_md(full_json)

        payload = {
            "en": {
                "decision_json": decision_json,
                "decision_md": decision_md,
                "full_json": full_json,
                "full_md": full_md,
            }
        }

        if ctx.input_data.options.language_mode.value == "en_zh":
            zh_decision_json = self._decision_json_zh(decision_json)
            zh_full_json = self._full_json_zh(full_json, ctx)
            payload["zh"] = {
                "decision_json": zh_decision_json,
                "decision_md": self._decision_md_zh(zh_decision_json),
                "full_json": zh_full_json,
                "full_md": self._full_md_zh(zh_full_json),
            }

        ctx.artifacts["reports"] = payload
        ctx.dump_json("artifacts/report.decision.en.json", decision_json)
        ctx.dump_json("artifacts/report.full.en.json", full_json)
        if "zh" in payload:
            ctx.dump_json("artifacts/report.decision.zh.json", payload["zh"]["decision_json"])
            ctx.dump_json("artifacts/report.full.zh.json", payload["zh"]["full_json"])

    @staticmethod
    def _decision(risks: list[dict]) -> str:
        has_p0 = any(r["severity"] == "P0" for r in risks)
        p1_count = sum(1 for r in risks if r["severity"] == "P1")
        if has_p0:
            return "Not Ready"
        if p1_count >= 3:
            return "Borderline"
        return "Ready"

    @staticmethod
    def _decision_md(payload: dict) -> str:
        lines = [
            "# Submission Decision Brief",
            "",
            f"Decision: **{payload['decision']}**",
            "",
            "## Scores",
            f"- Novelty: {payload['scores']['novelty']}",
            f"- Soundness: {payload['scores']['soundness']}",
            f"- Experiment: {payload['scores']['experiment']}",
            f"- Clarity: {payload['scores']['clarity']}",
            f"- Overall: {payload['scores']['overall']}",
            "",
            "## Top Rejection Risks",
        ]
        for risk in payload["top_risks"]:
            lines.append(f"- [{risk['severity']}] {risk['id']} ({risk['score']}): {risk['reason']}")

        lines.append("\n## Must-Do Experiments")
        for task in payload["top_remediation_tasks"]:
            lines.append(
                f"- {task['id']} ({task['priority']}, effort={task['effort']}): {task['title']}"
            )

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _full_md(payload: dict) -> str:
        lines = [
            "# Full Review Report",
            "",
            f"Decision: **{payload['decision']}**",
            "",
            "## Detailed Risks",
        ]
        for risk in payload["all_risks"]:
            lines.extend(
                [
                    f"### {risk['id']} [{risk['severity']}]",
                    f"- Score: {risk['score']}",
                    f"- Reason: {risk['reason']}",
                    f"- Likely Reject Phrase: {risk['likely_reject_phrase']}",
                    f"- Suggested Fix: {risk['fix_hint']}",
                    "",
                ]
            )

        lines.append("## Claim-Evidence Alignment")
        for row in payload["claim_evidence_matrix"]:
            lines.append(
                f"- {row['claim_id']} [{row['strength']}] score={row['score']} -> {len(row['evidence_refs'])} evidence refs"
            )

        lines.append("\n## Rebuttal Skeleton Included")
        lines.append("See `rebuttal.*` artifacts for per-review responses.")

        return "\n".join(lines).strip() + "\n"

    def _decision_json_zh(self, payload: dict) -> dict:
        return {
            "decision": self._decision_zh(payload["decision"]),
            "scores": payload["scores"],
            "top_risks": [self._risk_zh(r) for r in payload["top_risks"]],
            "top_remediation_tasks": [self._task_zh(t) for t in payload["top_remediation_tasks"]],
        }

    def _full_json_zh(self, payload: dict, ctx: PipelineContext) -> dict:
        return {
            "decision": self._decision_zh(payload["decision"]),
            "scores": payload["scores"],
            "all_risks": [self._risk_zh(r) for r in payload["all_risks"]],
            "claim_evidence_matrix": [self._alignment_zh(r) for r in payload["claim_evidence_matrix"]],
            "remediation_tasks": [self._task_zh(t) for t in payload["remediation_tasks"]],
            "rebuttal": ctx.artifacts["rebuttal"]["zh"]["bundle"],
        }

    def _risk_zh(self, risk: dict) -> dict:
        row = dict(risk)
        row["reason"] = self._phrase_zh(row["reason"])
        row["likely_reject_phrase"] = self._phrase_zh(row["likely_reject_phrase"])
        row["fix_hint"] = self._phrase_zh(row["fix_hint"])
        return row

    def _task_zh(self, task: dict) -> dict:
        row = dict(task)
        row["title"] = self._task_title_zh(row["title"])
        row["expected_gain"] = self._phrase_zh(row.get("expected_gain", ""))
        row["protocol"] = [self._phrase_zh(x) for x in row.get("protocol", [])]
        if row.get("priority") == "high":
            row["priority"] = "高"
        elif row.get("priority") == "medium":
            row["priority"] = "中"
        return row

    def _alignment_zh(self, row: dict) -> dict:
        out = dict(row)
        strength_map = {
            "Strong": "强",
            "Medium": "中",
            "Weak": "弱",
            "None": "无",
        }
        out["strength"] = strength_map.get(out.get("strength", ""), out.get("strength", ""))
        return out

    @staticmethod
    def _decision_zh(value: str) -> str:
        mapping = {
            "Not Ready": "不建议投稿",
            "Borderline": "边界状态",
            "Ready": "可投稿",
        }
        return mapping.get(value, value)

    def _phrase_zh(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        mapping = {
            "Statistical significance evidence appears missing.": "统计显著性证据可能缺失。",
            "Reproducibility details are likely incomplete.": "可复现性细节可能不完整。",
            "Experimental evidence does not yet meet venue expectations.": "实验性证据尚未达到目标会议预期。",
            "Core claims are not sufficiently supported by rigorous evidence.": "核心主张尚未得到严格证据的充分支撑。",
            "Address this with a focused experiment or analysis update.": "通过补充针对性实验或分析更新来解决该问题。",
            "Add direct experiments and statistical validation tied to this claim.": "补充与该主张直接对应的实验和统计验证。",
            "Reduce rejection likelihood by strengthening claim-evidence linkage.": "通过强化主张与证据链条，降低拒稿风险。",
            "Define exact hypothesis and target claim.": "明确待验证假设和目标主张。",
            "Run comparison against strong baselines with identical settings.": "在相同设置下与强基线进行对比实验。",
            "Report mean/std over multiple seeds and significance tests.": "报告多随机种子的均值/方差及显著性检验结果。",
            "Add analysis of failures and limitations.": "补充失败案例分析与局限性讨论。",
            "Mitigate": "缓解",
        }
        if text in mapping:
            return mapping[text]
        no_period = text.rstrip(".")
        if no_period in mapping:
            return mapping[no_period]
        m = re.match(r"^Claim (C\d+) has Weak evidence support\.$", text)
        if m:
            return f"主张 {m.group(1)} 的证据支持较弱。"
        return self.translator.to_zh(text)

    def _task_title_zh(self, title: str) -> str:
        m = re.match(r"^Mitigate (RISK-\d+) - (P\d) risk$", title)
        if m:
            return f"缓解 {m.group(1)} - {m.group(2)} 风险"
        return self._phrase_zh(title)

    @staticmethod
    def _decision_md_zh(payload: dict) -> str:
        lines = [
            "# 投稿决策简报",
            "",
            f"决策：**{payload['decision']}**",
            "",
            "## 评分",
            f"- 新颖性：{payload['scores']['novelty']}",
            f"- 技术正确性：{payload['scores']['soundness']}",
            f"- 实验充分性：{payload['scores']['experiment']}",
            f"- 写作清晰度：{payload['scores']['clarity']}",
            f"- 总分：{payload['scores']['overall']}",
            "",
            "## 拒稿风险 Top",
        ]
        for risk in payload["top_risks"]:
            lines.append(f"- [{risk['severity']}] {risk['id']} ({risk['score']}): {risk['reason']}")

        lines.append("\n## 必补实验")
        for task in payload["top_remediation_tasks"]:
            lines.append(
                f"- {task['id']} ({task['priority']}, effort={task['effort']}): {task['title']}"
            )

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _full_md_zh(payload: dict) -> str:
        lines = [
            "# 完整评审报告",
            "",
            f"决策：**{payload['decision']}**",
            "",
            "## 详细风险",
        ]
        for risk in payload["all_risks"]:
            lines.extend(
                [
                    f"### {risk['id']} [{risk['severity']}]",
                    f"- 分数：{risk['score']}",
                    f"- 原因：{risk['reason']}",
                    f"- 可能拒稿话术：{risk['likely_reject_phrase']}",
                    f"- 建议修复：{risk['fix_hint']}",
                    "",
                ]
            )

        lines.append("## 主张-证据对齐")
        for row in payload["claim_evidence_matrix"]:
            lines.append(
                f"- {row['claim_id']} [{row['strength']}] score={row['score']} -> {len(row['evidence_refs'])} 条证据"
            )

        lines.append("\n## Rebuttal 草稿")
        lines.append("请查看 `rebuttal.*` 产物。")

        return "\n".join(lines).strip() + "\n"
