from __future__ import annotations

import re

from ..models import RebuttalBundle, RebuttalItem
from ..services.translator import Translator
from .base import PipelineContext, PipelineStep


class RebuttalComposerStep(PipelineStep):
    name = "RebuttalComposer"

    def __init__(self, translator: Translator) -> None:
        self.translator = translator

    def run(self, ctx: PipelineContext) -> None:
        profile = ctx.artifacts["venue_profile"]["profile"]
        policy = profile["rebuttal_policy"]
        risks = ctx.artifacts["risk_ranking"]["risks"][:5]

        char_limit = int(policy.get("per_review_char_limit", 2500))
        mode = policy.get("mode", "per_review_only")
        global_mode = mode in {"global+per_review", "per_review_plus_global"}

        items = []
        for idx, risk in enumerate(risks, start=1):
            concern = risk["reason"]
            response = (
                "Thank you for this concern. We agree this point requires stronger support. "
                "We will add focused experiments and clearer statistical reporting to directly validate the claim."
            )
            new_evidence = [
                "New baseline comparison table with matched training budget.",
                "Seeded runs with mean/std and significance analysis.",
            ]
            paper_change = "Update Experiments, Ablation, and Limitations sections with explicit references."

            composed = f"Concern: {concern}\nResponse: {response}\nNew evidence: {'; '.join(new_evidence)}\nPaper change: {paper_change}"
            if len(composed) > char_limit:
                overflow = len(composed) - char_limit
                response = response[:-overflow] if overflow < len(response) else response[: max(0, char_limit // 4)]
                composed = f"Concern: {concern}\nResponse: {response}\nNew evidence: {'; '.join(new_evidence)}\nPaper change: {paper_change}"

            items.append(
                RebuttalItem(
                    review_id=f"R{idx}",
                    concern=concern,
                    response=response,
                    new_evidence=new_evidence,
                    paper_change=paper_change,
                    char_count=len(composed),
                    char_limit=char_limit,
                )
            )

        global_response = None
        if global_mode:
            global_response = (
                "We appreciate all reviews and will strengthen evidence quality, add statistical significance "
                "reporting, and clarify contribution boundaries in the revision."
            )

        bundle_en = RebuttalBundle(
            venue=ctx.input_data.venue.name,
            year=ctx.input_data.venue.year,
            mode=mode,
            items=items,
            global_response=global_response,
            attachment_pdf=None,
        )

        md_en = self._to_markdown(bundle_en)

        payload = {
            "en": {
                "bundle": bundle_en.model_dump(),
                "markdown": md_en,
            }
        }

        if ctx.input_data.options.language_mode.value == "en_zh":
            bundle_zh = self._translate_bundle(bundle_en)
            payload["zh"] = {
                "bundle": bundle_zh.model_dump(),
                "markdown": self._to_markdown_zh(bundle_zh),
            }

        ctx.artifacts["rebuttal"] = payload
        ctx.dump_json("artifacts/rebuttal_bundle.en.json", payload["en"]["bundle"])
        if "zh" in payload:
            ctx.dump_json("artifacts/rebuttal_bundle.zh.json", payload["zh"]["bundle"])

    def _translate_bundle(self, bundle: RebuttalBundle) -> RebuttalBundle:
        translated_items = []
        for item in bundle.items:
            translated_items.append(
                RebuttalItem(
                    review_id=item.review_id,
                    concern=self._to_zh(item.concern),
                    response=self._to_zh(item.response),
                    new_evidence=[self._to_zh(x) for x in item.new_evidence],
                    paper_change=self._to_zh(item.paper_change),
                    char_count=item.char_count,
                    char_limit=item.char_limit,
                )
            )

        return RebuttalBundle(
            venue=bundle.venue,
            year=bundle.year,
            mode=bundle.mode,
            items=translated_items,
            global_response=self._to_zh(bundle.global_response) if bundle.global_response else None,
            attachment_pdf=bundle.attachment_pdf,
        )

    def _to_zh(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r"\s+", " ", text.strip())

        mapping = {
            "Thank you for this concern. We agree this point requires stronger support. We will add focused experiments and clearer statistical reporting to directly validate the claim.": "感谢审稿意见。我们同意该问题需要更强支撑。我们将补充有针对性的实验，并提供更清晰的统计报告，以直接验证该主张。",
            "New baseline comparison table with matched training budget.": "新增与训练预算对齐的基线对比表。",
            "Seeded runs with mean/std and significance analysis.": "补充多随机种子运行结果（均值/标准差）及显著性分析。",
            "Update Experiments, Ablation, and Limitations sections with explicit references.": "在 Experiments、Ablation 和 Limitations 章节加入明确对应的补充说明与引用。",
            "We appreciate all reviews and will strengthen evidence quality, add statistical significance reporting, and clarify contribution boundaries in the revision.": "感谢所有审稿人的意见。我们将在修订稿中提升证据质量、补充统计显著性报告，并进一步澄清贡献边界。",
            "Statistical significance evidence appears missing.": "统计显著性证据可能缺失。",
            "Reproducibility details are likely incomplete.": "可复现性细节可能不完整。",
            "Experimental evidence does not yet meet venue expectations.": "实验性证据尚未达到目标会议预期。",
            "Core claims are not sufficiently supported by rigorous evidence.": "核心主张尚未得到严格证据的充分支撑。",
            "Address this with a focused experiment or analysis update.": "通过补充针对性实验或分析更新来解决该问题。",
            "Add direct experiments and statistical validation tied to this claim.": "补充与该主张直接对应的实验和统计验证。",
        }
        if text in mapping:
            return mapping[text]

        m = re.match(r"^Claim (C\d+) has Weak evidence support\.?$", text)
        if m:
            return f"主张 {m.group(1)} 的证据支持较弱。"

        return self.translator.to_zh(text)

    @staticmethod
    def _to_markdown(bundle: RebuttalBundle) -> str:
        lines = [f"# Rebuttal Draft ({bundle.venue} {bundle.year})", ""]
        if bundle.global_response:
            lines.extend(["## Global Positioning", "", bundle.global_response, ""])

        for item in bundle.items:
            lines.extend(
                [
                    f"## Response to Reviewer {item.review_id}",
                    "",
                    "### Concern",
                    "",
                    item.concern,
                    "",
                    "### Response",
                    "",
                    item.response,
                    "",
                    "### New Evidence",
                    "",
                    *[f"- {x}" for x in item.new_evidence],
                    "",
                    "### Paper Changes",
                    "",
                    item.paper_change,
                    "",
                    f"Character Budget: {item.char_count} / {item.char_limit}",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _to_markdown_zh(bundle: RebuttalBundle) -> str:
        lines = [f"# Rebuttal 草稿（{bundle.venue} {bundle.year}）", ""]
        if bundle.global_response:
            lines.extend(["## 总体回应", "", bundle.global_response, ""])

        for item in bundle.items:
            lines.extend(
                [
                    f"## 对审稿人 {item.review_id} 的回应",
                    "",
                    "### 审稿意见",
                    "",
                    item.concern,
                    "",
                    "### 回应",
                    "",
                    item.response,
                    "",
                    "### 新增证据",
                    "",
                    *[f"- {x}" for x in item.new_evidence],
                    "",
                    "### 论文修改位置",
                    "",
                    item.paper_change,
                    "",
                    f"字符预算：{item.char_count} / {item.char_limit}",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"
