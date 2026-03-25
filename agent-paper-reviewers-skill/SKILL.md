---
name: agent-paper-reviewers
description: 用于投稿前拒稿演练的审稿模拟 Skill。输入论文草稿、目标会议和主张，输出风险分级、补救实验与 rebuttal 初稿。
---

# agent-paper-reviewers Skill（中文）

## 适用场景
- 用户希望在投稿前进行“最苛刻审稿人视角”预演。
- 用户需要识别潜在拒稿理由，并得到可执行补救计划。
- 用户希望自动生成 rebuttal 初稿模板。

## 必要输入
1. 论文草稿路径（`pdf` 或 `md`）。
2. 目标会议与年份（如 `NeurIPS 2025`）。
3. 核心主张（创新点/贡献点）。
4. 实验资源约束（时间、GPU、最多补实验数等）。
5. 输出模式（`en` 或 `en_zh`）。

## 固定流程（Skill 驱动）
1. Intake
2. VenueProfileResolver
3. PaperParser
4. ClaimNormalizer
5. EvidenceIndexer
6. ClaimEvidenceAligner
7. GapDetector
8. RiskRanker
9. RemediationPlanner
10. RebuttalComposer
11. ReportBuilder
12. ExporterAndQAGate

流程顺序以 `flow_config.yaml` 为准。

## MCP 能力定位
- Skill 负责定义“做什么流程、按什么顺序做”。
- MCP 负责提供“具体工具能力”（例如 OpenReview 规则解析）。

## 常用命令
- 环境检查：
```bash
python -m agent_paper_reviewers.cli doctor
```

- 运行流程：
```bash
python -m agent_paper_reviewers.cli run --input <input.json> --output-dir output
```

- 刷新规则变更记录：
```bash
python -m agent_paper_reviewers.cli refresh-venue
```

## 参考文档
- 规则参考：`references/venue-rubrics.md`
- 输出契约：`references/report-contract.md`
- 流程配置：`flow_config.yaml`
