# agent-paper-reviewers

中文 | [English](README.en.md)

`agent-paper-reviewers` 是一个用于投稿前拒稿演练的审稿人模拟系统，采用 **Skill 驱动流程 + MCP 提供工具能力** 架构。

## 项目目标
给定论文草稿（PDF/Markdown）、目标会议与年份、核心主张和实验资源约束，自动输出：
- 审稿维度评分（新颖性、技术正确性、实验充分性、写作清晰度）
- 按严重度排序的拒稿风险列表（P0/P1/P2）
- 必补实验清单（优先级、工作量、预期收益）
- Rebuttal 初稿（按 reviewer concern 逐条回应）

## 架构设计
- Skill 层：定义固定流程与规则约束，流程顺序来自 `agent-paper-reviewers-skill/flow_config.yaml`
- MCP 层：提供具体工具能力（如 OpenReview 规则解析）
- Executor 层：对接模型/Agent 执行后端（`codex/openai/anthropic/qwen/local_vllm`）

## 核心能力
- 解析论文结构与关键章节（方法、实验、消融、局限）
- 主张-证据对齐检查（Claim-Evidence Alignment）
- 缺失检测（baseline、显著性、消融、误差分析）
- 风险分级与补救动作生成
- 双语输出（`en` 或 `en_zh`）
- 固定导出 `MD + JSON + PDF`

## 快速开始
```bash
python -m agent_paper_reviewers.cli doctor
python -m agent_paper_reviewers.cli run --input examples/sample_input.json --output-dir output
```

## 输入参数（核心）
`options` 支持：
- `language_mode`: `en` | `en_zh`
- `executor_backend`: `codex|agent_api|openai|anthropic|qwen|local_vllm`
- `mcp_backend`: `http|disabled`
- `always_export_pdf`: `true|false`

## 输出目录
所有产物写入：

```text
output/<paper_title>/
```

同一篇论文重复运行会覆盖该目录（先清空后写入）。

## 输出文件说明
### 核心报告文件
- `decision_brief.en.md/json/pdf`：短版决策报告（是否建议投稿 + Top 风险 + 必补实验）
- `full_review.en.md/json/pdf`：长版评审报告（逐条风险、对齐矩阵、修复建议）
- `rebuttal.en.md/json/pdf`：Rebuttal 草稿（按 reviewer concern 组织）

若 `language_mode=en_zh`，会额外生成中文镜像：
- `decision_brief.zh.md/json/pdf`
- `full_review.zh.md/json/pdf`
- `rebuttal.zh.md/json/pdf`

### 结构化与调试文件
- `claim_evidence_matrix.json`：主张与证据锚点映射
- `remediation_plan.json`：补救实验任务清单
- `venue_profile_used.json`：本次运行采用的会议规则快照
- `skill_flow_used.json`：本次实际执行的 Skill 流程顺序
- `mcp_runtime.json`：本次 MCP backend/provider 与能力开关
- `run_summary.json`：运行状态摘要
- `run_result.json`：完整运行状态对象
- `artifacts/`：中间产物（用于排查与回溯）

## 状态语义
- `success`：配置产物全部生成成功
- `partial_failed`：主产物已生成，但可选环节失败（常见为 PDF 引擎问题）
- `failed`：流程在核心产物生成前失败

## 文档与配置
- Skill 文档：`agent-paper-reviewers-skill/SKILL.md`
- Skill 流程配置：`agent-paper-reviewers-skill/flow_config.yaml`
- 产品规格：`docs/product-spec.md`
- 输入 Schema：`docs/output-schema.json`
