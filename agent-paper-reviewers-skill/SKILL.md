---
name: agent-paper-reviewers
description: 用于投稿前拒稿演练的审稿模拟 Skill。输入论文草稿、目标会议和主张，输出风险分级、补救实验与 rebuttal 初稿。
---

# agent-paper-reviewers Skill（中文）

## Skill 目标
这个 Skill 用来把“投稿前自检”变成可重复的标准流程：
- 用最苛刻审稿人视角识别拒稿风险。
- 把抽象问题转成可执行的补救实验任务。
- 产出可直接改稿和准备 rebuttal 的文档。
- 在不确定投稿会议时，给出反向会议推荐（匹配分 + 依据 + 缺口）。
- 给出“评分杠杆分析”，明确先改哪一维度最能快速提高 overall。

## 给 Agent 的一句话指令
可以直接对 Agent 说：

```text
请在当前仓库安装并初始化 agent-paper-reviewers：创建并激活 conda 环境 agent-paper-reviewers-gpu，执行 pip install -e .，运行 python -m agent_paper_reviewers.cli doctor，并用 examples/sample_input.json 跑一次验证。
```

跑自定义 PDF 时可以说：

```text
请帮我基于这篇 PDF 创建 input.json（paper.format=pdf，paper.path=绝对路径），然后运行 python -m agent_paper_reviewers.cli run --input <input.json> --output-dir output，并汇总拒稿风险、必补实验和 rebuttal 草稿。
```

如果 PDF 来自对话窗口附件且尚未落盘，可以说：

```text
请先把我上传的 PDF 保存到当前仓库 input_files/paper.pdf，再基于该路径生成 input.json 并执行完整流程。
```

如果你不确定投哪个会议，可以说：

```text
请基于论文摘要和贡献先做会议推荐（Top 5），给出每个会议的匹配分、推荐理由和主要缺口；然后再按最匹配会议生成完整拒稿演练报告。
```

如果你已经进入审稿讨论期，可以说：

```text
我现在在 reviewer discussion 阶段，请按 R1/R2/R3 的 concern 逐条生成高针对性的 rebuttal，并把风险优先级按这些 concern 重排。
```

## 适用场景
- 用户希望在投稿前进行“最苛刻审稿人视角”预演。
- 用户需要识别潜在拒稿理由，并得到可执行补救计划。
- 用户希望自动生成 rebuttal 初稿模板。
- 用户暂时不确定投稿会议，希望系统先反向推荐 venue。

## 输入要求
1. 论文草稿路径（`pdf` 或 `md`）。
2. 目标会议与年份（如 `ICLR 2026`）。
3. 核心主张（创新点/贡献点，可为空；为空时自动发现候选主张）。
4. 实验资源约束（时间、GPU、最多补实验数等）。
5. 输出模式（`en` 或 `en_zh`）。
6. 可选投稿者标识（`profile.author_hash` 或 `profile.author_id`），用于历史弱项画像累计。
   - 如果不提供作者标识，系统仍会按 `venue+year` 维度累计公共弱项画像。
7. 可选稿件阶段上下文（`review_context`）：
   - `manuscript_stage`: `initial_submission | rejected_after_reviews | meta_review_discussion`
   - `reviewer_comments`: `review_id + concern` 列表（讨论期强烈建议提供）

## 环境与依赖
- Python：`3.11.x`
- Conda 环境：`agent-paper-reviewers-cpu` 或 `agent-paper-reviewers-gpu`
- PDF 导出（可选）：`pandoc` + (`xelatex` / `lualatex` / `tectonic`)

建议初始化顺序：
1. `conda env create -f envs/environment.gpu.yml`
2. `conda activate agent-paper-reviewers-gpu`
3. `pip install -e .`
4. `python -m agent_paper_reviewers.cli doctor`

可选：如果需要 PDF 导出，再安装：
```bash
conda install -n agent-paper-reviewers-gpu -c conda-forge pandoc tectonic
# CPU 环境同理
conda install -n agent-paper-reviewers-cpu -c conda-forge pandoc tectonic
```

## 固定流程（Skill 驱动）
1. Intake
2. VenueProfileResolver
3. PaperParser
4. ClaimDiscoverer
5. ClaimNormalizer
6. EvidenceIndexer
7. ClaimEvidenceAligner
8. CitationGraph
9. GapDetector
10. VenueRecommender
11. RiskRanker
12. RemediationPlanner
13. RebuttalComposer
14. ReportBuilder
15. ExporterAndQAGate

流程顺序以 `flow_config.yaml` 为准。

## MCP 能力定位
- Skill 负责定义“做什么流程、按什么顺序做”。
- MCP 负责提供“具体工具能力”（例如 OpenReview 规则解析）。
- 未知会议处理：优先本地 `_fallback`，再尝试 OpenReview 动态发现，最后由 executor 自举可执行规则草案。

## 执行器后端（真实调用）
- `agent_api` -> OpenClawNodeExecutor（`/api/sessions/spawn`）
- `openai` / `codex` -> OpenAI 兼容接口
- `anthropic` -> Anthropic Messages API
- `qwen` -> Qwen OpenAI 兼容接口
- `local_vllm` -> 本地 vLLM OpenAI 兼容接口

常用环境变量：
- `AGENT_PAPER_REVIEWERS_OPENCLAW_URL`
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`
- `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`
- `QWEN_API_KEY`, `QWEN_BASE_URL`
- `LOCAL_VLLM_BASE_URL`, `LOCAL_VLLM_API_KEY`
- `SEMANTIC_SCHOLAR_API_KEY`（Citation Graph 检索）

## 运行命令
- 环境检查：
```bash
python -m agent_paper_reviewers.cli doctor
```

- 跑 Markdown 示例：
```bash
python -m agent_paper_reviewers.cli run --input examples/sample_input.json --output-dir output
```

- 跑 PDF 示例：
```bash
python -m agent_paper_reviewers.cli run --input examples/sql_translation_gpu_input.json --output-dir output
```

- 跑“未知会议推荐”示例：
```bash
python -m agent_paper_reviewers.cli run --input examples/venue_recommend_input.json --output-dir output
```

- 跑“讨论期逐条回复”示例模板：
```bash
python -m agent_paper_reviewers.cli run --input examples/meta_review_input.template.json --output-dir output
```

- 从模板创建自定义 PDF 输入：
```bash
copy examples\\pdf_input.template.json input.json
```

- 提交风险反馈（形成反馈闭环）：
```bash
python -m agent_paper_reviewers.cli submit-feedback --input output/<paper_title>/feedback_template.json
```

- 刷新规则变更记录：
```bash
python -m agent_paper_reviewers.cli refresh-venue --venue all --year 2026
```

## 重要说明（避免误解）
- `examples/sample_input.json` 默认指向 `examples/sample_paper.md`，不是用户自己的 PDF。
- `paper.path` 支持绝对路径和相对路径；相对路径会自动按 input.json 所在目录解析。
- 要跑用户 PDF，必须在 input 里设置：
```json
"paper": {
  "format": "pdf",
  "path": "用户PDF绝对路径"
}
```

## 输出文件与作用
核心报告：
- `decision_brief.en.md/json`：短版投稿决策（含各评分解释）。
- `full_review.en.md/json`：逐条风险与证据对齐明细（含各评分解释）。
  - 包含 `score_leverage_analysis`（维度权重、当前贡献、目标缺口、优先提升维度）。
- `diagnosis_report.en.md/json`：可读性更强的诊断报告（问题->原因->修复->影响）。
- `rebuttal.en.md/json`：逐 reviewer 的回应草稿。
  - 包含 `manuscript_stage`，讨论期会优先按 `reviewer_comments` 组织。
- 当 `options.always_export_pdf=true` 时，额外生成对应 `*.pdf`。

双语镜像（`language_mode=en_zh`）：
- `decision_brief.zh.md/json`
- `full_review.zh.md/json`
- `diagnosis_report.zh.md/json`
- `rebuttal.zh.md/json`
- 当 `options.always_export_pdf=true` 时，额外生成对应 `*.pdf`。

结构化与调试：
- `claim_discovery.json`：自动发现主张候选与确认建议。
- `claim_evidence_matrix.json`：主张-证据锚点。
- `remediation_plan.json`：补救任务清单。
- `venue_recommendations.json`：会议推荐列表（匹配分、理由、通过/未通过检查项）。
- `feedback_template.json`：风险反馈模板（每条风险可标记 `correct|incorrect|pending`）。
- `feedback_README.en.md` / `feedback_README.zh.md`：反馈模板填写与提交说明。
- `venue_profile_used.json`：会议规则快照。
  - 含 `required_check_specs`（阈值规则）和 `source/source_notes`（规则来源）。
- `skill_flow_used.json`：本次执行流程。
- `mcp_runtime.json`：MCP backend/provider 与能力开关。
- `pipeline_steps.json`：step 级执行轨迹（success/failed/skipped）。
- `run_summary.json`：状态摘要。
- `run_result.json`：完整运行状态（含每个 step 的 `success/failed/skipped`、`failed_step` 和已产生产物清单）。
- `historical_profile.json`：本次运行更新后的历史画像（作者级 + 会议/年份级弱项统计）。
- `artifacts/`：中间产物。
  - `artifacts/risk_ranking.json` 现在含 `stage_strategy`、`focus_risks`、`reviewer_comment_alignment`，用于解释阶段化策略。
  - `artifacts/citation_graph.json` 的 `stats` 包含 `content_novelty_score` 与 `content_novelty_components`，用于低引用论文的正文 novelty 估计。

## 运行状态语义
- `success`：所有配置产物生成成功。
- `partial_failed`：主产物生成成功，但可选环节失败（常见为启用 PDF 导出后工具链缺失）。
- `failed`：流程在核心产物生成前中断。

## 常见故障处理
- 默认为什么没有 PDF：默认关闭 PDF 导出，仅生成 `md+json`，便于直接复制到 Overleaf。
- 如何开启 PDF：在 input 配置中设置 `"options": {"always_export_pdf": true}`。
- PDF 失败：先 `doctor`，检查 `pandoc`/LaTeX 引擎。
- 出现 `pdf_parse_quality_*`：说明解析质量偏低，先确认 PDF 文本层，必要时先 OCR 或转 Markdown 再跑。
- 中文显示异常：确认编辑器使用 UTF-8 打开。
- `policy_needs_manual_check=true`：动态规则解析失败，已回退本地规则。

## 反馈闭环说明
每次 run 结束都会输出 `feedback_template.json`。用户可逐条标记风险判断是否正确：
- `correct`：系统判断正确。
- `incorrect`：系统判断错误（建议补充 comment）。
- `pending`：暂不评估。

提交后反馈会写入 `feedback/<venue>/<year>/`。后续同会议年份运行时，RiskRanker 会优先读取这些历史反馈并校准风险分数与优先级。

## 参考文件
- 流程配置：`flow_config.yaml`
- 规则参考：`references/venue-rubrics.md`
- 输出契约：`references/report-contract.md`

