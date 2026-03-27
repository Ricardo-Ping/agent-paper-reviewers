# agent-paper-reviewers

中文 | [English](README.en.md)

我做这个项目，是因为投稿前最痛苦的一件事，不是改字句，而是你不知道自己会被怎么拒。  
`agent-paper-reviewers` 的目标很直接：在你正式投稿前，先把“最苛刻审稿人视角”跑一遍，把风险、缺口和补救动作提前摊开。

## 为什么做这个项目
- 我希望把“被动等拒稿”变成“主动做拒稿演练”。
- 我不想只要泛泛建议，我需要可执行的实验清单和 rebuttal 草稿。
- 我希望这个流程可以复用，不依赖某个单一模型或单一 Agent。

## 这个项目能做什么
- 解析论文草稿（PDF/Markdown），抽取结构化章节。
- 做主张-证据对齐检查（Claim-Evidence Alignment）。
- 识别高频拒稿缺口：baseline、统计显著性、消融、误差分析、可复现性。
- 增加稿件风格结构检查：按 venue 检查 `introduction/method/experiments/discussion` 章节长度比例是否失衡。
- 增加术语一致性检查：检测 acronym 扩写漂移、术语多版本混用与命名不稳定问题。
- 生成风险分级（P0/P1/P2）与拒稿话术。
- 生成补救实验计划（优先级、工作量、预期收益）。
- 生成 rebuttal 初稿（按 reviewer concern 逐条回应）。
- 当你不确定投稿会议时，基于论文内容反向推荐 Top venue（匹配分 + 依据 + 主要缺口）。
- 增加评分杠杆分析（哪个维度最拖后腿、先改哪一维度能最快提高 overall）。
- 按配置导出单语或双语产物（`en` / `en_zh`）。

## 和常见工具的定位差异
很多工具都很有价值，但它们解决的问题不同。  
`agent-paper-reviewers` 的定位不是“替代所有工具”，而是补上“投稿前拒稿演练”这条最缺的链路。

| 类别 | 代表工具 | 他们擅长 | 常见盲区 | `agent-paper-reviewers` 的补位 |
|---|---|---|---|---|
| 语言润色 | Paperpal / Writefull / Trinka | 语法、表达、用词 | 只管语言，不判断论证是否会被拒 | 把“写得通顺”升级为“主张是否站得住” |
| 论文发现 / Q&A | Semantic Scholar / Elicit / Consensus | 找文献、问答检索 | 不做投稿风险分级 | 给出可执行拒稿风险（P0/P1/P2） |
| 投稿技术检查 | Paperpal Preflight / Typeset | 格式、字数、图表规范 | 偏格式，不评审主张证据 | 检查 claim-evidence 对齐与实验缺口 |
| 专业人工润稿 | Enago / AJE | 专家润稿与建议 | 价格高、周期长、结构化产物少 | 自动化、可复跑、可追踪的 JSON/MD/PDF 产物 |
| 学术 RAG | PaperQA / Keep | 跨论文问答与引用 | 偏问答，不输出系统化投稿诊断 | 输出“问题-原因-修复-影响”完整诊断报告 |
| 研究 Agent 助手 | Coauthor / AutoRF | 研究流程辅助 | 常聚焦生成，不聚焦拒稿演练闭环 | 从风险识别到补救计划到 rebuttal 一条链 |
| OpenReview 本身 | OpenReview | 真实审稿与讨论流程 | 那是正式审稿，不是投稿前训练 | 在投稿前先做“模拟严审 + 提前补洞” |

一句话总结：  
这个项目的核心独特性是把“审稿风险识别 -> 证据对齐 -> 补救实验 -> rebuttal 草稿 -> 自检”做成可重复执行的工程化流程。

## 架构思路
- Skill 驱动流程：流程顺序由 `agent-paper-reviewers-skill/flow_config.yaml` 定义。
- 纯 Skill 运行时：主流程使用本地规则与工具链。
- Executor 可插拔：支持 `codex|agent_api|openai|anthropic|qwen|local_vllm`。
- 未知会议自动降级：本地 `_fallback` 规则 -> executor 自举规则草案。

## 推荐工作方式（Agent 分析 + Skill 工具库）
现在推荐把本项目当成“工具层”，而不是强绑定单一内置分析逻辑：

1. 用 Skill 工具提取上下文：`tool-parse-paper` + `tool-venue-profile`。
2. 让上层 Agent 读论文并做语义分析：主张提炼、gap 识别、风险判断、rebuttal 草拟。
3. 用 Skill 工具做标准化输出：`tool-format-template` + `tool-format-student-pack`。

这样做的好处是：分析能力由你选择的 Agent 决定（Codex/OpenClaw/其他），而格式与流程资产保持稳定复用。

### 双人设使用策略（深度版）
**人设 A：Agent 编排者（自动化优先）**
- 用 `review-pdf --ai-summary --strict-quality` 作为标准入口。
- 首先读取 `AGENT_HANDOFF.json` 与 `ai_summary.json` 判断是否可继续。
- 若 `strict-quality` 非零退出，先处理模型后端/解析质量阻断，再做下一轮分析。
- 再读取 `PERSONA_PLAYBOOK.en.md`，按 “Agent Operator” 小节执行重跑命令与质量门禁。

**人设 B：研究生作者（改稿优先）**
- 先读 `STUDENT_BRIEF.*.md`，确认“今天先改哪 3 件事”。
- 再读 `PERSONA_PLAYBOOK.zh.md`（或 `.en.md`）确认“风险 -> 动作”的映射关系。
- 再按 `student_pack/*` 的 001 -> 002 -> 003 顺序推进。
- 最后参考 `full_review` 与 `diagnosis_report` 做细化补洞。

### 双人设协作闭环（建议固定）
1. Agent 先跑：`review-pdf --ai-summary --strict-quality`。
2. Agent 只交付 3 个入口给研究生：`STUDENT_BRIEF`、`PERSONA_PLAYBOOK`、`student_pack/002-action-items`。
3. 研究生改完后回填 `feedback_template.json` 并执行 `submit-feedback`。
4. Agent 基于反馈再跑一轮，检查 `run_result.json` 与 `pipeline_steps.json` 是否无阻断。

### Tool-Only 命令（给 Agent 调用）
```bash
# 1) 取会议规则（JSON）
python -m agent_paper_reviewers.cli tool-venue-profile --venue ICLR --year 2026 --json

# 2) 解析论文（JSON）
python -m agent_paper_reviewers.cli tool-parse-paper --paper-path /abs/path/paper.pdf --output parsed_paper.json

# 3) 导出分析模板（让 Agent 按这个结构回填）
python -m agent_paper_reviewers.cli tool-format-template --template student_pack_analysis --output analysis_template.json

# 4) 把 Agent 的分析结果格式化为 3 份研究生可读文档
python -m agent_paper_reviewers.cli tool-format-student-pack --analysis-json agent_analysis.json --output-dir output/student_pack/en --language en
```

此外，Skill 目录下提供了机器可读能力清单：
- `agent-paper-reviewers-skill/manifest.json`
- 建议 Agent 先读取 manifest 再决定调用路径。

兼容说明：`run --input ...` 的全流程模式仍然保留，适合一键跑通；上面是更推荐的“Agent 主导分析”模式。

## 给 Agent 的一句话指令
可以直接对 Agent 说：

```text
请在当前仓库安装并初始化 agent-paper-reviewers：创建并激活 conda 环境 agent-paper-reviewers-gpu，执行 pip install -e .，运行 python -m agent_paper_reviewers.cli doctor，并用 examples/sample_input.json 跑一次验证。
```

如果你要让 Agent 直接跑你的 PDF，可以说：

```text
请直接用 review-pdf 命令跑这篇论文（不用手写 input.json）：python -m agent_paper_reviewers.cli review-pdf --paper-path <绝对路径> --venue ICLR --year 2026 --output-dir output --ai-summary --strict-quality，并汇总输出目录中的关键结论。
```

如果 PDF 是你直接发在 Agent 对话窗口、但还没保存到本地，可以说：

```text
请先把我刚上传的 PDF 保存到当前仓库 input_files/paper.pdf，然后用 review-pdf 命令直接运行评审流程。
```

## 环境要求
- 操作系统：Windows / Linux（GPU 推理建议 Linux + CUDA 12.1）。
- Python：`3.11.x`。
- 包管理：Conda（推荐）。
- PDF 导出（可选）：`pandoc` + (`xelatex` 或 `lualatex` 或 `tectonic`)。
- 可选网络：用于 OpenReview 动态规则解析与在线翻译回退。

## 安装
### 1. 克隆仓库
```bash
git clone https://github.com/Ricardo-Ping/agent-paper-reviewers.git
cd agent-paper-reviewers
```

### 2. 创建 Conda 环境
CPU：
```bash
conda env create -f envs/environment.cpu.yml
conda activate agent-paper-reviewers-cpu
```

GPU：
```bash
conda env create -f envs/environment.gpu.yml
conda activate agent-paper-reviewers-gpu
```

### 3. 安装项目（开发模式）
```bash
pip install -e .
```

### 4. 运行环境自检
```bash
python -m agent_paper_reviewers.cli doctor
# 机器可读输出（用于自动化脚本/CI）
python -m agent_paper_reviewers.cli doctor --json
```

### 5. （可选）启用 PDF 导出工具链
默认环境不安装 PDF 工具链。只有在你需要 `*.pdf` 时再安装：
```bash
conda install -n agent-paper-reviewers-cpu -c conda-forge pandoc tectonic
# 或者安装到 GPU 环境
conda install -n agent-paper-reviewers-gpu -c conda-forge pandoc tectonic
```

## 5 分钟跑通
```bash
python -m agent_paper_reviewers.cli run --input examples/sample_input.json --output-dir output
# 如果你希望给 Agent 一个简洁机器摘要：
python -m agent_paper_reviewers.cli run --input examples/sample_input.json --output-dir output --ai-summary
```

也可以直接跑 PDF（无需手写 `input.json`）：
```bash
python -m agent_paper_reviewers.cli review-pdf \
  --paper-path /abs/path/paper.pdf \
  --venue ICLR \
  --year 2026 \
  --executor-backend codex \
  --language-mode en_zh \
  --output-dir output \
  --ai-summary \
  --strict-quality
```
该命令会自动写入 `generated_input.json` 到本次输出目录，便于复现和二次编辑。

说明：
- 这条命令会读取 `examples/sample_input.json`。
- 当前这个示例文件默认指向的是 `examples/sample_paper.md`（Markdown 示例），不是你的自定义 PDF。
- `paper.path` 支持绝对路径和相对路径；相对路径会自动按 input.json 所在目录解析。

如果你要直接跑自己的 PDF，有两种方式：

方式 1（推荐）：使用现成 PDF 示例输入
```bash
python -m agent_paper_reviewers.cli run --input examples/sql_translation_gpu_input.json --output-dir output
```

方式 2：新建你自己的 `input.json`，并确保：
```json
"paper": {
  "format": "pdf",
  "path": "你的PDF绝对路径"
}
```

也可以用模板文件：
- `examples/pdf_input.template.json`

如果你暂时不确定该投哪个会议，可以先跑“会议推荐”示例：
```bash
python -m agent_paper_reviewers.cli run --input examples/venue_recommend_input.json --output-dir output
```
该示例会输出 `venue_recommendations.json`，并在 `decision_brief/full_review` 中追加推荐会议与理由。

如果你处于审稿讨论期（已有 R1/R2/R3 意见），可用模板直接跑“讨论期模式”：
```bash
python -m agent_paper_reviewers.cli run --input examples/meta_review_input.template.json --output-dir output
```

运行结束后，结果会写到：
```text
output/<paper_title>/
```

同一篇论文重复运行会覆盖同名目录（先清空再写入）。

## 输入说明
`examples/sample_input.json` 结构如下：
- `paper`: 论文格式与路径（`pdf|md`）。
- `venue`: 会议名称与年份。
- `claims`: 核心主张列表（可为空；为空时会自动从摘要/贡献/结论段落发现候选主张）。
  - 当你不确定目标会议时，可先填占位会议名（如 `UnknownConf`），系统仍会输出 `venue_recommendations` 供你反向选会。
- `review_context.manuscript_stage`: 稿件阶段（关键分流开关）
  - `initial_submission`: 初次投稿前，关注“现在要不要投”
  - `rejected_after_reviews`: 已拒稿后，关注“能否挽救并复投”
  - `meta_review_discussion`: 讨论期，关注“逐条回应 reviewer concern”
- `review_context.reviewer_comments`: 可选审稿意见列表（`review_id + concern`），用于讨论期/拒稿后模式的风险重排和 rebuttal 对齐。
- `constraints`: 时间、GPU、最多补实验数等资源边界。
- `options.language_mode`: `en` 或 `en_zh`。
- `options.executor_backend`: 执行器后端。
- `options.always_export_pdf`: 是否导出 PDF（默认 `false`，仅输出 `md+json`）。
- `profile.author_hash` / `profile.author_id`: 可选投稿者标识（用于历史画像累计；`author_id` 会在本地转 hash）。
  - 未提供作者标识时，系统仍会按 `venue+year` 累计公共弱项画像。

## 执行器后端（真实调用）
- `agent_api`：`OpenClawNodeExecutor`，调用 `OpenClaw /api/sessions/spawn`。
- `openai`：OpenAI 兼容接口（`/v1/chat/completions`）。
- `anthropic`：Anthropic Messages API（`/v1/messages`）。
- `qwen`：Qwen OpenAI 兼容接口。
- `local_vllm`：本地 vLLM OpenAI 兼容接口。
- `codex`：走 OpenAI 兼容接口（可配置模型名）。

常用环境变量：
- `AGENT_PAPER_REVIEWERS_OPENCLAW_URL`（默认 `http://localhost:18789`）
- `OPENAI_API_KEY` / `OPENAI_BASE_URL`
- `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`
- `QWEN_API_KEY` / `QWEN_BASE_URL`
- `LOCAL_VLLM_BASE_URL` / `LOCAL_VLLM_API_KEY`
- `SEMANTIC_SCHOLAR_API_KEY`（用于 Citation Graph，避免匿名限流）
- `AGENT_PAPER_REVIEWERS_PROFILE_ROOT`（可选，覆盖默认 `./profiles` 历史画像存储目录）

## 输出文件说明
### 研究生优先入口（推荐先看）
- `START_HERE.md`：总入口，告诉你先看哪 3 个文件。
- `RUN_GUIDE.md`：运行状态 + 下一步建议（有阻断会直接给出解除方法）。
- `STUDENT_BRIEF.md`：研究生极简执行摘要（Top 阻断 + 前 24 小时动作）。
- `PERSONA_PLAYBOOK.md`：双人设执行手册（Agent 编排流 + 研究生改稿流）。
- `student_pack/en/001-submission-decision.md`：一页决策（是否建议投稿 + Top 阻断问题）。
- `student_pack/en/002-action-items.md`：可执行行动清单（按优先级、带证据锚点与实验工作量）。
- `student_pack/en/003-rebuttal-draft.md`：与风险映射的 rebuttal 草稿（逐 reviewer）。
- 双语模式下会同步生成 `student_pack/zh/*` 与 `START_HERE.zh.md`。

这套 `student_pack` 是给研究生“拿来就改”的人类可读入口；其余 JSON/中间文件主要用于调试、追溯和反馈闭环。

`ai_summary.json`（给 Agent 自动化）的关键字段：
- `degraded` / `degraded_reasons`：本次运行是否降级及原因
- `student_pack_ready`：三份 student pack 是否可直接使用
- `recommended_next_action`：推荐下一步动作
- `step_overview`：各 step 成功/失败/跳过计数
- `key_files`：关键文件路径索引（相对 `run_dir`，避免跨平台乱码）
- `persona_routes`：Agent-first 与 Student-first 建议读取路径
- 新增 `minimal_checks`：最小质量核查文件列表（自动化可直接据此做 gating）

`AGENT_HANDOFF.json`（给 Agent 的机器交接包）包含：
- 运行质量状态（是否有 executor warning / 是否会触发 strict 失败）
- Top 风险与 Top 动作清单
- 下一轮建议命令（含 `rerun_strict`）

### 核心报告
- `decision_brief.en.md/json`: 短版决策报告（投稿建议 + Top 风险 + 必补实验 + 各评分解释）。
- `full_review.en.md/json`: 长版评审报告（逐条风险、证据对齐、修复建议 + 各评分解释）。
  - 内含 `score_leverage_analysis`（维度权重、当前贡献、目标缺口、最快提升维度）。
- `diagnosis_report.en.md/json`: 面向研究生可读的诊断报告（问题 -> 原因 -> 修复 -> 影响解释）。
- `rebuttal.en.md/json`: rebuttal 草稿包（按 reviewer concern 组织）。
  - 现在会显式写入 `manuscript_stage`，并在讨论期优先按 `review_context.reviewer_comments` 逐条生成。
- 当 `options.always_export_pdf=true` 时，额外导出对应 `*.pdf` 文件。

### 双语镜像
- 当 `language_mode=en_zh` 时，额外生成：
- `decision_brief.zh.md/json`
- `full_review.zh.md/json`
- `diagnosis_report.zh.md/json`
- `rebuttal.zh.md/json`
- 当 `options.always_export_pdf=true` 时，额外导出对应 `*.pdf` 文件。

### 双语翻译质量说明
- 双语产物用于“阅读辅助与内部协作”，不等同于可直接对外提交的最终译稿。
- 翻译链路按回退顺序执行：术语表替换 -> GoogleTranslator -> MarianMT -> executor `translate_zh` -> 稳定回退（pseudo-translate）。
- 当你使用 deterministic executor、离线运行、或翻译后端不可用时，系统可能进入 pseudo-translate，中文内容会出现中英混杂或表达不自然。
- 正式投稿场景请以英文主稿为准，并对中文镜像进行人工校对。
- 如果你希望更高翻译质量：配置具备真实翻译能力的 executor backend，并保证网络/API 可用。

### 结构化与调试产物
- `claim_discovery.json`: 自动发现的主张候选、当前选择主张、确认建议。
- `claim_evidence_matrix.json`: 主张-证据锚点映射。
- `remediation_plan.json`: 补救实验任务清单。
- `venue_recommendations.json`: 当你不确定投稿会议时的推荐列表（匹配分、理由、通过/未通过检查项）。
- `feedback_template.json`: 风险反馈模板（每条 risk 可标记 `correct|incorrect|pending`）。
- `feedback_README.en.md` / `feedback_README.zh.md`: 反馈模板填写与提交说明。
- `venue_profile_used.json`: 本次运行使用的会议规则快照。
  - 包含 `required_check_specs`（可执行阈值）、`source/source_notes`（规则来源链路）。
- `skill_flow_used.json`: 本次实际执行的 Skill 流程。
- `runtime_context.json`：运行时能力上下文（当前固定为 `local_skill_tools_only`）。
- `pipeline_steps.json`: 每个 pipeline step 的执行状态轨迹（success/failed/skipped）。
- `run_summary.json`: 运行状态摘要。
- `run_result.json`: 完整运行状态对象（含每个 step 的 `success/failed/skipped` 与 `failed_step`，以及已落盘产物清单）。
- `historical_profile.json`: 本次运行后更新的历史画像快照（作者级 + venue/year 级弱项统计）。
- `artifacts/`: 中间产物（排查和复盘用）。
  - `artifacts/risk_ranking.json` 中新增 `stage_strategy`、`focus_risks`、`reviewer_comment_alignment`，用于解释“为什么该阶段优先这些风险”。
  - `artifacts/citation_graph.json` 的 `stats` 中新增 `content_novelty_score` 与 `content_novelty_components`，用于在低引用场景下从论文正文（intro/method/conclusion）估计 novelty 信号。

## 运行状态语义
- `success`: 所有配置产物生成成功。
- `partial_failed`: 主产物生成成功，但可选环节失败（常见为启用 PDF 导出后引擎缺失）。
- `failed`: 在核心产物生成前流程中断。

## 常见问题
- 默认为什么没有 PDF：默认关闭 PDF 导出，仅生成 `md+json`，便于直接复制到 Overleaf 或继续编辑。
- 如何开启 PDF：在 `input.json` 中设置 `"options": {"always_export_pdf": true}`。
- PDF 导出失败：先运行 `doctor`，检查 `pdf export capability (optional)` 这一行；或运行 `doctor --json` 看 `pdf_export.ready`。
- 出现 `pdf_parse_quality_*` 告警：说明 PDF 解析质量不足，建议确认原 PDF 是否包含文本层，或先做 OCR/转 Markdown 再评审。
- 中文显示异常：请用 UTF-8 打开 Markdown/JSON 文件。
- 诊断报告显示 `pseudo-translate`：说明当前走了翻译回退路径，中文仅供参考，建议启用真实翻译后端后重跑。
- 规则标记 `policy_needs_manual_check=true`：表示动态规则解析失败，已回退本地规则。
- `citation_graph_warning:semantic_scholar_status_429`：Semantic Scholar 匿名请求限流，建议配置 `SEMANTIC_SCHOLAR_API_KEY`。

## 反馈闭环（让系统持续变准）
运行后你可以把每条风险标记为“判断正确/错误”，下一次运行会自动参考历史反馈做风险分数校准。

1. 打开每次输出目录下的 `feedback_template.json`。
2. 将每条 `items[*].verdict` 改为 `correct` 或 `incorrect`（未确认可保留 `pending`）。
3. 可选填写 `comment` 说明误判原因。
4. 提交反馈：
```bash
python -m agent_paper_reviewers.cli submit-feedback --input output/<paper_title>/feedback_template.json
```
5. 反馈会落盘到：
```text
feedback/<venue>/<year>/
```
后续同会议年份运行会自动加载这些反馈信号并校准 `risk_ranking`。

## 项目结构
```text
agent_paper_reviewers/              # 核心代码
agent-paper-reviewers-skill/        # Skill 定义与流程配置
data/venue_rules/                   # 会议规则库（目录化）
  _fallback.yaml                    # 全局回退规则
  neurips/2024.yaml                 # 按会议+年份存放
  iclr/2026.yaml
  sigmod/2024.yaml
envs/                               # conda 环境定义
docs/                               # 规格与 schema
examples/                           # 示例输入
tests/                              # 自动化测试
```

## 会议规则刷新
```bash
python -m agent_paper_reviewers.cli refresh-venue --venue all --year 2026
```

常用参数：
- `--venue all`：刷新全部会议。
- `--venue iclr,icml`：只刷新指定会议。
- `--openreview-group <group_id>`：单会议时覆盖 OpenReview group。
- `--dry-run`：只预览，不写文件。


