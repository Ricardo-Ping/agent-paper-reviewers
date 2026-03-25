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
- 生成风险分级（P0/P1/P2）与拒稿话术。
- 生成补救实验计划（优先级、工作量、预期收益）。
- 生成 rebuttal 初稿（按 reviewer concern 逐条回应）。
- 按配置导出单语或双语产物（`en` / `en_zh`）。

## 架构思路
- Skill 驱动流程：流程顺序由 `agent-paper-reviewers-skill/flow_config.yaml` 定义。
- MCP 提供工具能力：例如 OpenReview 规则解析能力通过 MCP provider 注入。
- Executor 可插拔：支持 `codex|agent_api|openai|anthropic|qwen|local_vllm`。

## 环境要求
- 操作系统：Windows / Linux（GPU 推理建议 Linux + CUDA 12.1）。
- Python：`3.11.x`。
- 包管理：Conda（推荐）。
- PDF 导出：`pandoc` + (`xelatex` 或 `lualatex` 或 `tectonic`)。
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
```

## 5 分钟跑通
```bash
python -m agent_paper_reviewers.cli run --input examples/sample_input.json --output-dir output
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
- `claims`: 核心主张列表。
- `constraints`: 时间、GPU、最多补实验数等资源边界。
- `options.language_mode`: `en` 或 `en_zh`。
- `options.executor_backend`: 执行器后端。
- `options.mcp_backend`: `http` 或 `disabled`。
- `options.always_export_pdf`: 是否强制导出 PDF。

## 输出文件说明
### 核心报告
- `decision_brief.en.md/json/pdf`: 短版决策报告（投稿建议 + Top 风险 + 必补实验）。
- `full_review.en.md/json/pdf`: 长版评审报告（逐条风险、证据对齐、修复建议）。
- `rebuttal.en.md/json/pdf`: rebuttal 草稿包（按 reviewer concern 组织）。

### 双语镜像
- 当 `language_mode=en_zh` 时，额外生成：
- `decision_brief.zh.md/json/pdf`
- `full_review.zh.md/json/pdf`
- `rebuttal.zh.md/json/pdf`

### 结构化与调试产物
- `claim_evidence_matrix.json`: 主张-证据锚点映射。
- `remediation_plan.json`: 补救实验任务清单。
- `venue_profile_used.json`: 本次运行使用的会议规则快照。
- `skill_flow_used.json`: 本次实际执行的 Skill 流程。
- `mcp_runtime.json`: 本次 MCP backend/provider 与能力开关。
- `run_summary.json`: 运行状态摘要。
- `run_result.json`: 完整运行状态对象。
- `artifacts/`: 中间产物（排查和复盘用）。

## 运行状态语义
- `success`: 所有配置产物生成成功。
- `partial_failed`: 主产物生成成功，但可选环节失败（常见为 PDF 引擎缺失）。
- `failed`: 在核心产物生成前流程中断。

## 常见问题
- PDF 导出失败：先运行 `doctor`，确认 `pandoc` 和 LaTeX 引擎可用。
- 中文显示异常：请用 UTF-8 打开 Markdown/JSON 文件。
- 规则标记 `policy_needs_manual_check=true`：表示动态规则解析失败，已回退本地规则。

## 项目结构
```text
agent_paper_reviewers/              # 核心代码
agent-paper-reviewers-skill/        # Skill 定义与流程配置
data/venue_rules/                   # 会议规则库
envs/                               # conda 环境定义
docs/                               # 规格与 schema
examples/                           # 示例输入
tests/                              # 自动化测试
```

## 参考与致谢
文档组织方式参考了优秀开源项目常见结构（中英入口、快速开始、输出契约、FAQ）：
- OpenClaw Lark README: https://github.com/larksuite/openclaw-lark
- FastAPI README: https://github.com/fastapi/fastapi
- LangChain README: https://github.com/langchain-ai/langchain
