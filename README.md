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

## 给 Agent 的一句话指令
可以直接对 Agent 说：

```text
请在当前仓库安装并初始化 agent-paper-reviewers：创建并激活 conda 环境 agent-paper-reviewers-gpu，执行 pip install -e .，运行 python -m agent_paper_reviewers.cli doctor，并用 examples/sample_input.json 跑一次验证。
```

如果你要让 Agent 直接跑你的 PDF，可以说：

```text
请帮我基于这篇 PDF 创建 input.json（paper.format=pdf，paper.path=绝对路径），然后运行 python -m agent_paper_reviewers.cli run --input <input.json> --output-dir output，并汇总输出目录中的关键结论。
```

如果 PDF 是你直接发在 Agent 对话窗口、但还没保存到本地，可以说：

```text
请先把我刚上传的 PDF 保存到当前仓库 input_files/paper.pdf，然后基于这个路径生成 input.json 并运行评审流程。
```

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
