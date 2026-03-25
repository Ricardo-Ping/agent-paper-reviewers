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

## 适用场景
- 用户希望在投稿前进行“最苛刻审稿人视角”预演。
- 用户需要识别潜在拒稿理由，并得到可执行补救计划。
- 用户希望自动生成 rebuttal 初稿模板。

## 输入要求
1. 论文草稿路径（`pdf` 或 `md`）。
2. 目标会议与年份（如 `ICLR 2026`）。
3. 核心主张（创新点/贡献点）。
4. 实验资源约束（时间、GPU、最多补实验数等）。
5. 输出模式（`en` 或 `en_zh`）。

## 环境与依赖
- Python：`3.11.x`
- Conda 环境：`agent-paper-reviewers-cpu` 或 `agent-paper-reviewers-gpu`
- PDF 导出：`pandoc` + (`xelatex` / `lualatex` / `tectonic`)

建议初始化顺序：
1. `conda env create -f envs/environment.gpu.yml`
2. `conda activate agent-paper-reviewers-gpu`
3. `pip install -e .`
4. `python -m agent_paper_reviewers.cli doctor`

## 固定流程（Skill 驱动）
1. Intake
2. VenueProfileResolver
3. PaperParser
4. ClaimNormalizer
5. EvidenceIndexer
6. ClaimEvidenceAligner
7. CitationGraph
8. GapDetector
9. RiskRanker
10. RemediationPlanner
11. RebuttalComposer
12. ReportBuilder
13. ExporterAndQAGate

流程顺序以 `flow_config.yaml` 为准。

## MCP 能力定位
- Skill 负责定义“做什么流程、按什么顺序做”。
- MCP 负责提供“具体工具能力”（例如 OpenReview 规则解析）。

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

- 从模板创建自定义 PDF 输入：
```bash
copy examples\\pdf_input.template.json input.json
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
- `decision_brief.en.md/json/pdf`：短版投稿决策。
- `full_review.en.md/json/pdf`：逐条风险与证据对齐明细。
- `rebuttal.en.md/json/pdf`：逐 reviewer 的回应草稿。

双语镜像（`language_mode=en_zh`）：
- `decision_brief.zh.md/json/pdf`
- `full_review.zh.md/json/pdf`
- `rebuttal.zh.md/json/pdf`

结构化与调试：
- `claim_evidence_matrix.json`：主张-证据锚点。
- `remediation_plan.json`：补救任务清单。
- `venue_profile_used.json`：会议规则快照。
- `skill_flow_used.json`：本次执行流程。
- `mcp_runtime.json`：MCP backend/provider 与能力开关。
- `run_summary.json`：状态摘要。
- `run_result.json`：完整运行状态。
- `artifacts/`：中间产物。

## 运行状态语义
- `success`：所有配置产物生成成功。
- `partial_failed`：主产物生成成功，但可选环节失败（常见 PDF 导出失败）。
- `failed`：流程在核心产物生成前中断。

## 常见故障处理
- PDF 失败：先 `doctor`，检查 `pandoc`/LaTeX 引擎。
- 中文显示异常：确认编辑器使用 UTF-8 打开。
- `policy_needs_manual_check=true`：动态规则解析失败，已回退本地规则。

## 参考文件
- 流程配置：`flow_config.yaml`
- 规则参考：`references/venue-rubrics.md`
- 输出契约：`references/report-contract.md`

