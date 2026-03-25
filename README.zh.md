# agent-paper-reviewers（中文）

`agent-paper-reviewers` 是一个投稿前“审稿人模拟”系统，采用 **Skill 驱动流程 + MCP 提供工具能力** 的架构。

## 文档入口
- 英文输出文件说明：`OUTPUT_FILES.en.md`
- 中文输出文件说明：`OUTPUT_FILES.zh.md`
- 英文 README：`README.en.md`
- Skill 流程配置：`agent-paper-reviewers-skill/flow_config.yaml`

## 架构说明
- Skill 驱动流程：流水线步骤顺序由 `agent-paper-reviewers-skill/flow_config.yaml` 控制。
- MCP 能力层：具体工具能力（如 OpenReview rebuttal 规则解析）由 `agent_paper_reviewers.mcp` provider 提供。
- 执行器层：LLM/Agent 后端通过 `ExecutorAdapter` 接入。

## 功能
- 解析论文草稿（`pdf` / `md`）
- 主张-证据对齐
- 缺失项检测（baseline / 显著性 / 消融 / 误差分析）
- 拒稿风险分级（`P0/P1/P2`）
- 生成补救实验与 rebuttal 初稿
- 固定导出 `MD + JSON + PDF`

## 运行选项
- `language_mode`：`en` | `en_zh`
- `executor_backend`：`codex|agent_api|openai|anthropic|qwen|local_vllm`
- `mcp_backend`：`http|disabled`
- `always_export_pdf`：`true|false`

## 输出目录
结果固定写入：

```text
output/<论文标题>/
```

同一论文标题再次运行会覆盖该目录（先清空再写入）。

## 编码与 PDF 可读性
- Markdown 与 JSON 统一为 UTF-8 编码。
- PDF 通过 `pandoc` 生成，并按 `xelatex -> lualatex -> tectonic` 自动回退。
- 中文 PDF 显示质量依赖系统可用中文字体（默认 `Microsoft YaHei`）。

## 快速运行
```bash
python -m agent_paper_reviewers.cli doctor
python -m agent_paper_reviewers.cli run --input examples/sample_input.json --output-dir output
```

## 主要命令
- `doctor`：检查依赖（`pandoc`、LaTeX 引擎、conda）
- `run`：执行 Skill 配置的流水线
- `refresh-venue`：写入每月 venue 规则更新提醒

