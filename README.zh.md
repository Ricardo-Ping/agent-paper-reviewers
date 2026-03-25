# reviewers-sim（中文）

`reviewers-sim` 是一个投稿前“审稿人模拟”流水线系统。

## 文档入口
- 英文输出文件说明：`OUTPUT_FILES.en.md`
- 中文输出文件说明：`OUTPUT_FILES.zh.md`
- 英文 README：`README.en.md`

## 功能
- 解析论文草稿（`pdf` / `md`）
- 做主张-证据对齐
- 检测缺失项（baseline / 显著性 / 消融 / 误差分析）
- 拒稿风险分级（`P0/P1/P2`）
- 生成补救实验与 rebuttal 初稿
- 固定导出 `MD + JSON + PDF`

## 语言模式
- `en`：只输出英文
- `en_zh`：同时输出英文和中文（镜像产物）

## 输出目录
结果固定写入：

```text
output/<论文标题>/
```

同一论文标题再次运行会覆盖该目录（先清空再写入）。

## 编码与 PDF 可读性
- Markdown 与 JSON 统一为 UTF-8 编码。
- PDF 通过 `pandoc` 生成，并按 `xelatex -> lualatex -> tectonic` 自动回退。
- 中文 PDF 的显示质量依赖系统可用中文字体（本项目默认使用 `Microsoft YaHei`）。

## 快速运行
```bash
python -m reviewers_sim.cli doctor
python -m reviewers_sim.cli run --input examples/sample_input.json --output-dir output
```

## 主要命令
- `doctor`：检查依赖（`pandoc`、LaTeX 引擎、conda）
- `run`：执行 12 步流水线
- `refresh-venue`：写入每月 venue 规则更新提醒

## 架构建议：MCP + Skill
可以，而且很适合这个项目。

建议分层：
- Skill 层：定义流程、提示词、输出契约、执行规范。
- MCP/工具层：提供具体能力（OpenReview 拉取、文档解析、存储、模型调用）。

简化理解：Skill 负责“怎么做流程”，MCP 负责“能调用什么能力”。
