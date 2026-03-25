# 输出文件说明（中文）

所有结果都写入：

```text
output/<论文标题>/
```

## 核心产物
- `decision_brief.en.md/json/pdf`：短版投稿决策报告。
- `full_review.en.md/json/pdf`：长版逐条评审报告。
- `rebuttal.en.md/json/pdf`：rebuttal 草稿包。

若 `language_mode=en_zh`，还会生成中文镜像文件：
- `decision_brief.zh.md/json/pdf`
- `full_review.zh.md/json/pdf`
- `rebuttal.zh.md/json/pdf`

## 辅助产物
- `claim_evidence_matrix.json`：主张-证据对齐矩阵。
- `remediation_plan.json`：补救实验优先级清单。
- `venue_profile_used.json`：本次运行使用的 venue/year 规则快照。
- `run_summary.json`：运行状态摘要。
- `run_result.json`：运行状态对象。
- `artifacts/`：流水线中间产物（调试用）。

## 状态说明
- `success`：所有配置产物生成成功。
- `partial_failed`：主产物已生成，但部分可选环节失败（常见是缺 LaTeX 引擎导致 PDF 失败）。
- `failed`：主流程中断，核心产物未完整生成。
