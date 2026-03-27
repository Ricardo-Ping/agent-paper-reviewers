# agent-paper-reviewers Product Spec（实现对齐版）

Last Updated: 2026-03-26  
Version: v1.0（与 `main` 当前实现对齐）

## 0. 文档目的

这份文档不是宣传页，而是工程入口文档。它回答 5 个核心问题：

1. 这个项目解决什么问题。
2. 系统到底跑了哪些步骤。
3. 每个步骤产出什么数据。
4. 出错时会发生什么、如何降级。
5. 接入方应该读哪些文件、怎么判断结果可用性。

配套机器契约见：`docs/output-schema.json`。

---

## 1. 项目定位

`agent-paper-reviewers` 是一个“投稿前拒稿演练系统”。

它的核心能力不是语言润色，而是把论文当作审稿稿件，围绕“主张-证据-风险-补救-rebuttal”做结构化推演，目标是：

1. 提前暴露高风险拒稿点（P0/P1/P2）。
2. 给出可执行的补实验与修改方案（受时间/GPU预算约束）。
3. 生成可提交前复核的 rebuttal 草稿与预审结果。
4. 形成可追踪的运行审计（step 状态、QA issue、失败边界）。

---

## 2. 目标用户与使用场景

### 2.1 目标用户

1. 研究生/博士生：投稿前质量把关。
2. 课题组负责人：快速识别稿件短板与补救优先级。
3. 研究工程团队：把审稿演练流程接入 CI/自动化。

### 2.2 稿件阶段（`review_context.manuscript_stage`）

1. `initial_submission`：是否现在就投。
2. `rejected_after_reviews`：大修后是否具备重投条件。
3. `meta_review_discussion`：针对已收到 reviewer concern 做逐条闭环。

---

## 3. 总体架构

### 3.1 架构分层

1. Skill 流程层：`agent-paper-reviewers-skill/flow_config.yaml`。
2. 编排层：`agent_paper_reviewers/orchestrator.py`。
3. 执行器层：`ExecutorAdapter`（OpenClaw / OpenAI-compatible / Anthropic / local_vllm / deterministic fallback）。
4. 规则层：本地 venue 规则快照（`data/venue_rules/*`）与未知会议 executor 自举补充。
5. 产物层：统一输出 `MD + JSON + PDF(可选)`。

### 3.2 推荐运行模式（Agent 主导分析）

建议把本项目当“工具库”：

1. 用代码工具做确定性任务：venue 规则解析、论文文本提取、格式化输出。
2. 用上层 Agent 做语义任务：读论文、识别 gaps、风险排序、rebuttal 生成。
3. 再调用工具层输出标准化文档。

对应 CLI 工具命令：

1. `tool-venue-profile`
2. `tool-parse-paper`
3. `tool-format-template`
4. `tool-format-student-pack`
5. `review-pdf`（一键 PDF 跑通，不必手写 input.json）

说明：完整 pipeline（`run --input ...`）仍保留，适合一键执行与回归测试。
质量闸说明：`run` 与 `review-pdf` 支持 `--strict-quality`，用于 Agent 自动化时“有阻断即非零退出”。

双人设交付说明：
1. Agent 路径：优先读 `AGENT_HANDOFF.json`、`ai_summary.json`、`PERSONA_PLAYBOOK.en.md`。
2. 研究生路径：优先读 `STUDENT_BRIEF.*.md`、`PERSONA_PLAYBOOK.*.md`、`student_pack/*`。

### 3.3 固定流水线步骤（17 步）

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
12. ReviewerQuestionSimulator
13. RemediationPlanner
14. RebuttalComposer
15. PaperQAGate
16. ReportBuilder
17. ExporterAndQAGate

---

## 4. 输入契约（运行入口）

输入模型是 `ReviewRunInput`（见 `agent_paper_reviewers/models.py`），核心字段：

```json
{
  "paper": { "format": "pdf|md", "path": "..." },
  "venue": { "name": "...", "year": 2026 },
  "claims": ["..."],
  "constraints": {
    "time_days": 10,
    "gpu_budget_hours": 200,
    "max_new_experiments": 6,
    "cannot_run": []
  },
  "options": {
    "language_mode": "en|en_zh",
    "executor_backend": "codex|agent_api|openai|anthropic|qwen|local_vllm",
    "always_export_pdf": false
  },
  "review_context": {
    "manuscript_stage": "initial_submission|rejected_after_reviews|meta_review_discussion",
    "reviewer_comments": [{ "review_id": "R1", "concern": "..." }],
    "note": ""
  }
}
```

关键行为：

1. 允许 `claims=[]`，系统会先做 claim discovery。
2. `paper.path` 支持相对路径（相对 input json 目录解析）。
3. reviewer comment 里的空 concern 会被过滤。
4. 缺失 `review_id` 自动补成 `R1/R2/...`。

---

## 5. 关键步骤说明（按能力归类）

### 5.1 论文理解层

1. `PaperParser`：解析 PDF/Markdown，输出结构化章节、页面文本、解析质量信号。
2. `ClaimDiscoverer`：从论文文本自动提取候选主张（规则 + LLM），补齐用户未显式输入的 claim。
3. `ClaimNormalizer`：把自然语言 claim 规范成可验证结构（type、success_criteria、weakness_hint）。

### 5.2 证据层

1. `EvidenceIndexer`：分段、抽取 figure/table 相关内容、构建 passage 定位信息（section + passage_id + locator）。
2. `ClaimEvidenceAligner`：做 claim→evidence 匹配，输出强/中/弱证据，并检测潜在负面证据与冲突。

### 5.3 风险层

1. `CitationGraph`：结合 Semantic Scholar + 本地 references，提取引用覆盖、top venue 覆盖、novelty 信号。
2. `GapDetector`：基于 venue required checks 做缺口识别。
3. `RiskRanker`：把 gaps/alignments 映射成可执行风险列表，并输出分数解释。

### 5.4 决策与行动层

1. `ReviewerQuestionSimulator`：生成预测性 reviewer 追问（含角色化 persona）。
2. `RemediationPlanner`：按风险优先级和资源约束生成补实验计划。
3. `RebuttalComposer`：生成 rebuttal 草稿 + rebuttal 计划 + rebuttal 预审。
4. `PaperQAGate`：对 rebuttal 做自审，必要时触发重写。
5. `ReportBuilder`：汇总短版决策报告 + 长版全审稿报告 + 诊断报告。

---

## 6. 当前实现中的关键特性

### 6.1 统计显著性自动检测（新）

`GapDetector` 对 `statistical_significance/significance_reporting` 采用混合策略：

1. 正则检测：`mean±std`、`p-value`、`confidence interval`、seed 信息、检验名。
2. LLM 补充检测：在正则不足时调用 executor 识别统计信号。
3. 结果输出到 `required_check_outcomes[*].statistical_detection`，包含：
- `signals`
- `missing_core_signals`
- `regex` 命中明细
- `llm` 识别明细（若启用）

### 6.2 角色化审稿人模拟（persona）

`ReviewerQuestionSimulator` 当前覆盖至少三类核心 persona：

1. `methodology_reviewer`
2. `empirical_reviewer`
3. `theory_reviewer`

并按稿件阶段动态调整问题风格（初投 vs 讨论期）。

### 6.3 可追踪错误边界

每次运行都会落地：

1. `pipeline_steps.json`：每个 step 的状态与错误。
2. `run_result.json`：整体状态、失败步骤、QA issues、已产出文件。

即便中途失败，也尽量保留已完成产物，状态标记为 `partial_failed`。

---

## 7. 输出契约（人读版）

### 7.1 研究生优先入口（推荐默认阅读顺序）

1. `START_HERE.{md|en.md|zh.md}`：总入口，告诉用户先看哪 3 个文件。
2. `RUN_GUIDE.{md|en.md|zh.md}`：运行状态、阻断项、下一步动作。
3. `STUDENT_BRIEF.{md|en.md|zh.md}`：研究生极简执行摘要（Top 阻断 + 前 24 小时动作）。
4. `CHAT_SUMMARY.{md|en.md|zh.md}`：人话版双语总结（问题->原因->修复->预期收益）。
5. `CHAT_REBUTTAL.{md|en.md|zh.md}`：人话版双语 rebuttal 讲解（解释 R1/R2 是否模拟）。
6. `PERSONA_PLAYBOOK.{md|en.md|zh.md}`：双人设执行手册（Agent 编排流 + 研究生改稿流）。
7. `student_pack/en/001-submission-decision.md`：一页决策（是否投稿 + Top 阻断问题）。
8. `student_pack/en/002-action-items.md`：可执行动作清单（优先级、证据锚点、工作量）。
9. `student_pack/en/003-rebuttal-draft.md`：与风险映射的 rebuttal 草稿。
10. 双语模式下同步生成 `student_pack/zh/*`。

### 7.2 主报告（完整视图）

1. `decision_brief.{en|zh}.{md|json|pdf}`：用于快速决策。
2. `full_review.{en|zh}.{md|json|pdf}`：逐条风险、证据对齐、补救任务。
3. `diagnosis_report.{en|zh}.{md|json|pdf}`：问题-原因-修复-影响的诊断视图。
4. `rebuttal.{en|zh}.{md|json|pdf}`：可修改的 rebuttal 草稿。

### 7.3 关键支撑工件（JSON）

1. `claim_discovery.json`
2. `claim_evidence_matrix.json`
3. `reviewer_questions.json`
4. `remediation_plan.json`
5. `rebuttal_plan.json`
6. `rebuttal_precheck.json`
7. `paper_qa_gate.json`
8. `gaps.json`
9. `risk_ranking.json`
10. `venue_recommendations.json`
11. `venue_profile_used.json`
12. `skill_flow_used.json`
13. `runtime_context.json`
14. `generated_input.json`（`review-pdf` 模式自动生成）
15. `ai_summary.json`（Agent 可读摘要）
16. `AGENT_HANDOFF.json`（Agent 下一轮可直接接手的机器交接包）
17. `feedback_template.json`
18. `pipeline_steps.json`
19. `run_summary.json`
20. `run_result.json`

`ai_summary.json` 关键字段：
- `degraded` / `degraded_reasons`
- `student_pack_ready`
- `recommended_next_action`
- `step_overview`
- `key_files`（相对 `run_dir`）
- `persona_routes`
- `minimal_checks`

机器可校验结构请以 `docs/output-schema.json` 为准。

---

## 8. 降级策略与鲁棒性

### 8.1 执行器降级

1. 外部模型不可用：回退 deterministic executor，流程不中断。
2. `local_vllm` 502 或 API 不可达：记录 QA issue，回退规则逻辑。

### 8.2 规则回退

1. 本地 venue 规则缺失：回退 `_fallback`。
2. 未知会议：使用 executor 自举规则草案。
3. 以上回退都不应阻断主流程。

### 8.3 引用图降级

1. Semantic Scholar 429/403：回退本地 references。
2. 仍输出 citation summary，但附带来源和置信提示。

### 8.4 PDF 导出降级

1. `always_export_pdf=false`：只导出 md/json。
2. 开启 PDF 且引擎缺失：记录 QA issue，不影响 md/json。

---

## 9. 反馈闭环与历史画像

1. 系统会输出 `feedback_template.json` 供用户标注 risk 判断是否正确。
2. 反馈会进入历史画像（按 venue/year/author 维度）用于后续校准。
3. `historical_profile.json` 记录长期弱项分布（如统计显著性长期短板）。

---

## 10. 扩展点（给二次开发）

### 10.1 新增 venue 规则

在 `data/venue_rules/<venue>/<year>.yaml` 添加：

1. `scoring_axes`
2. `weights`
3. `required_checks`
4. `required_check_specs`
5. `rebuttal_policy`

### 10.2 新增检查项

在 `GapDetector` 中新增 check evaluator，并在 `required_check_specs` 配置阈值。

### 10.3 新增审稿 persona

在 `ReviewerQuestionSimulator` 中新增 persona 模板，并加入覆盖约束。

---

## 11. 非目标（明确边界）

1. 不替代真实同行评审。
2. 不保证接受率。
3. 不自动改写论文正文（当前只给建议与草稿，不直接改原稿）。

---

## 12. 文档维护规则

1. 代码改动涉及输入/输出字段时，必须同步更新：
- `docs/output-schema.json`
- `docs/product-spec.md`
2. 若两者冲突，以“当前代码行为 + schema”作为事实标准，并在同次提交修正文档。
