# agent-paper-reviewers

[中文](README.md) | English

I built this project for one practical reason: before submission, I want to run a strict rejection rehearsal instead of waiting passively for reviewer feedback.

`agent-paper-reviewers` simulates a strict reviewer perspective and produces actionable fixes before you submit.

## Why this project
- Convert “unknown rejection risk” into a visible, ranked risk list.
- Move from vague advice to executable remediation tasks.
- Keep the workflow reusable and backend-agnostic.

## What it can do
- Parse paper drafts (`pdf` / `md`) into structured sections.
- Run claim-evidence alignment checks.
- Detect common missing checks: baseline, significance, ablation, error analysis, reproducibility.
- Add venue-aware writing-structure checks for section-length balance (`introduction/method/experiments/discussion`).
- Add terminology consistency checks (acronym expansion drift, mixed term variants, unstable naming).
- Rank rejection risks (P0/P1/P2).
- Generate prioritized remediation plans.
- Draft rebuttal responses per reviewer concern.
- Recommend Top venues when you are not sure where to submit (with match score, reasons, and key gaps).
- Add score-leverage analysis (which axis drags overall score most and what to improve first).
- Export single-language or bilingual artifacts (`en` / `en_zh`).

## Positioning vs Common Tools
Many tools are useful, but they solve different problems.  
`agent-paper-reviewers` is not trying to replace everything. It fills the missing “pre-submission rejection rehearsal” layer.

| Category | Typical Tools | What they do well | Common blind spot | How `agent-paper-reviewers` complements |
|---|---|---|---|---|
| Language polishing | Paperpal / Writefull / Trinka | Grammar, wording, fluency | Language quality only; not rejection-risk reasoning | Upgrades from “well written” to “claim is defensible” |
| Paper discovery / Q&A | Semantic Scholar / Elicit / Consensus | Search and evidence lookup | No submission-risk grading | Produces actionable rejection risk levels (P0/P1/P2) |
| Submission technical checks | Paperpal Preflight / Typeset | Format, length, figure/table compliance | Format-heavy, weak argument validation | Adds claim-evidence alignment and experiment-gap checks |
| Professional editing services | Enago / AJE | Human expert editing | Expensive, slower cycle, less structured outputs | Automated, repeatable, structured JSON/MD/PDF artifacts |
| Academic RAG | PaperQA / Keep | Cross-paper QA and citation support | QA-centric, not submission-diagnosis-centric | Generates full issue-cause-fix-impact diagnosis reports |
| Research agent assistants | Coauthor / AutoRF | Research workflow assistance | Often generation-heavy, weak rejection-rehearsal loop | End-to-end loop: risk -> remediation -> rebuttal |
| OpenReview itself | OpenReview | Real review and discussion platform | Real review happens after submission | Simulated strict review before submission |

One-line summary:  
The unique value here is an executable loop: risk detection -> evidence alignment -> remediation experiments -> rebuttal drafting -> self-check.

## Architecture
- Skill-driven flow: ordered by `agent-paper-reviewers-skill/flow_config.yaml`.
- MCP capabilities: concrete tool capabilities are injected via MCP providers.
- Pluggable executors: `codex|agent_api|openai|anthropic|qwen|local_vllm`.
- Unknown-venue fallback chain: local `_fallback` rule -> OpenReview dynamic discovery -> executor bootstrap draft.

## Recommended Mode (Agent Analysis + Skill Toolkit)
Use this project as a toolkit layer, and let your preferred agent handle semantic reasoning.

1. Use toolkit commands to fetch structured context (`tool-parse-paper`, `tool-venue-profile`).
2. Let the agent read/analyze the paper (claims, gaps, risk ranking, rebuttal drafting).
3. Use toolkit formatter commands to output standardized student-facing files.

### Tool-only CLI commands
```bash
# 1) Resolve venue policy profile
python -m agent_paper_reviewers.cli tool-venue-profile --venue ICLR --year 2026 --json

# 2) Parse paper into structured JSON
python -m agent_paper_reviewers.cli tool-parse-paper --paper-path /abs/path/paper.pdf --output parsed_paper.json

# 3) Get the agent-analysis template contract
python -m agent_paper_reviewers.cli tool-format-template --template student_pack_analysis --output analysis_template.json

# 4) Format agent analysis to 3 student-facing markdown files
python -m agent_paper_reviewers.cli tool-format-student-pack --analysis-json agent_analysis.json --output-dir output/student_pack/en --language en
```

Compatibility: the full `run --input ...` pipeline is still available for one-shot runs.

## One-line command for Agent
You can tell your Agent directly:

```text
Please set up and initialize agent-paper-reviewers in this repo: create and activate conda env agent-paper-reviewers-gpu, run pip install -e ., run python -m agent_paper_reviewers.cli doctor, and execute one validation run with examples/sample_input.json.
```

If the PDF is only uploaded in chat and not saved yet:

```text
Please save the PDF I uploaded to input_files/paper.pdf in this repo first, then create input.json using that path and run the full review flow.
```

## Requirements
- OS: Windows or Linux (Linux + CUDA 12.1 recommended for GPU inference).
- Python: `3.11.x`.
- Package manager: Conda.
- PDF export (optional): `pandoc` + (`xelatex` or `lualatex` or `tectonic`).
- Optional network access: OpenReview dynamic policy resolve and online translation fallback.

## Installation
### 1. Clone
```bash
git clone https://github.com/Ricardo-Ping/agent-paper-reviewers.git
cd agent-paper-reviewers
```

### 2. Create conda env
CPU:
```bash
conda env create -f envs/environment.cpu.yml
conda activate agent-paper-reviewers-cpu
```

GPU:
```bash
conda env create -f envs/environment.gpu.yml
conda activate agent-paper-reviewers-gpu
```

### 3. Install in editable mode
```bash
pip install -e .
```

### 4. Verify runtime
```bash
python -m agent_paper_reviewers.cli doctor
# machine-readable output (for scripts/CI)
python -m agent_paper_reviewers.cli doctor --json
```

### 5. (Optional) Enable PDF export toolchain
Base environments do not install PDF toolchain by default. Install only if you need `*.pdf` outputs:
```bash
conda install -n agent-paper-reviewers-cpu -c conda-forge pandoc tectonic
# or install into the GPU env
conda install -n agent-paper-reviewers-gpu -c conda-forge pandoc tectonic
```

## Quick start
```bash
python -m agent_paper_reviewers.cli run --input examples/sample_input.json --output-dir output
```

Note:
- This command reads `examples/sample_input.json`.
- That sample input currently points to `examples/sample_paper.md` (a Markdown sample), not your custom PDF.
- `paper.path` supports both absolute and relative paths; relative paths are resolved against the input.json directory.

To run your own PDF directly, use either:

Option 1 (recommended): use the provided PDF sample input
```bash
python -m agent_paper_reviewers.cli run --input examples/sql_translation_gpu_input.json --output-dir output
```

Option 2: create your own `input.json` and make sure:
```json
"paper": {
  "format": "pdf",
  "path": "absolute path to your PDF"
}
```

You can also start from:
- `examples/pdf_input.template.json`

If you are unsure which venue to target, run the venue recommendation sample:
```bash
python -m agent_paper_reviewers.cli run --input examples/venue_recommend_input.json --output-dir output
```
This produces `venue_recommendations.json` and appends recommendation sections in `decision_brief/full_review`.

If you are already in reviewer discussion (R1/R2/R3 available), run discussion-stage mode:
```bash
python -m agent_paper_reviewers.cli run --input examples/meta_review_input.template.json --output-dir output
```

Outputs are written to:
```text
output/<paper_title>/
```

Rerun on the same paper title overwrites that folder.

## Input overview
`examples/sample_input.json` includes:
- `paper`: input format/path.
- `venue`: venue + year.
- `claims`: core claims list (optional; if empty, the pipeline auto-discovers candidate claims from abstract/contribution/conclusion sections).
  - If you are unsure about the target venue, you can use a placeholder name (for example `UnknownConf`) and rely on `venue_recommendations`.
- `review_context.manuscript_stage`: manuscript stage (strategy switch)
  - `initial_submission`: decide whether to submit now
  - `rejected_after_reviews`: focus on salvageability for resubmission
  - `meta_review_discussion`: focus on point-by-point reviewer responses
- `review_context.reviewer_comments`: optional reviewer concerns (`review_id + concern`) used to reprioritize risks and rebuttal targets in post-reject/discussion mode.
- `constraints`: time/GPU/experiment boundaries.
- `options.language_mode`: `en` or `en_zh`.
- `options.executor_backend`: executor backend.
- `options.mcp_backend`: `http` or `disabled`.
- `options.always_export_pdf`: whether to export PDFs (default `false`, only `md+json` by default).
- `profile.author_hash` / `profile.author_id`: optional author identity for historical weakness profiling (`author_id` is hashed locally).
  - If no author identity is provided, the system still accumulates profile statistics at `venue+year` scope.

## Executor backends (real calls)
- `agent_api`: `OpenClawNodeExecutor` via `OpenClaw /api/sessions/spawn`.
- `openai`: OpenAI-compatible chat completions.
- `anthropic`: Anthropic Messages API.
- `qwen`: Qwen OpenAI-compatible endpoint.
- `local_vllm`: local vLLM OpenAI-compatible endpoint.
- `codex`: OpenAI-compatible endpoint with configurable model.

Common environment variables:
- `AGENT_PAPER_REVIEWERS_OPENCLAW_URL` (default `http://localhost:18789`)
- `OPENAI_API_KEY` / `OPENAI_BASE_URL`
- `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`
- `QWEN_API_KEY` / `QWEN_BASE_URL`
- `LOCAL_VLLM_BASE_URL` / `LOCAL_VLLM_API_KEY`
- `SEMANTIC_SCHOLAR_API_KEY` (for Citation Graph, helps avoid anonymous rate limits)
- `AGENT_PAPER_REVIEWERS_PROFILE_ROOT` (optional override for historical profile storage; default `./profiles`)

## Output artifacts
### Student-first entry (recommended)
- `START_HERE.md`: quick entry that tells you which 3 files to read first.
- `student_pack/en/001-submission-decision.md`: one-page submit/hold decision with top blockers.
- `student_pack/en/002-action-items.md`: executable action list with anchors and effort hints.
- `student_pack/en/003-rebuttal-draft.md`: risk-mapped rebuttal draft by reviewer concern.
- In bilingual mode, matching `student_pack/zh/*` and `START_HERE.zh.md` are also generated.

This `student_pack` is the human-readable default workflow. The other JSON/intermediate files are mainly for debugging, traceability, and feedback loops.

### Core reports
- `decision_brief.en.md/json`: short decision report (with per-axis score rationale).
- `full_review.en.md/json`: full detailed review report (with per-axis score rationale).
  - Includes `score_leverage_analysis` (weights, current weighted contribution, target gap, and fastest-improvement axis).
- `diagnosis_report.en.md/json`: student-friendly diagnosis report (issue -> cause -> fix -> why it matters).
- `rebuttal.en.md/json`: rebuttal draft package.
  - Includes `manuscript_stage`; in discussion mode it prioritizes `review_context.reviewer_comments`.
- If `options.always_export_pdf=true`, matching `*.pdf` files are also generated.

### Bilingual mirror
- If `language_mode=en_zh`, also generate:
- `decision_brief.zh.md/json`
- `full_review.zh.md/json`
- `diagnosis_report.zh.md/json`
- `rebuttal.zh.md/json`
- If `options.always_export_pdf=true`, matching `*.pdf` files are also generated.

### Bilingual translation quality note
- Bilingual outputs are for readability and internal collaboration, not publication-grade final translation.
- Translation uses a fallback chain: glossary replacement -> GoogleTranslator -> MarianMT -> executor `translate_zh` -> stable pseudo-translate fallback.
- Under deterministic executor, offline mode, or unavailable translation backends, the pipeline may fall back to pseudo-translate; Chinese text can be mixed or unnatural.
- For submission-facing text, keep English as the source of truth and manually proofread Chinese mirrors.
- To improve Chinese quality, use an executor backend with real translation capability and ensure network/API availability.

### Structured/debug outputs
- `claim_discovery.json`
- `claim_evidence_matrix.json`
- `remediation_plan.json`
- `venue_recommendations.json` (recommended venues with match score, reasons, and pass/fail checks)
- `feedback_template.json` (mark each risk as `correct|incorrect|pending`)
- `feedback_README.en.md` / `feedback_README.zh.md` (how to submit feedback)
- `venue_profile_used.json`
  - Includes `required_check_specs` (executable thresholds) and `source/source_notes` (rule provenance chain).
- `skill_flow_used.json`
- `mcp_runtime.json`
- `pipeline_steps.json` (step-by-step execution trace with success/failed/skipped)
- `run_summary.json`
- `run_result.json` (full run object with per-step `success/failed/skipped`, `failed_step`, and produced artifact list)
- `historical_profile.json` (updated historical profile snapshot: author-level + venue/year-level weaknesses)
- `artifacts/`
  - `artifacts/risk_ranking.json` now includes `stage_strategy`, `focus_risks`, and `reviewer_comment_alignment` to explain stage-dependent prioritization.
  - `artifacts/citation_graph.json` now includes `content_novelty_score` and `content_novelty_components` to estimate novelty from paper content when citation data is sparse.

## Run status semantics
- `success`: all configured outputs generated.
- `partial_failed`: core outputs generated, optional parts failed (typically when PDF export is enabled but toolchain is missing).
- `failed`: pipeline failed before core deliverables.

## Venue rule refresh
```bash
python -m agent_paper_reviewers.cli refresh-venue --venue all --year 2026
```

Common options:
- `--venue all`: refresh all venues.
- `--venue iclr,icml`: refresh selected venues only.
- `--openreview-group <group_id>`: override OpenReview group for a single venue refresh.
- `--dry-run`: preview without writing files.

Venue rules are now directory-based under `data/venue_rules/<venue>/<year>.yaml`, plus global fallback `_fallback.yaml`.

## FAQ
- Why no PDFs by default: PDF export is disabled by default; the default workflow outputs `md+json`, which is usually enough for Overleaf copy/paste and editing.
- How to enable PDFs: set `"options": {"always_export_pdf": true}` in your input JSON.
- PDF export fails: run `doctor` and check the `pdf export capability (optional)` row, or run `doctor --json` and verify `pdf_export.ready`.
- `pdf_parse_quality_*` warnings: parser quality is likely low; verify the PDF has a text layer, or run OCR / convert to clean Markdown before review.
- Chinese text looks broken: open files with UTF-8 encoding.
- `pseudo-translate` appears in diagnosis output: translation fallback was used; Chinese output is draft-quality only. Enable a real translation backend and rerun.
- `policy_needs_manual_check=true`: dynamic policy resolve failed and local fallback was used.
- `citation_graph_warning:semantic_scholar_status_429`: Semantic Scholar request was rate-limited; configure `SEMANTIC_SCHOLAR_API_KEY`.

## Feedback loop
Each run exports `feedback_template.json`. You can mark whether each risk judgment was correct and feed it back into future runs.

1. Open `feedback_template.json`.
2. Set each `items[*].verdict` to `correct` or `incorrect` (`pending` to skip).
3. Optionally add `comment` for incorrect items.
4. Submit:
```bash
python -m agent_paper_reviewers.cli submit-feedback --input output/<paper_title>/feedback_template.json
```
5. Feedback records are stored under:
```text
feedback/<venue>/<year>/
```
Future runs for the same venue/year automatically load these signals to calibrate risk scoring.


