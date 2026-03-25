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
- Rank rejection risks (P0/P1/P2).
- Generate prioritized remediation plans.
- Draft rebuttal responses per reviewer concern.
- Export single-language or bilingual artifacts (`en` / `en_zh`).

## Architecture
- Skill-driven flow: ordered by `agent-paper-reviewers-skill/flow_config.yaml`.
- MCP capabilities: concrete tool capabilities are injected via MCP providers.
- Pluggable executors: `codex|agent_api|openai|anthropic|qwen|local_vllm`.

## Requirements
- OS: Windows or Linux (Linux + CUDA 12.1 recommended for GPU inference).
- Python: `3.11.x`.
- Package manager: Conda.
- PDF export: `pandoc` + (`xelatex` or `lualatex` or `tectonic`).
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
```

## Quick start
```bash
python -m agent_paper_reviewers.cli run --input examples/sample_input.json --output-dir output
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
- `claims`: core claims list.
- `constraints`: time/GPU/experiment boundaries.
- `options.language_mode`: `en` or `en_zh`.
- `options.executor_backend`: executor backend.
- `options.mcp_backend`: `http` or `disabled`.
- `options.always_export_pdf`: always export PDF or not.

## Output artifacts
### Core reports
- `decision_brief.en.md/json/pdf`: short decision report.
- `full_review.en.md/json/pdf`: full detailed review report.
- `rebuttal.en.md/json/pdf`: rebuttal draft package.

### Bilingual mirror
- If `language_mode=en_zh`, also generate:
- `decision_brief.zh.md/json/pdf`
- `full_review.zh.md/json/pdf`
- `rebuttal.zh.md/json/pdf`

### Structured/debug outputs
- `claim_evidence_matrix.json`
- `remediation_plan.json`
- `venue_profile_used.json`
- `skill_flow_used.json`
- `mcp_runtime.json`
- `run_summary.json`
- `run_result.json`
- `artifacts/`

## Run status semantics
- `success`: all configured outputs generated.
- `partial_failed`: core outputs generated, optional parts failed.
- `failed`: pipeline failed before core deliverables.

## FAQ
- PDF export fails: run `doctor` and check `pandoc`/LaTeX engines.
- Chinese text looks broken: open files with UTF-8 encoding.
- `policy_needs_manual_check=true`: dynamic policy resolve failed and local fallback was used.

## References
- OpenClaw Lark README: https://github.com/larksuite/openclaw-lark
- FastAPI README: https://github.com/fastapi/fastapi
- LangChain README: https://github.com/langchain-ai/langchain
