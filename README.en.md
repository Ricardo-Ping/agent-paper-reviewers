# agent-paper-reviewers (English)

`agent-paper-reviewers` is a pre-submission reviewer simulator using **Skill-driven flow + MCP-provided tool capabilities**.

## Documentation
- Output artifact guide: `OUTPUT_FILES.en.md`
- Chinese output artifact guide: `OUTPUT_FILES.zh.md`
- Chinese README: `README.zh.md`
- Skill flow config: `agent-paper-reviewers-skill/flow_config.yaml`

## Architecture
- Skill-driven flow: pipeline order is loaded from `agent-paper-reviewers-skill/flow_config.yaml`.
- MCP capability layer: tool capabilities (for example OpenReview policy resolution) are provided through `agent_paper_reviewers.mcp` providers.
- Executor layer: LLM/agent backends are still plugged via `ExecutorAdapter`.

## What it does
- Parses paper drafts (`pdf` / `md`)
- Aligns claims with evidence
- Detects gaps (baseline / significance / ablation / error analysis)
- Ranks reject risks (`P0/P1/P2`)
- Generates remediation plan and rebuttal draft
- Exports `MD + JSON + PDF`

## Runtime options
- `language_mode`: `en` | `en_zh`
- `executor_backend`: `codex|agent_api|openai|anthropic|qwen|local_vllm`
- `mcp_backend`: `http|disabled`
- `always_export_pdf`: `true|false`

## Output location
Outputs are always written to:

```text
output/<paper_title>/
```

The folder is replaced on each run for the same paper title.

## Encoding and PDF readability
- Markdown and JSON are written in UTF-8.
- PDF is generated via `pandoc` with engine fallback: `xelatex -> lualatex -> tectonic`.
- Chinese PDF rendering quality depends on available CJK fonts (default is `Microsoft YaHei` in this project).

## Quick start
```bash
python -m agent_paper_reviewers.cli doctor
python -m agent_paper_reviewers.cli run --input examples/sample_input.json --output-dir output
```

## Main commands
- `doctor`: dependency check (`pandoc`, LaTeX engines, conda)
- `run`: execute the configured Skill pipeline
- `refresh-venue`: append monthly venue policy refresh reminder

