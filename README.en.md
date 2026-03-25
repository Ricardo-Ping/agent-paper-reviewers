# reviewers-sim (English)

`reviewers-sim` is a pipeline-based pre-submission reviewer simulator.

## Documentation
- Output artifact guide: `OUTPUT_FILES.en.md`
- Chinese output artifact guide: `OUTPUT_FILES.zh.md`
- Chinese README: `README.zh.md`

## What it does
- Parses paper drafts (`pdf` / `md`)
- Aligns claims with evidence
- Detects gaps (baseline / significance / ablation / error analysis)
- Ranks reject risks (`P0/P1/P2`)
- Generates remediation plan and rebuttal draft
- Exports `MD + JSON + PDF`

## Language mode
- `en`: English outputs only
- `en_zh`: English + Chinese outputs (mirrored artifacts)

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
python -m reviewers_sim.cli doctor
python -m reviewers_sim.cli run --input examples/sample_input.json --output-dir output
```

## Main commands
- `doctor`: dependency check (`pandoc`, LaTeX engines, conda)
- `run`: execute the 12-step pipeline
- `refresh-venue`: append monthly venue policy refresh reminder

## Architecture note: MCP + Skill
Yes, combining MCP + Skill is a good approach here.

Recommended split:
- Skill layer: workflow policy, prompt templates, output contract, runbook.
- MCP/tool layer: concrete capability calls (OpenReview fetch, file parsing, storage, model endpoints).

Use Skill to define "how to run the process", and MCP to provide "what tools can be called".
