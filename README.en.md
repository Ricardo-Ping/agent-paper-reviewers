# agent-paper-reviewers

[中文](README.md) | English

`agent-paper-reviewers` is a pre-submission reviewer simulator for rejection-risk rehearsal, powered by **Skill-driven workflow + MCP-provided tool capabilities**.

## What it does
Given a paper draft (PDF/Markdown), venue+year, core claims, and resource constraints, it outputs:
- scoring across reviewer dimensions
- ranked rejection risks (P0/P1/P2)
- must-do remediation experiments
- rebuttal draft templates

## Architecture
- Skill layer: workflow and policy sequencing from `agent-paper-reviewers-skill/flow_config.yaml`
- MCP layer: concrete tool capabilities (e.g., OpenReview policy resolution)
- Executor layer: model/agent backend execution (`codex/openai/anthropic/qwen/local_vllm`)

## Quick start
```bash
python -m agent_paper_reviewers.cli doctor
python -m agent_paper_reviewers.cli run --input examples/sample_input.json --output-dir output
```

## Output directory
All artifacts are written to:

```text
output/<paper_title>/
```

The same paper title will overwrite the folder on rerun.

## Output artifacts
### Core reports
- `decision_brief.en.md/json/pdf`: short submission decision brief
- `full_review.en.md/json/pdf`: full review report with detailed risks
- `rebuttal.en.md/json/pdf`: rebuttal draft package

With `language_mode=en_zh`, mirrored Chinese files are also produced:
- `decision_brief.zh.md/json/pdf`
- `full_review.zh.md/json/pdf`
- `rebuttal.zh.md/json/pdf`

### Structured and debug files
- `claim_evidence_matrix.json`: claim-evidence anchors
- `remediation_plan.json`: prioritized remediation tasks
- `venue_profile_used.json`: resolved venue/year policy snapshot
- `skill_flow_used.json`: resolved skill workflow used in the run
- `mcp_runtime.json`: MCP backend/provider and capability switches
- `run_summary.json`: run status summary
- `run_result.json`: detailed run status object
- `artifacts/`: intermediate pipeline artifacts

## Status semantics
- `success`: all configured outputs generated
- `partial_failed`: primary outputs generated, optional parts failed
- `failed`: pipeline failed before core deliverables were completed

## References
- Skill doc: `agent-paper-reviewers-skill/SKILL.md`
- Skill flow: `agent-paper-reviewers-skill/flow_config.yaml`
- Product spec: `docs/product-spec.md`
- Input schema: `docs/output-schema.json`
