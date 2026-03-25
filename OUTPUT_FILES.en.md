# Output Files (English)

All run artifacts are saved under:

```text
output/<paper_title>/
```

## Core outputs
- `decision_brief.en.md/json/pdf`: short submission decision report.
- `full_review.en.md/json/pdf`: full risk and evidence report.
- `rebuttal.en.md/json/pdf`: rebuttal draft package.

If `language_mode=en_zh`, mirrored Chinese files are also generated:
- `decision_brief.zh.md/json/pdf`
- `full_review.zh.md/json/pdf`
- `rebuttal.zh.md/json/pdf`

## Skill + MCP runtime outputs
- `skill_flow_used.json`: resolved Skill workflow order and source config.
- `mcp_runtime.json`: active MCP backend/provider and exposed capabilities.

## Supporting outputs
- `claim_evidence_matrix.json`: claim-to-evidence mapping.
- `remediation_plan.json`: prioritized experiment actions.
- `venue_profile_used.json`: resolved venue/year policy used in this run.
- `run_summary.json`: run status summary.
- `run_result.json`: detailed run status object.
- `artifacts/`: intermediate pipeline artifacts for debugging.

## Status semantics
- `success`: all configured outputs generated successfully.
- `partial_failed`: main outputs generated, but some optional parts failed (commonly PDF export when LaTeX engine is missing).
- `failed`: pipeline failed before core output completion.
