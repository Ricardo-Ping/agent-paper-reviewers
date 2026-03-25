---
name: agent-paper-reviewers
description: Simulate strict conference reviewers before submission, score risk dimensions, detect missing experiments, and draft rebuttal with executable remediation actions. Use when users provide paper drafts (PDF/Markdown), target venue/year, and claims for reject-risk rehearsal.
---

# agent-paper-reviewers Skill

## When to use
- The user asks for pre-submission rehearsal or reject-risk analysis.
- The user wants claim-evidence alignment, missing experiment detection, and rebuttal drafting.

## Inputs required
1. Paper draft path (`pdf` or `md`).
2. Venue + year.
3. Core claims.
4. Experiment resource constraints.
5. Output mode (`en` or `en_zh`).

## Workflow
1. Validate input schema and load venue profile.
2. Parse manuscript structure.
3. Normalize claims and align evidence.
4. Detect missing checks and rank reject risks (P0/P1/P2).
5. Generate remediation tasks and rebuttal package.
6. Export fixed deliverables (`MD + JSON + PDF`).

The runtime step order is Skill-driven via:
- `flow_config.yaml` (pipeline step sequence)
- `mcp_capabilities` section in the same file (capability intent)

## Commands
- Health check:
```bash
python -m agent_paper_reviewers.cli doctor
```

- Run pipeline:
```bash
python -m agent_paper_reviewers.cli run --input <input.json> --output-dir runs
```

- Refresh venue rule changelog:
```bash
python -m agent_paper_reviewers.cli refresh-venue
```

## Additional references
- Venue policy notes: `references/venue-rubrics.md`
- Output templates: `references/report-contract.md`
- Skill flow config: `flow_config.yaml`

