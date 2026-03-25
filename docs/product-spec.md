# Product Spec - Reviewers Sim v3

## Goal
Simulate strict peer review before submission and output:
- dimension scores
- ranked rejection risks
- prioritized remediation experiments
- rebuttal drafts

## Pipeline
The runtime pipeline is Skill-driven from `reviewers-sim-skill/flow_config.yaml`.

1. Intake
2. VenueProfileResolver
3. PaperParser
4. ClaimNormalizer
5. EvidenceIndexer
6. ClaimEvidenceAligner
7. GapDetector
8. RiskRanker
9. RemediationPlanner
10. RebuttalComposer
11. ReportBuilder
12. ExporterAndQAGate

## Output Contract
Always emit MD + JSON + PDF (PDF may fail and mark partial_failed if LaTeX engine unavailable).

Language modes:
- en
- en_zh (produce mirrored Chinese outputs)

## Runtime
- `python -m reviewers_sim.cli doctor`
- `python -m reviewers_sim.cli run --input <json> --output-dir runs`

## Architecture
- Skill layer: controls workflow order and capability intent.
- MCP layer: provides concrete tool capabilities (e.g., OpenReview policy resolver).
- Executor layer: provides model/agent task execution.
