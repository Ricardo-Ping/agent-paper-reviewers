from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_STEP_ORDER = [
    "Intake",
    "VenueProfileResolver",
    "PaperParser",
    "ClaimDiscoverer",
    "ClaimNormalizer",
    "EvidenceIndexer",
    "ClaimEvidenceAligner",
    "CitationGraph",
    "GapDetector",
    "VenueRecommender",
    "RiskRanker",
    "ReviewerQuestionSimulator",
    "RemediationPlanner",
    "RebuttalComposer",
    "PaperQAGate",
    "ReportBuilder",
    "ExporterAndQAGate",
]


@dataclass
class SkillFlowProfile:
    steps: list[str]
    source: str
    warnings: list[str]
    mcp_capabilities: dict[str, bool]


def load_skill_flow(repo_root: Path) -> SkillFlowProfile:
    path = repo_root / "agent-paper-reviewers-skill" / "flow_config.yaml"
    if not path.exists():
        return SkillFlowProfile(
            steps=list(DEFAULT_STEP_ORDER),
            source="default",
            warnings=["skill_flow_missing_use_default"],
            mcp_capabilities={"openreview_policy_resolver": True},
        )

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return SkillFlowProfile(
            steps=list(DEFAULT_STEP_ORDER),
            source=str(path),
            warnings=[f"skill_flow_parse_error:{exc}"],
            mcp_capabilities={"openreview_policy_resolver": True},
        )

    configured_steps = raw.get("pipeline_steps")
    mcp_capabilities = raw.get("mcp_capabilities") or {"openreview_policy_resolver": True}
    warnings: list[str] = []

    if not isinstance(configured_steps, list) or not all(isinstance(x, str) for x in configured_steps):
        warnings.append("skill_flow_invalid_steps_use_default")
        return SkillFlowProfile(
            steps=list(DEFAULT_STEP_ORDER),
            source=str(path),
            warnings=warnings,
            mcp_capabilities=mcp_capabilities,
        )

    unknown = [step for step in configured_steps if step not in DEFAULT_STEP_ORDER]
    missing = [step for step in DEFAULT_STEP_ORDER if step not in configured_steps]
    duplicated = [step for i, step in enumerate(configured_steps) if step in configured_steps[:i]]

    if unknown:
        warnings.append("skill_flow_unknown_steps:" + ",".join(unknown))
    if missing:
        warnings.append("skill_flow_missing_steps:" + ",".join(missing))
    if duplicated:
        warnings.append("skill_flow_duplicated_steps:" + ",".join(sorted(set(duplicated))))

    if warnings:
        return SkillFlowProfile(
            steps=list(DEFAULT_STEP_ORDER),
            source=str(path),
            warnings=warnings,
            mcp_capabilities=mcp_capabilities,
        )

    return SkillFlowProfile(
        steps=configured_steps,
        source=str(path),
        warnings=[],
        mcp_capabilities=mcp_capabilities,
    )


