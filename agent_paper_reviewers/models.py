from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class LanguageMode(str, Enum):
    EN = "en"
    EN_ZH = "en_zh"


class ExecutorBackend(str, Enum):
    CODEX = "codex"
    AGENT_API = "agent_api"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    QWEN = "qwen"
    LOCAL_VLLM = "local_vllm"


class MCPBackend(str, Enum):
    HTTP = "http"
    DISABLED = "disabled"


class ManuscriptStage(str, Enum):
    INITIAL_SUBMISSION = "initial_submission"
    REJECTED_AFTER_REVIEWS = "rejected_after_reviews"
    META_REVIEW_DISCUSSION = "meta_review_discussion"


class PaperInput(BaseModel):
    format: str = Field(pattern=r"^(pdf|md)$")
    path: str


class VenueInput(BaseModel):
    name: str
    year: int


class Constraints(BaseModel):
    time_days: int = 10
    gpu_budget_hours: int = 200
    max_new_experiments: int = 6
    cannot_run: list[str] = Field(default_factory=list)


class RunOptions(BaseModel):
    language_mode: LanguageMode = LanguageMode.EN
    executor_backend: ExecutorBackend = ExecutorBackend.CODEX
    mcp_backend: MCPBackend = MCPBackend.HTTP
    always_export_pdf: bool = False


class ProfileInput(BaseModel):
    author_hash: str = ""
    author_id: str = ""


class ReviewerComment(BaseModel):
    review_id: str = ""
    concern: str


class ReviewContextInput(BaseModel):
    manuscript_stage: ManuscriptStage = ManuscriptStage.INITIAL_SUBMISSION
    reviewer_comments: list[ReviewerComment] = Field(default_factory=list)
    note: str = ""


class ReviewRunInput(BaseModel):
    paper: PaperInput
    venue: VenueInput
    claims: list[str] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    options: RunOptions = Field(default_factory=RunOptions)
    profile: ProfileInput = Field(default_factory=ProfileInput)
    review_context: ReviewContextInput = Field(default_factory=ReviewContextInput)

    @model_validator(mode="after")
    def normalize(self) -> "ReviewRunInput":
        self.venue.name = self.venue.name.strip()
        self.claims = [c.strip() for c in self.claims if c.strip()]
        self.profile.author_hash = self.profile.author_hash.strip()
        self.profile.author_id = self.profile.author_id.strip()
        self.review_context.note = self.review_context.note.strip()

        cleaned_comments: list[ReviewerComment] = []
        for idx, item in enumerate(self.review_context.reviewer_comments, start=1):
            concern = str(item.concern or "").strip()
            if not concern:
                continue
            review_id = str(item.review_id or "").strip() or f"R{idx}"
            cleaned_comments.append(
                ReviewerComment(
                    review_id=review_id,
                    concern=concern,
                )
            )
        self.review_context.reviewer_comments = cleaned_comments
        return self


class RebuttalPolicy(BaseModel):
    mode: str = "per_review_only"
    per_review_char_limit: int = 2500
    global_char_limit: int = 0
    allow_attachment_pdf: bool = False
    attachment_page_limit: int = 0
    allow_links: bool = False
    dynamic_from_openreview: bool = False


class DecisionPolicy(BaseModel):
    strictness_tier: str = "default"
    p0_not_ready: bool = True
    p1_not_ready_threshold: int = 99
    p1_borderline_threshold: int = 3
    min_overall_ready: float = 6.0
    min_overall_borderline: float = 5.2
    notes: str = ""


class RequiredCheckSpec(BaseModel):
    check_name: str = ""
    gap_code: str = ""
    description: str = ""
    severity_hint: str = "P2"
    keywords: list[str] = Field(default_factory=list)
    min_hits: int = 1
    min_distinct_sections: int = 0
    min_citation_outgoing: int = 0
    min_citation_baseline_like: int = 0
    min_citation_top_venue: int = 0
    min_citation_top_venue_recent: int = 0
    section_ratio_targets: dict[str, float] = Field(default_factory=dict)
    section_ratio_tolerance: float = 0.0
    section_ratio_min_total_words: int = 0
    section_ratio_min_bucket_words: int = 0
    section_aliases: dict[str, list[str]] = Field(default_factory=dict)
    terminology_min_mentions: int = 0
    terminology_min_variant_hits: int = 0
    terminology_exempt_terms: list[str] = Field(default_factory=list)
    notes: str = ""


class VenueYearProfile(BaseModel):
    scoring_axes: list[str]
    weights: dict[str, float]
    common_reject_reasons: list[str]
    required_checks: list[str]
    required_check_specs: dict[str, RequiredCheckSpec] = Field(default_factory=dict)
    rebuttal_policy: RebuttalPolicy
    decision_policy: DecisionPolicy = Field(default_factory=DecisionPolicy)
    openreview_group_id: str = ""
    version_date: str


class VenueRuleSnapshot(BaseModel):
    schema_version: int = 1
    venue: str
    year: int
    display_name: str
    profile: VenueYearProfile


class VenueProfile(BaseModel):
    name: str
    default_year: int
    years: dict[str, VenueYearProfile]


class EvidenceRef(BaseModel):
    section: str
    passage_id: str
    excerpt: str
    section_id: str = ""
    section_index: int = 0
    page: int = 0
    kind: str = ""
    anchor_label: str = ""
    anchor_type: str = ""
    locator: dict[str, Any] = Field(default_factory=dict)
    confidence_level: str = ""
    confidence_score: float = 0.0
    conflict_alert: bool = False
    conflict_reason: str = ""
    relation: str = "support"


class ClaimAlignment(BaseModel):
    claim_id: str
    claim_text: str
    strength: str
    score: float
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


class GapItem(BaseModel):
    code: str
    severity_hint: str
    description: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


class RiskItem(BaseModel):
    id: str
    severity: str
    score: float
    reason: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    likely_reject_phrase: str
    fix_hint: str


class ExperimentTask(BaseModel):
    id: str
    risk_id: str
    title: str
    priority: str
    effort: str
    est_time_days: float
    est_gpu_hours: int
    expected_gain: str
    protocol: list[str]


class RebuttalItem(BaseModel):
    review_id: str
    concern: str
    response: str
    new_evidence: list[str]
    paper_change: str
    evidence_anchor_refs: list[EvidenceRef] = Field(default_factory=list)
    evidence_anchor_hint: str = ""
    char_count: int
    char_limit: int


class RebuttalBundle(BaseModel):
    venue: str
    year: int
    manuscript_stage: str = ManuscriptStage.INITIAL_SUBMISSION.value
    mode: str
    items: list[RebuttalItem]
    global_response: str | None
    attachment_pdf: str | None


class ScoreBundle(BaseModel):
    novelty: float
    soundness: float
    experiment: float
    clarity: float
    overall: float


class TaskSpec(BaseModel):
    task_type: str
    prompt: str
    context: dict[str, Any]
    output_schema: dict[str, Any]
    model_profile: str


class TaskResult(BaseModel):
    ok: bool
    output: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RunStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"


class RunSummary(BaseModel):
    run_id: str
    status: RunStatus
    output_dir: str
    qa_issues: list[str] = Field(default_factory=list)
    step_statuses: list[dict[str, Any]] = Field(default_factory=list)
    produced_artifacts: list[str] = Field(default_factory=list)
    historical_profile: dict[str, Any] = Field(default_factory=dict)


def ensure_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)
