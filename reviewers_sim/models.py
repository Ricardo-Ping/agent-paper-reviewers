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
    always_export_pdf: bool = True


class ReviewRunInput(BaseModel):
    paper: PaperInput
    venue: VenueInput
    claims: list[str] = Field(min_length=1)
    constraints: Constraints = Field(default_factory=Constraints)
    options: RunOptions = Field(default_factory=RunOptions)

    @model_validator(mode="after")
    def normalize(self) -> "ReviewRunInput":
        self.venue.name = self.venue.name.strip()
        self.claims = [c.strip() for c in self.claims if c.strip()]
        return self


class RebuttalPolicy(BaseModel):
    mode: str = "per_review_only"
    per_review_char_limit: int = 2500
    global_char_limit: int = 0
    allow_attachment_pdf: bool = False
    attachment_page_limit: int = 0
    allow_links: bool = False
    dynamic_from_openreview: bool = False


class VenueYearProfile(BaseModel):
    scoring_axes: list[str]
    weights: dict[str, float]
    common_reject_reasons: list[str]
    required_checks: list[str]
    rebuttal_policy: RebuttalPolicy
    openreview_group_id: str = ""
    version_date: str


class VenueProfile(BaseModel):
    name: str
    default_year: int
    years: dict[str, VenueYearProfile]


class EvidenceRef(BaseModel):
    section: str
    passage_id: str
    excerpt: str


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
    char_count: int
    char_limit: int


class RebuttalBundle(BaseModel):
    venue: str
    year: int
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


def ensure_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)
