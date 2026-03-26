from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import ReviewRunInput
from .venue_loader import normalize_venue_slug


def profiles_root(repo_root: Path) -> Path:
    env_path = os.getenv("AGENT_PAPER_REVIEWERS_PROFILE_ROOT", "").strip()
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = repo_root / p
        return p
    return repo_root / "profiles"


def resolve_author_hash(input_data: ReviewRunInput) -> str:
    direct = str(input_data.profile.author_hash or "").strip().lower()
    if direct:
        return "".join(ch for ch in direct if ch.isalnum() or ch in {"-", "_"})

    author_id = str(input_data.profile.author_id or "").strip()
    if author_id:
        return hashlib.sha1(author_id.encode("utf-8")).hexdigest()[:16]
    return ""


def load_historical_profile_prior(repo_root: Path, input_data: ReviewRunInput) -> dict[str, Any]:
    root = profiles_root(repo_root)
    venue_slug = normalize_venue_slug(input_data.venue.name)
    venue_year_path = root / "venue_year" / venue_slug / f"{input_data.venue.year}.json"
    venue_doc = _load_doc(venue_year_path)

    author_hash = resolve_author_hash(input_data)
    author_doc = _load_doc(root / "authors" / f"{author_hash}.json") if author_hash else {}

    return {
        "available": bool(venue_doc or author_doc),
        "author_hash": author_hash or None,
        "author_profile": _profile_view(author_doc) if author_doc else None,
        "venue_year_profile": _profile_view(venue_doc) if venue_doc else None,
    }


def update_historical_profiles(
    repo_root: Path,
    *,
    run_id: str,
    input_data: ReviewRunInput,
    risk_ranking: dict | None,
    gaps: dict | None,
    alignments: dict | None,
) -> dict[str, Any]:
    if not isinstance(risk_ranking, dict):
        return {
            "updated": False,
            "reason": "risk_ranking_missing",
            "author_hash": resolve_author_hash(input_data) or None,
            "author_profile": None,
            "venue_year_profile": None,
            "run_weaknesses": [],
            "alerts": [],
        }

    run_metrics = _extract_run_metrics(risk_ranking, gaps, alignments)
    root = profiles_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)

    venue_slug = normalize_venue_slug(input_data.venue.name)
    venue_doc_path = root / "venue_year" / venue_slug / f"{input_data.venue.year}.json"
    venue_doc = _load_doc(venue_doc_path)
    updated_venue_doc = _merge_profile_doc(
        current=venue_doc,
        run_id=run_id,
        key_type="venue_year",
        key=f"{venue_slug}:{input_data.venue.year}",
        run_metrics=run_metrics,
    )
    _save_doc(venue_doc_path, updated_venue_doc)

    author_hash = resolve_author_hash(input_data)
    updated_author_doc = None
    author_path = None
    if author_hash:
        author_path = root / "authors" / f"{author_hash}.json"
        author_doc = _load_doc(author_path)
        updated_author_doc = _merge_profile_doc(
            current=author_doc,
            run_id=run_id,
            key_type="author",
            key=author_hash,
            run_metrics=run_metrics,
        )
        _save_doc(author_path, updated_author_doc)

    alerts = _build_alerts(
        _profile_view(updated_author_doc) if updated_author_doc else None,
        _profile_view(updated_venue_doc),
    )
    return {
        "updated": True,
        "author_hash": author_hash or None,
        "author_profile": _profile_view(updated_author_doc) if updated_author_doc else None,
        "venue_year_profile": _profile_view(updated_venue_doc),
        "run_weaknesses": run_metrics.get("run_weaknesses", []),
        "alerts": alerts,
        "storage_paths": {
            "venue_year": str(venue_doc_path),
            "author": str(author_path) if author_path else None,
        },
    }


def _extract_run_metrics(risk_ranking: dict, gaps: dict | None, alignments: dict | None) -> dict[str, Any]:
    scores = risk_ranking.get("scores", {}) if isinstance(risk_ranking, dict) else {}
    risks = risk_ranking.get("risks", []) if isinstance(risk_ranking, dict) else []
    gap_rows = gaps.get("gaps", []) if isinstance(gaps, dict) else []
    alignment_rows = alignments.get("alignments", []) if isinstance(alignments, dict) else []

    gap_code_counts: dict[str, int] = {}
    weakness_counts: dict[str, int] = {}
    weakness_weighted_counts: dict[str, float] = {}

    for row in gap_rows:
        if not isinstance(row, dict):
            continue
        code = str(row.get("code", "")).strip().lower()
        if not code:
            continue
        gap_code_counts[code] = int(gap_code_counts.get(code, 0)) + 1
        weakness = _weakness_from_gap_code(code)
        weakness_counts[weakness] = int(weakness_counts.get(weakness, 0)) + 1
        weakness_weighted_counts[weakness] = round(
            float(weakness_weighted_counts.get(weakness, 0.0) or 0.0) + 1.0,
            6,
        )

    for row in risks:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason", "")).lower()
        weakness = _weakness_from_reason(reason)
        if not weakness:
            continue
        weakness_counts[weakness] = int(weakness_counts.get(weakness, 0)) + 1
        weakness_weighted_counts[weakness] = round(
            float(weakness_weighted_counts.get(weakness, 0.0) or 0.0)
            + _feedback_weight_for_risk(row),
            6,
        )

    none_or_weak = sum(
        1
        for row in alignment_rows
        if str(row.get("strength", "")).lower() in {"none", "weak"}
    )
    if none_or_weak > 0:
        weakness_counts["claim_evidence_alignment"] = int(
            weakness_counts.get("claim_evidence_alignment", 0)
        ) + none_or_weak
        weakness_weighted_counts["claim_evidence_alignment"] = round(
            float(weakness_weighted_counts.get("claim_evidence_alignment", 0.0) or 0.0) + float(none_or_weak),
            6,
        )

    p0 = sum(1 for r in risks if str(r.get("severity", "")).upper() == "P0")
    p1 = sum(1 for r in risks if str(r.get("severity", "")).upper() == "P1")
    p2 = sum(1 for r in risks if str(r.get("severity", "")).upper() == "P2")

    axis_scores = {}
    for axis in ("novelty", "soundness", "experiment", "clarity", "overall"):
        try:
            axis_scores[axis] = round(float(scores.get(axis, 0.0) or 0.0), 3)
        except (TypeError, ValueError):
            axis_scores[axis] = 0.0

    run_weaknesses = [
        {
            "name": name,
            "count": int(weakness_counts.get(name, 0) or 0),
            "weighted_count": round(float(weakness_weighted_counts.get(name, 0.0) or 0.0), 3),
        }
        for name in sorted(
            set(weakness_counts.keys()) | set(weakness_weighted_counts.keys()),
            key=lambda key: (
                -float(weakness_weighted_counts.get(key, 0.0) or 0.0),
                -int(weakness_counts.get(key, 0) or 0),
                key,
            ),
        )
    ]

    return {
        "scores": axis_scores,
        "p0_count": p0,
        "p1_count": p1,
        "p2_count": p2,
        "gap_code_counts": gap_code_counts,
        "weakness_counts": weakness_counts,
        "weakness_weighted_counts": {k: round(float(v or 0.0), 6) for k, v in weakness_weighted_counts.items()},
        "run_weaknesses": run_weaknesses,
    }


def _merge_profile_doc(
    *,
    current: dict[str, Any],
    run_id: str,
    key_type: str,
    key: str,
    run_metrics: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    doc = dict(current) if isinstance(current, dict) else {}
    runs = int(doc.get("runs", 0) or 0)
    new_runs = runs + 1

    avg_scores = dict(doc.get("average_scores", {})) if isinstance(doc.get("average_scores"), dict) else {}
    for axis, value in run_metrics.get("scores", {}).items():
        old = float(avg_scores.get(axis, 0.0) or 0.0)
        avg_scores[axis] = round((old * runs + float(value)) / new_runs, 4)

    gap_code_counts = _merge_counter(doc.get("gap_code_counts", {}), run_metrics.get("gap_code_counts", {}))
    weakness_counts = _merge_counter(doc.get("weakness_counts", {}), run_metrics.get("weakness_counts", {}))
    weakness_weighted_counts = _merge_counter_float(
        doc.get("weakness_weighted_counts", {}),
        run_metrics.get("weakness_weighted_counts", {}),
    )

    recent_runs = list(doc.get("recent_run_ids", [])) if isinstance(doc.get("recent_run_ids"), list) else []
    recent_runs.append(run_id)
    recent_runs = recent_runs[-20:]

    return {
        "schema_version": 1,
        "key_type": key_type,
        "key": key,
        "runs": new_runs,
        "created_at": str(doc.get("created_at") or now),
        "updated_at": now,
        "average_scores": avg_scores,
        "p0_total": int(doc.get("p0_total", 0) or 0) + int(run_metrics.get("p0_count", 0) or 0),
        "p1_total": int(doc.get("p1_total", 0) or 0) + int(run_metrics.get("p1_count", 0) or 0),
        "p2_total": int(doc.get("p2_total", 0) or 0) + int(run_metrics.get("p2_count", 0) or 0),
        "gap_code_counts": gap_code_counts,
        "weakness_counts": weakness_counts,
        "weakness_weighted_counts": weakness_weighted_counts,
        "recent_run_ids": recent_runs,
    }


def _merge_counter(base: object, delta: object) -> dict[str, int]:
    merged: dict[str, int] = {}
    if isinstance(base, dict):
        for k, v in base.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                merged[key] = int(v or 0)
            except (TypeError, ValueError):
                continue
    if isinstance(delta, dict):
        for k, v in delta.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                merged[key] = int(merged.get(key, 0)) + int(v or 0)
            except (TypeError, ValueError):
                continue
    return merged


def _merge_counter_float(base: object, delta: object) -> dict[str, float]:
    merged: dict[str, float] = {}
    if isinstance(base, dict):
        for k, v in base.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                merged[key] = round(float(v or 0.0), 6)
            except (TypeError, ValueError):
                continue
    if isinstance(delta, dict):
        for k, v in delta.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                merged[key] = round(float(merged.get(key, 0.0)) + float(v or 0.0), 6)
            except (TypeError, ValueError):
                continue
    return merged


def _weakness_from_gap_code(code: str) -> str:
    mapping = {
        "missing_significance": "statistical_significance",
        "missing_baseline": "baseline_coverage",
        "missing_ablation": "ablation_completeness",
        "missing_reproducibility": "reproducibility_details",
        "missing_reference_coverage": "related_work_coverage",
        "missing_top_venue_related_work_coverage": "related_work_coverage",
        "missing_top_venue_recent_coverage": "related_work_coverage",
        "weak_claim_alignment": "claim_evidence_alignment",
    }
    return mapping.get(code, code or "unknown")


def _weakness_from_reason(reason: str) -> str:
    text = reason.lower()
    if "significance" in text or "statistical" in text:
        return "statistical_significance"
    if "baseline" in text:
        return "baseline_coverage"
    if "ablation" in text:
        return "ablation_completeness"
    if "reproduc" in text:
        return "reproducibility_details"
    if "related work" in text or "citation" in text:
        return "related_work_coverage"
    if "claim" in text and "evidence" in text:
        return "claim_evidence_alignment"
    return ""


def _profile_view(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(doc, dict) or not doc:
        return None
    weakness_counts = doc.get("weakness_counts", {})
    weakness_weighted_counts = doc.get("weakness_weighted_counts", {})
    top_weaknesses: list[dict[str, Any]] = []
    if isinstance(weakness_counts, dict) or isinstance(weakness_weighted_counts, dict):
        names = set()
        if isinstance(weakness_counts, dict):
            names.update(str(k) for k in weakness_counts.keys())
        if isinstance(weakness_weighted_counts, dict):
            names.update(str(k) for k in weakness_weighted_counts.keys())
        top_weaknesses = [
            {
                "name": name,
                "count": int(weakness_counts.get(name, 0) or 0) if isinstance(weakness_counts, dict) else 0,
                "weighted_count": round(
                    float(weakness_weighted_counts.get(name, 0.0) or 0.0)
                    if isinstance(weakness_weighted_counts, dict)
                    else 0.0,
                    3,
                ),
            }
            for name in sorted(
                (n for n in names if n),
                key=lambda n: (
                    -(
                        float(weakness_weighted_counts.get(n, 0.0) or 0.0)
                        if isinstance(weakness_weighted_counts, dict)
                        else 0.0
                    ),
                    -(
                        int(weakness_counts.get(n, 0) or 0)
                        if isinstance(weakness_counts, dict)
                        else 0
                    ),
                    n,
                ),
            )[:5]
        ]
    return {
        "runs": int(doc.get("runs", 0) or 0),
        "average_scores": doc.get("average_scores", {}),
        "top_weaknesses": top_weaknesses,
        "p0_total": int(doc.get("p0_total", 0) or 0),
        "p1_total": int(doc.get("p1_total", 0) or 0),
        "updated_at": doc.get("updated_at"),
    }


def _feedback_weight_for_risk(risk: dict[str, Any]) -> float:
    adjustment = risk.get("feedback_adjustment", {})
    if not isinstance(adjustment, dict):
        return 1.0
    action = str(adjustment.get("action", "")).strip().lower()
    try:
        conf = float(adjustment.get("calibration_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    if action == "down":
        # Historical feedback suggests this risk is often over-reported.
        return round(max(0.45, 1.0 - 0.55 * conf), 6)
    if action == "up":
        # Historical feedback suggests this risk is often valid and under-weighted.
        return round(min(1.45, 1.0 + 0.40 * conf), 6)
    return 1.0


def _build_alerts(author_profile: dict[str, Any] | None, venue_profile: dict[str, Any] | None) -> list[str]:
    alerts: list[str] = []

    if author_profile:
        for weakness in author_profile.get("top_weaknesses", [])[:2]:
            if int(weakness.get("count", 0) or 0) >= 2:
                alerts.append(
                    f"author_repeated_weakness:{weakness.get('name')}:{weakness.get('count')}"
                )
    if venue_profile:
        for weakness in venue_profile.get("top_weaknesses", [])[:1]:
            if int(weakness.get("count", 0) or 0) >= 5:
                alerts.append(
                    f"venue_year_common_weakness:{weakness.get('name')}:{weakness.get('count')}"
                )
    return alerts


def _load_doc(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_doc(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
