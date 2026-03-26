from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .venue_loader import normalize_venue_slug


def feedback_root(repo_root: Path) -> Path:
    return repo_root / "feedback"


def make_risk_fingerprint(reason: str, likely_reject_phrase: str = "") -> str:
    def normalize(text: str) -> str:
        text = text or ""
        text = re.sub(r"\bRISK-\d+\b", "RISK", text, flags=re.IGNORECASE)
        text = re.sub(r"\bC\d+\b", "CLAIM", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text.strip().lower())
        return text

    payload = normalize(reason) + "||" + normalize(likely_reject_phrase)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def build_feedback_template(
    *,
    run_id: str,
    paper_title: str,
    venue: str,
    year: int,
    risks: list[dict],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for risk in risks:
        reason = str(risk.get("reason", ""))
        reject_phrase = str(risk.get("likely_reject_phrase", ""))
        items.append(
            {
                "risk_id": str(risk.get("id", "")),
                "risk_fingerprint": make_risk_fingerprint(reason, reject_phrase),
                "reason": reason,
                "likely_reject_phrase": reject_phrase,
                "fix_hint": str(risk.get("fix_hint", "")),
                "verdict": "pending",  # set to correct|incorrect|pending
                "comment": "",
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "paper_title": paper_title,
        "venue": venue,
        "year": year,
        "items": items,
    }


def submit_feedback(repo_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    venue = str(payload.get("venue", "")).strip()
    year = int(payload.get("year", 0) or 0)
    run_id = str(payload.get("run_id", "")).strip()
    paper_title = str(payload.get("paper_title", "")).strip()
    items = payload.get("items", [])
    if not venue or year <= 0:
        raise ValueError("feedback payload missing venue/year")
    if not isinstance(items, list):
        raise ValueError("feedback payload items must be a list")

    normalized_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        verdict_raw = str(item.get("verdict", "pending")).strip().lower()
        if verdict_raw not in {"correct", "incorrect", "pending"}:
            continue
        if verdict_raw == "pending":
            continue
        reason = str(item.get("reason", "")).strip()
        reject_phrase = str(item.get("likely_reject_phrase", "")).strip()
        fingerprint = str(item.get("risk_fingerprint", "")).strip()
        if not fingerprint:
            fingerprint = make_risk_fingerprint(reason, reject_phrase)
        normalized_items.append(
            {
                "risk_id": str(item.get("risk_id", "")).strip(),
                "risk_fingerprint": fingerprint,
                "reason": reason,
                "likely_reject_phrase": reject_phrase,
                "verdict": verdict_raw,
                "comment": str(item.get("comment", "")).strip(),
            }
        )

    submitted_at = datetime.now(timezone.utc).isoformat()
    record = {
        "schema_version": 1,
        "submitted_at": submitted_at,
        "run_id": run_id,
        "paper_title": paper_title,
        "venue": venue,
        "year": year,
        "items": normalized_items,
    }

    venue_slug = normalize_venue_slug(venue)
    out_dir = feedback_root(repo_root) / venue_slug / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{run_id or 'manual'}.json"
    out_path = out_dir / filename
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"saved_to": str(out_path), "accepted_items": len(normalized_items)}


def load_feedback_records(repo_root: Path, venue: str, year: int) -> list[dict[str, Any]]:
    venue_slug = normalize_venue_slug(venue)
    base = feedback_root(repo_root) / venue_slug
    if not base.exists():
        return []

    files: list[Path] = []
    year_dir = base / str(year)
    if year_dir.exists():
        files.extend(sorted(year_dir.glob("*.json")))

    # Include nearby years as weak signal.
    for fallback_year in (year - 1, year + 1):
        d = base / str(fallback_year)
        if d.exists():
            files.extend(sorted(d.glob("*.json")))

    out: list[dict[str, Any]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def build_feedback_profile(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    profile: dict[str, dict[str, Any]] = {}
    for record in records:
        items = record.get("items", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            fp = str(item.get("risk_fingerprint", "")).strip()
            verdict = str(item.get("verdict", "")).strip().lower()
            if not fp or verdict not in {"correct", "incorrect"}:
                continue
            bucket = profile.setdefault(
                fp,
                {"correct": 0, "incorrect": 0, "comments": []},
            )
            bucket[verdict] += 1
            comment = str(item.get("comment", "")).strip()
            if comment:
                bucket["comments"].append(comment)
    return profile


def apply_feedback_profile(risks: list[dict], profile: dict[str, dict[str, Any]]) -> tuple[list[dict], dict[str, Any]]:
    adjusted: list[dict] = []
    signals = {
        "matched_risks": 0,
        "profiles_loaded": len(profile),
        "adjustments": [],
    }
    for risk in risks:
        row = dict(risk)
        reason = str(row.get("reason", ""))
        phrase = str(row.get("likely_reject_phrase", ""))
        fp = make_risk_fingerprint(reason, phrase)
        row["risk_fingerprint"] = fp
        stat = profile.get(fp)
        if not stat:
            adjusted.append(row)
            continue

        correct = int(stat.get("correct", 0) or 0)
        incorrect = int(stat.get("incorrect", 0) or 0)
        total = correct + incorrect
        if total <= 0:
            adjusted.append(row)
            continue

        signals["matched_risks"] += 1
        original = float(row.get("score", 0.0) or 0.0)
        new_score = original
        action = "none"

        incorrect_ratio = incorrect / total
        correct_ratio = correct / total
        if total >= 2 and incorrect_ratio >= 0.6:
            new_score = max(0.0, original - 0.08)
            action = "down"
        elif total >= 3 and correct_ratio >= 0.7:
            new_score = min(1.0, original + 0.05)
            action = "up"

        if action != "none":
            row["score"] = round(new_score, 3)
            row["feedback_adjustment"] = {
                "action": action,
                "original_score": round(original, 3),
                "correct": correct,
                "incorrect": incorrect,
                "comments": stat.get("comments", [])[:2],
            }
            row["severity"] = _severity_from_score(new_score)
            signals["adjustments"].append(
                {
                    "risk_id": row.get("id", ""),
                    "action": action,
                    "original_score": round(original, 3),
                    "new_score": round(new_score, 3),
                    "correct": correct,
                    "incorrect": incorrect,
                }
            )
        adjusted.append(row)

    adjusted.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    return adjusted, signals


def _severity_from_score(score: float) -> str:
    if score >= 0.75:
        return "P0"
    if score >= 0.45:
        return "P1"
    return "P2"
