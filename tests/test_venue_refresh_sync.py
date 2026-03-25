from __future__ import annotations

from pathlib import Path

import yaml

from agent_paper_reviewers.mcp.base import PolicyResolveResult
from agent_paper_reviewers.models import RebuttalPolicy, VenueRuleSnapshot
from agent_paper_reviewers.services.venue_loader import load_venue_snapshot
from agent_paper_reviewers.services.venue_sync import refresh_venue_rules


def _write_snapshot(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "venue": "iclr",
        "year": 2026,
        "display_name": "ICLR",
        "profile": {
            "scoring_axes": ["novelty", "soundness", "experiment", "clarity"],
            "weights": {"novelty": 0.25, "soundness": 0.3, "experiment": 0.3, "clarity": 0.15},
            "common_reject_reasons": ["Old reason"],
            "required_checks": ["baseline_coverage"],
            "rebuttal_policy": {
                "mode": "per_review_only",
                "per_review_char_limit": 2500,
                "global_char_limit": 0,
                "allow_attachment_pdf": False,
                "attachment_page_limit": 0,
                "allow_links": False,
                "dynamic_from_openreview": True,
            },
            "openreview_group_id": "ICLR.cc/2026/Conference",
            "version_date": "2026-01-01",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_fallback(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "venue": "_fallback",
        "year": 0,
        "display_name": "GlobalFallback",
        "profile": {
            "scoring_axes": ["novelty", "soundness", "experiment", "clarity"],
            "weights": {"novelty": 0.25, "soundness": 0.3, "experiment": 0.3, "clarity": 0.15},
            "common_reject_reasons": ["fallback"],
            "required_checks": ["baseline_coverage"],
            "rebuttal_policy": {
                "mode": "per_review_only",
                "per_review_char_limit": 2500,
                "global_char_limit": 0,
                "allow_attachment_pdf": False,
                "attachment_page_limit": 0,
                "allow_links": False,
                "dynamic_from_openreview": False,
            },
            "openreview_group_id": "",
            "version_date": "2026-01-01",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_refresh_venue_rules_writes_updated_snapshot(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path
    rules_root = repo / "data" / "venue_rules"
    _write_snapshot(rules_root / "iclr" / "2026.yaml")
    _write_fallback(rules_root / "_fallback.yaml")

    def fake_resolve(self, group_id: str):
        return PolicyResolveResult(
            policy=RebuttalPolicy(
                mode="per_review_only",
                per_review_char_limit=5000,
                global_char_limit=0,
                allow_attachment_pdf=False,
                attachment_page_limit=0,
                allow_links=True,
                dynamic_from_openreview=True,
            ),
            profile_overrides={
                "scoring_axes": ["novelty", "soundness", "clarity"],
                "weights": {"novelty": 4, "soundness": 4, "clarity": 2},
                "common_reject_reasons": ["Refreshed reason"],
            },
            warning=None,
        )

    monkeypatch.setattr(
        "agent_paper_reviewers.mcp.http_provider.HttpMCPToolProvider.resolve_openreview_policy",
        fake_resolve,
    )

    summary = refresh_venue_rules(repo, venue="iclr", year=2026, dry_run=False)
    assert summary["updated_count"] == 1
    assert summary["failed_count"] == 0

    snap = load_venue_snapshot(rules_root / "iclr" / "2026.yaml")
    assert isinstance(snap, VenueRuleSnapshot)
    assert snap.profile.rebuttal_policy.per_review_char_limit == 5000
    assert snap.profile.scoring_axes == ["novelty", "soundness", "clarity"]
    assert snap.profile.common_reject_reasons[0] == "Refreshed reason"
