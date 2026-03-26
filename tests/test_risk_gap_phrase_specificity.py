from __future__ import annotations

from agent_paper_reviewers.executors.deterministic import DeterministicExecutor
from agent_paper_reviewers.pipeline.step_risk_ranker import RiskRankerStep


def test_gap_phrase_and_fix_are_specific_for_non_core_fallback_gaps() -> None:
    gap = {"code": "missing_limitations"}
    phrase = RiskRankerStep._likely_reject_phrase_for_gap(gap)
    fix = RiskRankerStep._fix_hint_for_gap(gap)

    assert "limitations" in phrase.lower()
    assert "limitations" in fix.lower()

    gap2 = {"code": "missing_system_setting_reproducibility"}
    phrase2 = RiskRankerStep._likely_reject_phrase_for_gap(gap2)
    fix2 = RiskRankerStep._fix_hint_for_gap(gap2)

    assert "configuration" in phrase2.lower() or "system" in phrase2.lower()
    assert "environment" in fix2.lower() or "configuration" in fix2.lower()


def test_deterministic_gap_phrase_and_fix_are_specific() -> None:
    phrase = DeterministicExecutor._reject_phrase_for_gap("missing_limitations")
    fix = DeterministicExecutor._fix_hint_for_gap("missing_limitations")
    assert "limitations" in phrase.lower()
    assert "limitations" in fix.lower() or "scope" in fix.lower()

    phrase2 = DeterministicExecutor._reject_phrase_for_gap("missing_top_venue_related_work_coverage")
    fix2 = DeterministicExecutor._fix_hint_for_gap("missing_top_venue_related_work_coverage")
    assert "top-venue" in phrase2.lower() or "recent" in phrase2.lower()
    assert "top-venue" in fix2.lower() or "last 2-3 years" in fix2.lower()

    phrase3 = DeterministicExecutor._reject_phrase_for_gap("terminology_inconsistency")
    fix3 = DeterministicExecutor._fix_hint_for_gap("terminology_inconsistency")
    assert "terminology" in phrase3.lower() or "acronym" in phrase3.lower()
    assert "terminology" in fix3.lower() or "notation" in fix3.lower()
