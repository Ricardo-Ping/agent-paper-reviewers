from __future__ import annotations

from typing import Any

from agent_paper_reviewers.services.venue_sync import OpenReviewPolicyResolver


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


def test_openreview_policy_extracts_real_fields(monkeypatch) -> None:
    def fake_get(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 10):
        if url.endswith("/groups"):
            return _FakeResponse(
                200,
                {
                    "groups": [
                        {
                            "id": "ICLR.cc/2026/Conference",
                            "details": {
                                "review_guidance": "Review novelty, technical correctness, empirical evaluation, and clarity.",
                                "weights": {
                                    "novelty": 25,
                                    "soundness": 35,
                                    "experiment": 25,
                                    "clarity": 15,
                                },
                                "rebuttal": {
                                    "maxLength": 5000,
                                    "allowLinks": True,
                                },
                                "weakness_examples": [
                                    "Missing baseline comparison against strong methods.",
                                    "Statistical significance details are insufficient.",
                                ],
                            },
                        }
                    ]
                },
            )
        if url.endswith("/invitations") and params and params.get("prefix"):
            return _FakeResponse(
                200,
                {
                    "invitations": [
                        {
                            "id": "ICLR.cc/2026/Conference/-/Author_Response",
                            "reply": {"content": {"response": {"value": {"param": {"maxLength": 5000}}}}},
                        }
                    ]
                },
            )
        if url.endswith("/invitations") and params and params.get("invitee"):
            return _FakeResponse(200, {"invitations": []})
        return _FakeResponse(404, {})

    monkeypatch.setattr("agent_paper_reviewers.services.venue_sync.requests.get", fake_get)

    resolver = OpenReviewPolicyResolver(base_url="https://api2.openreview.net", token="fake-token")
    result = resolver.resolve_policy("ICLR.cc/2026/Conference")

    assert result.policy is not None
    assert result.policy.per_review_char_limit == 5000
    assert result.policy.dynamic_from_openreview is True
    assert result.profile_overrides is not None
    assert result.profile_overrides["scoring_axes"] == ["novelty", "soundness", "experiment", "clarity"]

    weights = result.profile_overrides["weights"]
    assert abs(sum(weights.values()) - 1.0) < 0.0001
    assert weights["soundness"] > weights["clarity"]

    reasons = result.profile_overrides["common_reject_reasons"]
    assert any("baseline" in r.lower() for r in reasons)
    assert any("significance" in r.lower() for r in reasons)
    assert (result.warning or "") != "policy_not_extracted_use_fallback"


def test_openreview_policy_handles_forbidden_without_token_as_soft_fallback(monkeypatch) -> None:
    def fake_get(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 10):
        return _FakeResponse(403, {})

    monkeypatch.setattr("agent_paper_reviewers.services.venue_sync.requests.get", fake_get)

    resolver = OpenReviewPolicyResolver(base_url="https://api2.openreview.net")
    result = resolver.resolve_policy("ICLR.cc/2026/Conference")

    assert result.policy is None
    assert result.warning is None


def test_openreview_policy_handles_forbidden_with_token(monkeypatch) -> None:
    def fake_get(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 10):
        return _FakeResponse(403, {})

    monkeypatch.setattr("agent_paper_reviewers.services.venue_sync.requests.get", fake_get)

    resolver = OpenReviewPolicyResolver(base_url="https://api2.openreview.net", token="fake-token")
    result = resolver.resolve_policy("ICLR.cc/2026/Conference")

    assert result.policy is None
    assert result.warning is not None
    assert "openreview_forbidden_with_token" in result.warning


def test_openreview_policy_can_discover_group_by_venue(monkeypatch) -> None:
    def fake_get(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 10):
        if url.endswith("/groups"):
            group_id = (params or {}).get("id", "")
            if group_id == "ICLR.cc/2026/Conference":
                return _FakeResponse(200, {"groups": [{"id": group_id, "details": {}}]})
            return _FakeResponse(200, {"groups": []})
        if url.endswith("/invitations"):
            return _FakeResponse(200, {"invitations": []})
        if url.endswith("/notes"):
            return _FakeResponse(200, {"notes": []})
        return _FakeResponse(404, {})

    monkeypatch.setattr("agent_paper_reviewers.services.venue_sync.requests.get", fake_get)

    resolver = OpenReviewPolicyResolver(base_url="https://api2.openreview.net", token="fake-token")
    result = resolver.resolve_policy_by_venue("iclr", 2026)
    assert result.resolved_group_id == "ICLR.cc/2026/Conference"
