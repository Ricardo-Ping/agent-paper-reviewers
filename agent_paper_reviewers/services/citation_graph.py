from __future__ import annotations

import os
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

import requests


class SemanticScholarClient:
    def __init__(self) -> None:
        self.base_url = os.getenv(
            "AGENT_PAPER_REVIEWERS_SEMANTIC_SCHOLAR_BASE_URL",
            "https://api.semanticscholar.org/graph/v1",
        ).rstrip("/")
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        self.timeout = int(os.getenv("AGENT_PAPER_REVIEWERS_S2_TIMEOUT_SECONDS", "12"))
        self.enabled = os.getenv("AGENT_PAPER_REVIEWERS_DISABLE_S2", "0") not in {"1", "true", "TRUE"}

    def fetch_citation_graph(self, title: str) -> tuple[dict[str, Any] | None, list[str]]:
        warnings: list[str] = []
        if not self.enabled:
            warnings.append("semantic_scholar_disabled_by_env")
            return None, warnings
        if not title.strip():
            warnings.append("semantic_scholar_title_missing")
            return None, warnings

        search = self._search_paper_by_title(title)
        if search["error"]:
            warnings.append(search["error"])
            return None, warnings
        if not search["data"]:
            warnings.append("semantic_scholar_no_search_result")
            return None, warnings

        candidates = search["data"]
        best = self._pick_best_candidate(title, candidates)
        paper_id = best.get("paperId")
        if not paper_id:
            warnings.append("semantic_scholar_missing_paper_id")
            return None, warnings

        details = self._get_paper_details(str(paper_id))
        if details["error"]:
            warnings.append(details["error"])
            # Still return the selected search result as minimal citation graph.
            return {
                "paper": {
                    "paper_id": best.get("paperId"),
                    "title": best.get("title", ""),
                    "year": best.get("year"),
                    "url": best.get("url", ""),
                },
                "outgoing_references": [],
                "incoming_citations": [],
                "stats": {
                    "outgoing_count": int(best.get("referenceCount") or 0),
                    "incoming_count": int(best.get("citationCount") or 0),
                    "baseline_like_reference_count": 0,
                    "reference_coverage_score": 0.0,
                    "novelty_signal_score": 0.0,
                },
                "source": "semantic_scholar_search_only",
            }, warnings

        paper = details["data"]
        outgoing = self._parse_relation_list(paper.get("references", []), relation="references")
        incoming = self._parse_relation_list(paper.get("citations", []), relation="citations")

        for item in outgoing:
            item["is_baseline_candidate"] = self._is_baseline_candidate(item.get("title", ""))

        stats = self._build_stats(
            outgoing_count=len(outgoing),
            incoming_count=len(incoming),
            baseline_like_count=sum(1 for x in outgoing if x.get("is_baseline_candidate")),
            paper_year=paper.get("year"),
        )

        return {
            "paper": {
                "paper_id": paper.get("paperId", ""),
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "url": paper.get("url", ""),
                "venue": paper.get("venue", ""),
            },
            "outgoing_references": outgoing,
            "incoming_citations": incoming,
            "stats": stats,
            "source": "semantic_scholar",
        }, warnings

    def _search_paper_by_title(self, title: str) -> dict[str, Any]:
        params = {
            "query": title,
            "limit": 5,
            "fields": "paperId,title,year,url,citationCount,referenceCount",
        }
        data, error = self._request("/paper/search", params=params)
        if error:
            return {"data": None, "error": error}
        rows = data.get("data", []) if isinstance(data, dict) else []
        return {"data": rows, "error": None}

    def _get_paper_details(self, paper_id: str) -> dict[str, Any]:
        params = {
            "fields": (
                "paperId,title,year,url,venue,citationCount,referenceCount,"
                "references.paperId,references.title,references.year,"
                "citations.paperId,citations.title,citations.year"
            )
        }
        data, error = self._request(f"/paper/{paper_id}", params=params)
        return {"data": data, "error": error}

    def _request(self, path: str, *, params: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        headers = {"User-Agent": "agent-paper-reviewers/0.1"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        for attempt in range(2):
            try:
                resp = requests.get(
                    f"{self.base_url}{path}",
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                return None, f"semantic_scholar_request_failed:{exc}"

            if resp.status_code == 200:
                try:
                    return resp.json(), None
                except ValueError:
                    return None, "semantic_scholar_invalid_json"

            if resp.status_code == 429 and attempt == 0:
                time.sleep(1.0)
                continue
            if resp.status_code == 401:
                return None, "semantic_scholar_unauthorized_check_api_key"
            if resp.status_code == 403:
                return None, "semantic_scholar_forbidden_check_api_key_or_quota"
            if resp.status_code == 404:
                return None, "semantic_scholar_not_found"
            return None, f"semantic_scholar_status_{resp.status_code}"

        return None, "semantic_scholar_rate_limited"

    @staticmethod
    def _pick_best_candidate(title: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            return {}
        title_l = title.lower().strip()
        best = candidates[0]
        best_score = -1.0
        for row in candidates:
            cand_title = str(row.get("title", "")).lower().strip()
            sim = SequenceMatcher(None, title_l, cand_title).ratio()
            if sim > best_score:
                best_score = sim
                best = row
        return best

    @staticmethod
    def _parse_relation_list(items: Any, *, relation: str) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        rows: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            # Support both flattened and nested response shapes.
            nested_key = "citedPaper" if relation == "references" else "citingPaper"
            nested = item.get(nested_key) if isinstance(item.get(nested_key), dict) else {}
            paper_id = item.get("paperId") or nested.get("paperId") or ""
            title = item.get("title") or nested.get("title") or ""
            year = item.get("year") or nested.get("year")
            if not title and not paper_id:
                continue
            rows.append(
                {
                    "paper_id": str(paper_id),
                    "title": str(title).strip(),
                    "year": year,
                }
            )
        return rows

    @staticmethod
    def _is_baseline_candidate(title: str) -> bool:
        t = title.lower()
        baseline_keywords = [
            "baseline",
            "benchmark",
            "state of the art",
            "sota",
            "comparison",
            "evaluation",
            "dataset",
            "translation",
            "model",
        ]
        return any(k in t for k in baseline_keywords)

    @staticmethod
    def _build_stats(
        *,
        outgoing_count: int,
        incoming_count: int,
        baseline_like_count: int,
        paper_year: Any,
    ) -> dict[str, Any]:
        outgoing_norm = min(1.0, outgoing_count / 25.0)
        baseline_norm = min(1.0, baseline_like_count / max(3.0, outgoing_count * 0.25 or 1.0))
        reference_coverage_score = round(0.6 * outgoing_norm + 0.4 * baseline_norm, 3)

        novelty_signal_score = 0.0
        try:
            year_int = int(paper_year)
        except (TypeError, ValueError):
            year_int = None
        now_year = datetime.now().year
        if incoming_count > 0:
            novelty_signal_score = round(min(1.0, incoming_count / 40.0), 3)
        elif year_int is not None and now_year - year_int <= 1:
            novelty_signal_score = 0.5
        elif year_int is not None and now_year - year_int <= 3:
            novelty_signal_score = 0.2
        else:
            novelty_signal_score = 0.1

        return {
            "outgoing_count": int(outgoing_count),
            "incoming_count": int(incoming_count),
            "baseline_like_reference_count": int(baseline_like_count),
            "reference_coverage_score": reference_coverage_score,
            "novelty_signal_score": novelty_signal_score,
        }


def extract_local_references(raw_text: str) -> list[dict[str, Any]]:
    if not raw_text.strip():
        return []

    text = raw_text.replace("\r\n", "\n")
    lower = text.lower()
    idx = max(lower.rfind("\nreferences"), lower.rfind("\nbibliography"))
    if idx < 0:
        return []
    tail = text[idx:]
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return []

    refs: list[str] = []
    cur = ""
    for line in lines[1:]:
        if re.match(r"^(\[\d+\]|\d+\.)\s+", line):
            if cur:
                refs.append(cur.strip())
            cur = re.sub(r"^(\[\d+\]|\d+\.)\s+", "", line).strip()
        else:
            if len(line) < 5:
                continue
            if cur:
                cur = f"{cur} {line}"
            else:
                cur = line
    if cur:
        refs.append(cur.strip())

    rows = []
    for ref in refs:
        if len(ref) < 20:
            continue
        year_match = re.search(r"(19|20)\d{2}", ref)
        year = int(year_match.group(0)) if year_match else None
        rows.append({"paper_id": "", "title": ref[:320], "year": year})
    return rows


def build_citation_graph(structured_paper: dict[str, Any]) -> dict[str, Any]:
    title = str(structured_paper.get("title") or "").strip()
    raw_text = str(structured_paper.get("raw_text") or "")

    client = SemanticScholarClient()
    remote_graph, warnings = client.fetch_citation_graph(title)
    local_refs = extract_local_references(raw_text)

    if remote_graph is None:
        baseline_like = sum(1 for x in local_refs if client._is_baseline_candidate(x.get("title", "")))
        stats = client._build_stats(
            outgoing_count=len(local_refs),
            incoming_count=0,
            baseline_like_count=baseline_like,
            paper_year=None,
        )
        return {
            "paper": {"paper_id": "", "title": title, "year": None, "url": "", "venue": ""},
            "outgoing_references": local_refs,
            "incoming_citations": [],
            "stats": stats,
            "source": "local_only" if local_refs else "none",
            "warnings": warnings,
        }

    merged_outgoing = list(remote_graph.get("outgoing_references", []))
    if local_refs:
        existing_titles = {str(x.get("title", "")).lower().strip() for x in merged_outgoing}
        for row in local_refs:
            t = str(row.get("title", "")).lower().strip()
            if t and t not in existing_titles:
                row = dict(row)
                row["is_baseline_candidate"] = client._is_baseline_candidate(row.get("title", ""))
                merged_outgoing.append(row)
                existing_titles.add(t)

    baseline_like = sum(1 for x in merged_outgoing if x.get("is_baseline_candidate"))
    stats = client._build_stats(
        outgoing_count=len(merged_outgoing),
        incoming_count=int(remote_graph.get("stats", {}).get("incoming_count", 0)),
        baseline_like_count=baseline_like,
        paper_year=remote_graph.get("paper", {}).get("year"),
    )

    source = "hybrid" if local_refs else remote_graph.get("source", "semantic_scholar")
    return {
        "paper": remote_graph.get("paper", {}),
        "outgoing_references": merged_outgoing,
        "incoming_citations": remote_graph.get("incoming_citations", []),
        "stats": stats,
        "source": source,
        "warnings": warnings,
    }
