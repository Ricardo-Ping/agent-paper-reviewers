from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

import requests

TOP_VENUE_ALIASES: dict[str, tuple[str, ...]] = {
    "neurips": ("neurips", "nips"),
    "icml": ("icml",),
    "iclr": ("iclr",),
    "kdd": ("kdd", "sigkdd"),
    "aaai": ("aaai",),
    "acl": ("acl", "naacl", "eacl"),
    "emnlp": ("emnlp",),
    "cvpr": ("cvpr",),
    "eccv": ("eccv",),
    "sigmod": ("sigmod", "acm sigmod", "proc. acm manag. data"),
    "vldb": ("vldb", "pvldb", "very large data bases"),
    "icde": ("icde", "ieee icde"),
    "pods": ("pods",),
}


def _canonicalize_venue(venue: str) -> str:
    value = re.sub(r"[^a-z0-9]+", " ", venue.lower()).strip()
    if not value:
        return ""
    for canonical, aliases in TOP_VENUE_ALIASES.items():
        if any(alias in value for alias in aliases):
            return canonical
    return value[:80]


def _infer_venue_from_text(text: str) -> str:
    lowered = text.lower()
    for canonical, aliases in TOP_VENUE_ALIASES.items():
        for alias in aliases:
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, lowered):
                return canonical
    return ""


def _coerce_year(value: Any) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if 1900 <= year <= datetime.now().year + 1:
        return year
    return None


def _extract_focus_text(structured_paper: dict[str, Any]) -> dict[str, str]:
    sections = structured_paper.get("sections", []) if isinstance(structured_paper, dict) else []
    raw_text = str(structured_paper.get("raw_text") or "")
    buckets = {
        "abstract": "",
        "introduction": "",
        "method": "",
        "experiments": "",
        "conclusion": "",
    }
    if isinstance(sections, list):
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            name = str(sec.get("name", "")).strip().lower()
            text = str(sec.get("text", "")).strip()
            if not text:
                continue
            if "abstract" in name:
                buckets["abstract"] += "\n" + text
            elif "intro" in name:
                buckets["introduction"] += "\n" + text
            elif any(k in name for k in ["method", "approach", "model", "system"]):
                buckets["method"] += "\n" + text
            elif any(k in name for k in ["experiment", "evaluation", "result"]):
                buckets["experiments"] += "\n" + text
            elif any(k in name for k in ["conclusion", "discussion"]):
                buckets["conclusion"] += "\n" + text

    if not any(v.strip() for v in buckets.values()):
        lower = raw_text.lower()
        buckets["abstract"] = raw_text[: min(len(raw_text), 1800)]
        if "introduction" in lower:
            pos = lower.find("introduction")
            buckets["introduction"] = raw_text[pos : pos + 2200]
        if "conclusion" in lower:
            pos = lower.find("conclusion")
            buckets["conclusion"] = raw_text[pos : pos + 1800]
    return {k: v.strip() for k, v in buckets.items()}


def _keyword_hit_score(text: str, keywords: list[str], *, cap: int = 4) -> tuple[float, int]:
    t = text.lower()
    hits = 0
    for kw in keywords:
        if kw in t:
            hits += 1
    score = min(1.0, hits / max(1, cap))
    return score, hits


def _content_novelty_signals(structured_paper: dict[str, Any]) -> dict[str, Any]:
    buckets = _extract_focus_text(structured_paper)
    intro_pack = "\n".join(
        [
            buckets.get("abstract", ""),
            buckets.get("introduction", ""),
            buckets.get("method", ""),
            buckets.get("conclusion", ""),
        ]
    )
    exp_pack = "\n".join([buckets.get("experiments", ""), buckets.get("method", "")])

    method_keywords = [
        "we propose",
        "novel",
        "new",
        "first",
        "framework",
        "architecture",
        "module",
        "mechanism",
        "paradigm",
        "algorithm",
        "system",
    ]
    task_keywords = [
        "new task",
        "new benchmark",
        "new dataset",
        "problem setting",
        "cross-domain",
        "cross-dialect",
        "unexplored",
        "first benchmark",
    ]
    system_keywords = [
        "end-to-end",
        "production",
        "deployment",
        "prototype",
        "engine",
        "pipeline",
        "platform",
        "translator",
    ]
    evidence_keywords = [
        "ablation",
        "baseline",
        "significance",
        "p-value",
        "confidence interval",
        "benchmark",
        "error analysis",
        "limitations",
    ]

    method_score, method_hits = _keyword_hit_score(intro_pack, method_keywords, cap=5)
    task_score, task_hits = _keyword_hit_score(intro_pack, task_keywords, cap=3)
    system_score, system_hits = _keyword_hit_score(intro_pack, system_keywords, cap=4)
    evidence_score, evidence_hits = _keyword_hit_score(exp_pack, evidence_keywords, cap=4)

    content_score = (
        0.40 * method_score
        + 0.25 * task_score
        + 0.25 * system_score
        + 0.10 * evidence_score
    )

    return {
        "content_novelty_score": round(min(1.0, content_score), 3),
        "content_novelty_components": {
            "method_component_score": round(method_score, 3),
            "task_component_score": round(task_score, 3),
            "system_component_score": round(system_score, 3),
            "evidence_component_score": round(evidence_score, 3),
            "keyword_hits": {
                "method": method_hits,
                "task": task_hits,
                "system": system_hits,
                "evidence": evidence_hits,
            },
        },
    }


def filter_references_by_venue_year(
    references: list[dict[str, Any]],
    *,
    venue: str | None = None,
    year: int | None = None,
) -> list[dict[str, Any]]:
    venue_key = _canonicalize_venue(venue or "") if venue else ""
    out: list[dict[str, Any]] = []
    for ref in references:
        if not isinstance(ref, dict):
            continue
        ref_venue = str(ref.get("venue") or "").strip().lower()
        ref_year = _coerce_year(ref.get("year"))
        if venue_key and ref_venue != venue_key:
            continue
        if year is not None and ref_year != int(year):
            continue
        out.append(ref)
    return out


def _build_venue_filtered_reference_stats(references: list[dict[str, Any]]) -> dict[str, Any]:
    now_year = datetime.now().year
    venue_counter: Counter[str] = Counter()
    venue_year_counter: Counter[str] = Counter()
    top_venue_reference_count = 0
    recent_top_venue_reference_count = 0

    for row in references:
        if not isinstance(row, dict):
            continue
        venue = str(row.get("venue") or "").strip().lower()
        year = _coerce_year(row.get("year"))
        if not venue:
            continue
        venue_counter[venue] += 1
        if year is not None:
            venue_year_counter[f"{venue}:{year}"] += 1

        if venue in TOP_VENUE_ALIASES:
            top_venue_reference_count += 1
            if year is not None and now_year - year <= 3:
                recent_top_venue_reference_count += 1

    return {
        "venue_reference_counts": dict(sorted(venue_counter.items())),
        "venue_year_reference_counts": dict(sorted(venue_year_counter.items())),
        "top_venue_reference_count": int(top_venue_reference_count),
        "recent_top_venue_reference_count": int(recent_top_venue_reference_count),
    }


class SemanticScholarClient:
    def __init__(self) -> None:
        self.base_url = os.getenv(
            "AGENT_PAPER_REVIEWERS_SEMANTIC_SCHOLAR_BASE_URL",
            "https://api.semanticscholar.org/graph/v1",
        ).rstrip("/")
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        self.timeout = int(os.getenv("AGENT_PAPER_REVIEWERS_S2_TIMEOUT_SECONDS", "12"))
        self.max_retries = max(1, int(os.getenv("AGENT_PAPER_REVIEWERS_S2_MAX_RETRIES", "3")))
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
                    "venue": _canonicalize_venue(str(best.get("venue", ""))),
                },
                "outgoing_references": [],
                "incoming_citations": [],
                "stats": self._build_stats(
                    outgoing_references=[],
                    incoming_citations=[],
                    baseline_like_count=0,
                    paper_year=best.get("year"),
                    outgoing_count_override=int(best.get("referenceCount") or 0),
                    incoming_count_override=int(best.get("citationCount") or 0),
                ),
                "source": "semantic_scholar_search_only",
            }, warnings

        paper = details["data"]
        outgoing = self._parse_relation_list(paper.get("references", []), relation="references")
        incoming = self._parse_relation_list(paper.get("citations", []), relation="citations")

        for item in outgoing:
            item["is_baseline_candidate"] = self._is_baseline_candidate(item.get("title", ""))

        stats = self._build_stats(
            outgoing_references=outgoing,
            incoming_citations=incoming,
            baseline_like_count=sum(1 for x in outgoing if x.get("is_baseline_candidate")),
            paper_year=paper.get("year"),
        )

        return {
            "paper": {
                "paper_id": paper.get("paperId", ""),
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "url": paper.get("url", ""),
                "venue": _canonicalize_venue(str(paper.get("venue", ""))),
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
            "fields": "paperId,title,year,url,venue,citationCount,referenceCount",
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
                "references.paperId,references.title,references.year,references.venue,"
                "citations.paperId,citations.title,citations.year,citations.venue"
            )
        }
        data, error = self._request(f"/paper/{paper_id}", params=params)
        return {"data": data, "error": error}

    def _request(self, path: str, *, params: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        headers = {"User-Agent": "agent-paper-reviewers/0.1"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        for attempt in range(self.max_retries):
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

            if resp.status_code in {429, 503} and attempt < self.max_retries - 1:
                retry_after = resp.headers.get("Retry-After")
                try:
                    sleep_s = float(retry_after) if retry_after else (1.0 * (2**attempt))
                except ValueError:
                    sleep_s = 1.0 * (2**attempt)
                time.sleep(max(0.5, min(6.0, sleep_s)))
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
            venue_raw = item.get("venue") or nested.get("venue") or ""
            venue = _canonicalize_venue(str(venue_raw))
            if not venue and title:
                venue = _infer_venue_from_text(str(title))
            if not title and not paper_id:
                continue
            rows.append(
                {
                    "paper_id": str(paper_id),
                    "title": str(title).strip(),
                    "year": _coerce_year(year),
                    "venue": venue,
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
        outgoing_references: list[dict[str, Any]],
        incoming_citations: list[dict[str, Any]],
        baseline_like_count: int,
        paper_year: Any,
        content_novelty_signal: dict[str, Any] | None = None,
        outgoing_count_override: int | None = None,
        incoming_count_override: int | None = None,
    ) -> dict[str, Any]:
        outgoing_count = int(outgoing_count_override) if outgoing_count_override is not None else len(outgoing_references)
        incoming_count = int(incoming_count_override) if incoming_count_override is not None else len(incoming_citations)

        outgoing_norm = min(1.0, outgoing_count / 25.0)
        baseline_norm = min(1.0, baseline_like_count / max(3.0, outgoing_count * 0.25 or 1.0))
        reference_coverage_score = round(0.6 * outgoing_norm + 0.4 * baseline_norm, 3)

        year_int = _coerce_year(paper_year)
        now_year = datetime.now().year

        venue_stats = _build_venue_filtered_reference_stats(outgoing_references)
        top_venue_ref_count = int(venue_stats["top_venue_reference_count"])
        recent_top_venue_ref_count = int(venue_stats["recent_top_venue_reference_count"])

        recent_reference_count = 0
        venue_set: set[str] = set()
        for row in outgoing_references:
            if not isinstance(row, dict):
                continue
            year = _coerce_year(row.get("year"))
            if year is not None and now_year - year <= 3:
                recent_reference_count += 1
            venue = str(row.get("venue") or "").strip().lower()
            if venue:
                venue_set.add(venue)

        incoming_norm = min(1.0, math.log1p(max(0, incoming_count)) / math.log1p(40))
        if outgoing_count > 0:
            recent_ref_ratio = recent_reference_count / outgoing_count
        else:
            recent_ref_ratio = 0.0
        recent_ref_norm = min(1.0, 0.5 * recent_ref_ratio + 0.5 * min(1.0, recent_reference_count / 8.0))
        venue_diversity_norm = min(1.0, len(venue_set) / 6.0)
        outgoing_depth_norm = min(1.0, outgoing_count / 25.0)

        if year_int is None:
            paper_age = None
        else:
            paper_age = max(0, now_year - year_int)

        if paper_age is None:
            w_in, w_recent, w_div, w_depth = 0.35, 0.25, 0.2, 0.2
        elif paper_age <= 1:
            w_in, w_recent, w_div, w_depth = 0.15, 0.35, 0.25, 0.25
        elif paper_age <= 3:
            w_in, w_recent, w_div, w_depth = 0.3, 0.3, 0.2, 0.2
        else:
            w_in, w_recent, w_div, w_depth = 0.5, 0.2, 0.15, 0.15

        citation_novelty_score = (
            w_in * incoming_norm
            + w_recent * recent_ref_norm
            + w_div * venue_diversity_norm
            + w_depth * outgoing_depth_norm
        )

        content_signal = content_novelty_signal if isinstance(content_novelty_signal, dict) else {}
        content_novelty_score = float(content_signal.get("content_novelty_score", 0.0) or 0.0)
        content_novelty_score = max(0.0, min(1.0, content_novelty_score))

        # Blend citation-based and paper-content-based novelty evidence.
        # For new/low-citation papers, content signal should carry more weight.
        if incoming_count == 0:
            alpha_citation = 0.45
        else:
            alpha_citation = 0.70
        novelty_signal_score = (
            alpha_citation * citation_novelty_score
            + (1.0 - alpha_citation) * content_novelty_score
        )

        # Fairness guard for newly published papers with few or no incoming citations.
        if (paper_age is None or paper_age <= 1) and incoming_count == 0:
            if outgoing_count >= 10 and recent_top_venue_ref_count >= 2:
                novelty_signal_score = max(novelty_signal_score, 0.52)
            elif outgoing_count >= 8 and recent_reference_count >= 4:
                novelty_signal_score = max(novelty_signal_score, 0.45)
            if content_novelty_score >= 0.70:
                novelty_signal_score = max(novelty_signal_score, 0.58)
            elif content_novelty_score >= 0.55:
                novelty_signal_score = max(novelty_signal_score, 0.50)

        return {
            "outgoing_count": int(outgoing_count),
            "incoming_count": int(incoming_count),
            "baseline_like_reference_count": int(baseline_like_count),
            "reference_coverage_score": reference_coverage_score,
            "novelty_signal_score": round(min(1.0, novelty_signal_score), 3),
            "citation_novelty_score": round(min(1.0, citation_novelty_score), 3),
            "content_novelty_score": round(min(1.0, content_novelty_score), 3),
            "top_venue_reference_count": top_venue_ref_count,
            "recent_top_venue_reference_count": recent_top_venue_ref_count,
            "venue_reference_counts": venue_stats["venue_reference_counts"],
            "venue_year_reference_counts": venue_stats["venue_year_reference_counts"],
            "novelty_components": {
                "incoming_norm": round(incoming_norm, 3),
                "recent_reference_norm": round(recent_ref_norm, 3),
                "venue_diversity_norm": round(venue_diversity_norm, 3),
                "outgoing_depth_norm": round(outgoing_depth_norm, 3),
                "paper_age_years": paper_age,
                "blend_alpha_citation": round(alpha_citation, 3),
            },
            "content_novelty_components": content_signal.get("content_novelty_components", {}),
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
        venue = _infer_venue_from_text(ref)
        rows.append({"paper_id": "", "title": ref[:320], "year": year, "venue": venue})
    return rows


def build_citation_graph(structured_paper: dict[str, Any]) -> dict[str, Any]:
    title = str(structured_paper.get("title") or "").strip()
    raw_text = str(structured_paper.get("raw_text") or "")
    content_novelty = _content_novelty_signals(structured_paper)

    client = SemanticScholarClient()
    remote_graph, warnings = client.fetch_citation_graph(title)
    local_refs = extract_local_references(raw_text)

    if remote_graph is None:
        baseline_like = sum(1 for x in local_refs if client._is_baseline_candidate(x.get("title", "")))
        stats = client._build_stats(
            outgoing_references=local_refs,
            incoming_citations=[],
            baseline_like_count=baseline_like,
            paper_year=None,
            content_novelty_signal=content_novelty,
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
                if not row.get("venue"):
                    row["venue"] = _infer_venue_from_text(str(row.get("title", "")))
                row["is_baseline_candidate"] = client._is_baseline_candidate(row.get("title", ""))
                merged_outgoing.append(row)
                existing_titles.add(t)

    baseline_like = sum(1 for x in merged_outgoing if x.get("is_baseline_candidate"))
    stats = client._build_stats(
        outgoing_references=merged_outgoing,
        incoming_citations=list(remote_graph.get("incoming_citations", []) or []),
        baseline_like_count=baseline_like,
        paper_year=remote_graph.get("paper", {}).get("year"),
        content_novelty_signal=content_novelty,
        incoming_count_override=int(remote_graph.get("stats", {}).get("incoming_count", 0)),
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
