from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .venue_loader import list_venues, load_venue_profile

_STOPWORDS = {
    "the",
    "and",
    "with",
    "that",
    "this",
    "from",
    "into",
    "for",
    "are",
    "was",
    "were",
    "been",
    "have",
    "has",
    "had",
    "using",
    "used",
    "our",
    "their",
    "your",
    "you",
    "can",
    "not",
    "but",
    "than",
    "also",
    "into",
    "over",
    "under",
    "more",
    "less",
    "each",
    "per",
}


def recommend_venues(
    repo_root: Path,
    *,
    target_year: int,
    paper_structured: dict[str, Any],
    claims_normalized: dict[str, Any],
    evidence_index: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    candidates = list_venues(repo_root)
    corpus = _build_corpus_text(paper_structured, claims_normalized)
    corpus_tokens = _tokenize(corpus)
    section_texts = _section_text_map(evidence_index)

    ranked: list[dict[str, Any]] = []
    for venue_slug in candidates:
        try:
            profile, used_fallback, source = load_venue_profile(repo_root, venue_slug, target_year)
        except Exception:  # noqa: BLE001
            continue

        required_checks = list(profile.required_checks)
        required_specs_raw = profile.required_check_specs
        required_specs = {
            key: value.model_dump() if hasattr(value, "model_dump") else dict(value)
            for key, value in required_specs_raw.items()
        }

        check_eval = _evaluate_checks(required_checks, required_specs, corpus, section_texts)
        readiness = check_eval["readiness_score"]

        venue_tokens = _venue_keyword_tokens(profile, required_specs)
        topic_overlap = _jaccard(corpus_tokens, venue_tokens)

        novelty_weight = float(profile.weights.get("novelty", 0.25))
        system_bias = _system_bias_bonus(required_checks, corpus_tokens)
        final_score = (
            0.45 * readiness
            + 0.35 * topic_overlap
            + 0.10 * novelty_weight
            + 0.10 * system_bias
        )
        final_score = max(0.0, min(1.0, final_score))

        reasons = _build_reasons(
            venue_slug=venue_slug,
            profile_source=source,
            readiness=readiness,
            readiness_strict=check_eval["readiness_strict"],
            readiness_weighted=check_eval["readiness_weighted"],
            topic_overlap=topic_overlap,
            passed_checks=check_eval["passed_checks"],
            failed_checks=check_eval["failed_checks"],
            check_details=check_eval["check_details"],
            used_fallback=used_fallback,
        )
        required_check_mapping = [
            {
                "check_name": row.get("check_name"),
                "description": row.get("description", ""),
                "severity_hint": row.get("severity_hint", "P2"),
                "keywords": row.get("keywords", []),
                "thresholds": row.get("thresholds", {}),
            }
            for row in check_eval["check_details"]
        ]
        ranked.append(
            {
                "venue": venue_slug,
                "year": target_year,
                "match_score": round(final_score, 3),
                "readiness_score": round(readiness, 3),
                "rule_readiness": {
                    "score": round(readiness, 3),
                    "strict_pass_ratio": round(check_eval["readiness_strict"], 3),
                    "weighted_coverage": round(check_eval["readiness_weighted"], 3),
                    "formula": check_eval["readiness_formula"],
                    "total_required_checks": check_eval["total_checks"],
                    "passed_checks_count": len(check_eval["passed_checks"]),
                    "failed_checks_count": len(check_eval["failed_checks"]),
                },
                "topic_overlap_score": round(topic_overlap, 3),
                "system_bias_score": round(system_bias, 3),
                "weights": profile.weights,
                "reasons": reasons,
                "passed_checks": check_eval["passed_checks"][:6],
                "failed_checks": check_eval["failed_checks"][:6],
                "check_diagnostics": check_eval["check_details"][:12],
                "required_check_mapping": required_check_mapping[:12],
                "profile_source": source,
            }
        )

    ranked.sort(key=lambda x: x["match_score"], reverse=True)
    top = ranked[: max(1, top_k)]
    return {
        "method": "rule_based_reverse_matching_from_abstract_contributions_and_venue_rules",
        "target_year": target_year,
        "candidate_count": len(ranked),
        "recommended_venues": top,
    }


def _build_corpus_text(paper_structured: dict[str, Any], claims_normalized: dict[str, Any]) -> str:
    title = str(paper_structured.get("title", "") or "")
    sections = paper_structured.get("sections", [])
    section_blocks: list[str] = []
    if isinstance(sections, list):
        for section in sections[:8]:
            if not isinstance(section, dict):
                continue
            name = str(section.get("name", "")).strip().lower()
            text = str(section.get("text", "")).strip()
            if name in {"abstract", "introduction", "conclusion", "experiments", "method"}:
                section_blocks.append(text)
    claims = claims_normalized.get("claims", []) if isinstance(claims_normalized, dict) else []
    claim_texts: list[str] = []
    if isinstance(claims, list):
        for row in claims[:20]:
            if not isinstance(row, dict):
                continue
            claim_texts.append(str(row.get("claim_text", "")))
            claim_texts.append(str(row.get("verifiable_claim", "")))
    return "\n".join([title] + section_blocks + claim_texts)


def _section_text_map(evidence_index: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    passages = evidence_index.get("passages", []) if isinstance(evidence_index, dict) else []
    if not isinstance(passages, list):
        return out
    for row in passages[:250]:
        if not isinstance(row, dict):
            continue
        section = str(row.get("section", "")).strip().lower() or "unknown"
        text = str(row.get("text", "")).strip().lower()
        if not text:
            continue
        if section not in out:
            out[section] = text
        else:
            out[section] += "\n" + text
    return out


def _evaluate_checks(
    required_checks: list[str],
    required_specs: dict[str, dict],
    corpus: str,
    section_texts: dict[str, str],
) -> dict[str, Any]:
    corpus_lower = corpus.lower()
    passed: list[str] = []
    failed: list[str] = []
    total = len(required_checks) if required_checks else 1
    check_details: list[dict[str, Any]] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for check in required_checks:
        spec = required_specs.get(check, {})
        keywords = spec.get("keywords", [])
        if not isinstance(keywords, list) or not keywords:
            keywords = _default_keywords_for_check(check)
        keywords = [str(k).strip().lower() for k in keywords if str(k).strip()]
        min_hits = int(spec.get("min_hits", 1) or 1)
        min_sections = int(spec.get("min_distinct_sections", 0) or 0)
        severity_hint = str(spec.get("severity_hint", "P2") or "P2").upper()
        description = str(spec.get("description", "")).strip()

        hit_keywords: list[str] = []
        hit_sections: set[str] = set()
        section_hits_by_keyword: dict[str, list[str]] = {}
        for kw in keywords:
            kw_hit = False
            if kw and kw in corpus_lower:
                kw_hit = True
            matched_sections: list[str] = []
            for section, text in section_texts.items():
                if kw and kw in text:
                    hit_sections.add(section)
                    matched_sections.append(section)
                    kw_hit = True
            if kw_hit:
                hit_keywords.append(kw)
            if matched_sections:
                section_hits_by_keyword[kw] = sorted(set(matched_sections))

        hits = len(set(hit_keywords))
        is_passed = hits >= max(1, min_hits) and len(hit_sections) >= max(0, min_sections)
        coverage_score = _check_coverage_score(
            keyword_hits=hits,
            keyword_total=len(keywords),
            min_hits=max(1, min_hits),
            section_hit_count=len(hit_sections),
            min_sections=max(0, min_sections),
        )
        weight = _severity_weight(severity_hint)
        weighted_sum += weight * coverage_score
        weight_total += weight

        missing_keywords = [kw for kw in keywords if kw not in set(hit_keywords)]
        detail = {
            "check_name": check,
            "passed": is_passed,
            "coverage_score": round(coverage_score, 3),
            "severity_hint": severity_hint,
            "description": description or _humanize_check_name(check),
            "keywords": keywords,
            "hit_keywords": sorted(set(hit_keywords)),
            "missing_keywords": missing_keywords[:8],
            "hit_count": hits,
            "section_hit_count": len(hit_sections),
            "section_hits": sorted(hit_sections),
            "section_hits_by_keyword": section_hits_by_keyword,
            "thresholds": {
                "min_hits": max(1, min_hits),
                "min_distinct_sections": max(0, min_sections),
            },
        }
        check_details.append(detail)
        if is_passed:
            passed.append(check)
        else:
            failed.append(check)

    readiness_strict = len(passed) / max(total, 1)
    readiness_weighted = weighted_sum / max(weight_total, 1.0)
    readiness = 0.55 * readiness_strict + 0.45 * readiness_weighted
    return {
        "readiness_score": readiness,
        "readiness_strict": readiness_strict,
        "readiness_weighted": readiness_weighted,
        "readiness_formula": "0.55*strict_pass_ratio + 0.45*weighted_check_coverage",
        "total_checks": len(required_checks),
        "passed_checks": passed,
        "failed_checks": failed,
        "check_details": check_details,
    }


def _default_keywords_for_check(check_name: str) -> list[str]:
    c = str(check_name).strip().lower()
    mapping = {
        "baseline_coverage": ["baseline", "comparison", "state-of-the-art"],
        "statistical_significance": ["significance", "p-value", "confidence interval", "std"],
        "ablation_completeness": ["ablation", "without", "remove component"],
        "reproducibility_details": ["seed", "hyperparameter", "code", "implementation details"],
        "workload_diversity": ["workload", "benchmark", "oltp", "olap", "trace", "tpc"],
        "scalability_evaluation": ["scalability", "scale-out", "cluster size", "data size"],
        "efficiency_tradeoff_reporting": ["throughput", "latency", "runtime", "cost", "memory"],
    }
    return mapping.get(c, [c.replace("_", " ")])


def _venue_keyword_tokens(profile: Any, required_specs: dict[str, dict]) -> set[str]:
    text_chunks: list[str] = []
    text_chunks.extend(list(profile.common_reject_reasons or []))
    text_chunks.extend([str(x) for x in list(profile.required_checks or [])])
    for value in required_specs.values():
        if not isinstance(value, dict):
            continue
        text_chunks.append(str(value.get("description", "")))
        kws = value.get("keywords", [])
        if isinstance(kws, list):
            text_chunks.extend(str(x) for x in kws)
    return _tokenize(" ".join(text_chunks))


def _tokenize(text: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", str(text).lower())
    out = {tok for tok in raw if tok not in _STOPWORDS}
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    inter = a & b
    return len(inter) / len(union)


def _system_bias_bonus(required_checks: list[str], corpus_tokens: set[str]) -> float:
    system_checks = {
        "workload_diversity",
        "scalability_evaluation",
        "efficiency_tradeoff_reporting",
        "system_setting_reproducibility",
    }
    if not required_checks:
        return 0.5
    coverage = len(system_checks & set(required_checks)) / max(1, len(system_checks))
    paper_system_terms = {"throughput", "latency", "workload", "benchmark", "cluster", "scalability", "database"}
    paper_signal = len(paper_system_terms & corpus_tokens) / max(1, len(paper_system_terms))
    return min(1.0, 0.5 * coverage + 0.5 * paper_signal)


def _build_reasons(
    *,
    venue_slug: str,
    profile_source: str,
    readiness: float,
    readiness_strict: float,
    readiness_weighted: float,
    topic_overlap: float,
    passed_checks: list[str],
    failed_checks: list[str],
    check_details: list[dict[str, Any]],
    used_fallback: bool,
) -> list[str]:
    total = len(passed_checks) + len(failed_checks)
    detail_by_name = {
        str(row.get("check_name", "")): row
        for row in check_details
        if isinstance(row, dict)
    }
    passed_detail = [
        detail_by_name[name]
        for name in passed_checks
        if name in detail_by_name
    ]
    failed_detail = [
        detail_by_name[name]
        for name in failed_checks
        if name in detail_by_name
    ]
    passed_detail.sort(key=lambda x: float(x.get("coverage_score", 0.0)), reverse=True)
    failed_detail.sort(key=lambda x: float(x.get("coverage_score", 0.0)))

    reasons = [
        (
            f"Rule readiness for {venue_slug.upper()} is {readiness:.2f} "
            f"(strict={readiness_strict:.2f}, weighted={readiness_weighted:.2f}; "
            f"{len(passed_checks)}/{max(total,1)} checks passed)."
        ),
        f"Topic overlap between paper and {venue_slug.upper()} rule language is {topic_overlap:.2f}.",
    ]
    if passed_detail:
        signals: list[str] = []
        for row in passed_detail[:3]:
            signals.append(
                f"{row.get('check_name')} (coverage={float(row.get('coverage_score', 0.0)):.2f})"
            )
        reasons.append("Strong aligned checks: " + ", ".join(signals))
    if failed_detail:
        gaps: list[str] = []
        for row in failed_detail[:3]:
            missing = row.get("missing_keywords", [])
            missing_text = ""
            if isinstance(missing, list) and missing:
                missing_text = f"; missing keywords: {', '.join(str(x) for x in missing[:2])}"
            gaps.append(
                f"{row.get('check_name')} (hits={row.get('hit_count',0)}/{row.get('thresholds',{}).get('min_hits',1)}, "
                f"sections={row.get('section_hit_count',0)}/{row.get('thresholds',{}).get('min_distinct_sections',0)}"
                f"{missing_text})"
            )
        reasons.append("Main venue-specific gaps: " + "; ".join(gaps))
        first_desc = str(failed_detail[0].get("description", "")).strip()
        if first_desc:
            reasons.append(f"Most critical gap meaning: {first_desc}")
    if used_fallback:
        reasons.append("This recommendation uses fallback-year venue profile (not exact current-year snapshot).")
    if profile_source:
        reasons.append(f"Profile source: {profile_source}.")
    return reasons


def _severity_weight(severity_hint: str) -> float:
    s = str(severity_hint or "").upper()
    if s == "P0":
        return 3.0
    if s == "P1":
        return 2.0
    return 1.0


def _check_coverage_score(
    *,
    keyword_hits: int,
    keyword_total: int,
    min_hits: int,
    section_hit_count: int,
    min_sections: int,
) -> float:
    hit_ratio = min(1.0, float(keyword_hits) / max(1.0, float(min_hits)))
    lexical_coverage = (
        min(1.0, float(keyword_hits) / float(keyword_total))
        if keyword_total > 0
        else 1.0
    )
    section_ratio = (
        min(1.0, float(section_hit_count) / max(1.0, float(min_sections)))
        if min_sections > 0
        else 1.0
    )
    return max(0.0, min(1.0, 0.65 * hit_ratio + 0.20 * lexical_coverage + 0.15 * section_ratio))


def _humanize_check_name(check_name: str) -> str:
    text = str(check_name or "").strip().replace("_", " ")
    if not text:
        return "required check"
    return text[0].upper() + text[1:]
