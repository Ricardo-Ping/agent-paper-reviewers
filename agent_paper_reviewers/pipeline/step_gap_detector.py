from __future__ import annotations

from collections import Counter
from datetime import datetime

from ..models import EvidenceRef, GapItem
from .base import PipelineContext, PipelineStep


class GapDetectorStep(PipelineStep):
    name = "GapDetector"

    def run(self, ctx: PipelineContext) -> None:
        raw_text = ctx.artifacts["paper_structured"].get("raw_text", "").lower()
        passages = ctx.artifacts["evidence_index"].get("passages", [])
        venue_profile = ctx.artifacts["venue_profile"]["profile"]
        required_checks = venue_profile.get("required_checks", [])
        required_check_specs = venue_profile.get("required_check_specs", {})
        alignments = ctx.artifacts["claim_evidence_matrix"]["alignments"]
        citation_graph = ctx.artifacts.get("citation_graph", {})

        gaps: list[dict] = []
        seen_codes: set[str] = set()
        check_outcomes: list[dict] = []

        for check_name in required_checks:
            outcome = self._evaluate_required_check(
                check_name,
                raw_text,
                passages,
                citation_graph=citation_graph,
                check_specs=required_check_specs,
            )
            check_outcomes.append(outcome)
            if outcome["passed"]:
                continue

            gap_code = outcome["gap_code"]
            if gap_code in seen_codes:
                continue
            seen_codes.add(gap_code)

            refs = self._probe_refs(passages, outcome["keywords"])
            gaps.append(
                GapItem(
                    code=gap_code,
                    severity_hint=outcome["severity_hint"],
                    description=outcome["description"],
                    evidence_refs=refs,
                ).model_dump()
            )

        weak_claims = [a for a in alignments if a.get("strength") in {"None", "Weak"}]
        if weak_claims:
            refs = []
            for claim in weak_claims[:2]:
                for ref in claim.get("evidence_refs", [])[:1]:
                    refs.append(EvidenceRef.model_validate(ref))
            gaps.append(
                GapItem(
                    code="weak_claim_alignment",
                    severity_hint="P1",
                    description="One or more key claims have weak evidence alignment and need direct support.",
                    evidence_refs=refs,
                ).model_dump()
            )

        contradiction_claims = [
            a
            for a in alignments
            if float(a.get("contradiction_score", 0.0) or 0.0) >= 0.55
        ]
        if contradiction_claims:
            refs = []
            claim_ids: list[str] = []
            max_score = 0.0
            for claim in contradiction_claims[:3]:
                claim_ids.append(str(claim.get("claim_id", "")))
                max_score = max(max_score, float(claim.get("contradiction_score", 0.0) or 0.0))
                rows = claim.get("contradictory_evidence_refs", [])
                if isinstance(rows, list):
                    for ref in rows[:1]:
                        try:
                            refs.append(EvidenceRef.model_validate(ref))
                        except Exception:  # noqa: BLE001
                            continue
            severity = "P0" if max_score >= 0.72 else "P1"
            gaps.append(
                GapItem(
                    code="claim_evidence_contradiction",
                    severity_hint=severity,
                    description=(
                        "Detected potential claim-result contradiction: one or more evidence anchors "
                        f"appear to conflict with claim direction ({', '.join([c for c in claim_ids if c])})."
                    ),
                    evidence_refs=refs[:2],
                ).model_dump()
            )

        citation_gaps = self._collect_citation_gaps(citation_graph, passages)
        gaps.extend(citation_gaps)

        # De-duplicate by description and keep deterministic ordering.
        dedup: list[dict] = []
        seen_desc = set()
        for gap in gaps:
            key = (gap["code"], gap["description"])
            if key in seen_desc:
                continue
            seen_desc.add(key)
            dedup.append(gap)

        dedup.sort(key=lambda x: (self._severity_rank(x["severity_hint"]), x["code"]))

        payload = {
            "gaps": dedup,
            "required_checks": required_checks,
            "required_check_specs": required_check_specs if isinstance(required_check_specs, dict) else {},
            "required_check_outcomes": check_outcomes,
            "alignment_strength_distribution": dict(
                Counter(a.get("strength", "None") for a in alignments)
            ),
            "citation_graph_summary": citation_graph.get("stats", {}),
        }
        ctx.artifacts["gaps"] = payload
        ctx.dump_json("artifacts/gaps.json", payload)

    def _evaluate_required_check(
        self,
        check_name: str,
        raw_text: str,
        passages: list[dict],
        *,
        citation_graph: dict | None = None,
        check_specs: dict | None = None,
    ) -> dict:
        normalized = str(check_name).strip().lower()

        default_specs = {
            "baseline_coverage": {
                "gap_code": "missing_baseline",
                "keywords": ["baseline", "sota", "comparison", "compared with", "state-of-the-art"],
                "description": "Strong baseline or SOTA comparison appears insufficient.",
                "severity_hint": "P1",
                "min_hits": 2,
            },
            "statistical_significance": {
                "gap_code": "missing_significance",
                "keywords": ["p-value", "confidence interval", "std", "variance", "significance", "t-test"],
                "description": "Statistical significance evidence appears missing.",
                "severity_hint": "P1",
                "min_hits": 2,
            },
            "significance_reporting": {
                "gap_code": "missing_significance",
                "keywords": ["p-value", "confidence interval", "std", "significance"],
                "description": "Significance reporting is not sufficiently explicit.",
                "severity_hint": "P1",
                "min_hits": 2,
            },
            "ablation_completeness": {
                "gap_code": "missing_ablation",
                "keywords": ["ablation", "w/o", "without", "remove component"],
                "description": "Ablation study does not look comprehensive.",
                "severity_hint": "P1",
                "min_hits": 1,
            },
            "reproducibility_details": {
                "gap_code": "missing_reproducibility",
                "keywords": ["seed", "hyperparameter", "implementation details", "code", "github"],
                "description": "Reproducibility details are likely incomplete.",
                "severity_hint": "P1",
                "min_hits": 2,
            },
            "error_analysis": {
                "gap_code": "missing_error_analysis",
                "keywords": ["error analysis", "failure case", "qualitative", "where it fails"],
                "description": "Error analysis / failure cases are under-developed.",
                "severity_hint": "P2",
                "min_hits": 1,
            },
            "qualitative_error_analysis": {
                "gap_code": "missing_error_analysis",
                "keywords": ["qualitative", "visualization", "failure case", "error analysis"],
                "description": "Qualitative error analysis appears insufficient.",
                "severity_hint": "P2",
                "min_hits": 1,
            },
            "failure_case_analysis": {
                "gap_code": "missing_error_analysis",
                "keywords": ["failure case", "fails", "error analysis"],
                "description": "Failure-case analysis appears insufficient.",
                "severity_hint": "P2",
                "min_hits": 1,
            },
            "limitation_discussion": {
                "gap_code": "missing_limitations",
                "keywords": ["limitation", "scope", "future work", "risk"],
                "description": "Limitation discussion is under-specified.",
                "severity_hint": "P2",
                "min_hits": 1,
            },
            "limitations": {
                "gap_code": "missing_limitations",
                "keywords": ["limitation", "future work", "constraint"],
                "description": "Limitations section appears incomplete.",
                "severity_hint": "P2",
                "min_hits": 1,
            },
            "qualitative_analysis": {
                "gap_code": "missing_qualitative_analysis",
                "keywords": ["qualitative", "case study", "human evaluation", "examples"],
                "description": "Qualitative analysis appears underdeveloped for this venue.",
                "severity_hint": "P2",
                "min_hits": 1,
            },
            "robustness_checks": {
                "gap_code": "missing_robustness",
                "keywords": ["robust", "ood", "noise", "perturbation", "stress test"],
                "description": "Robustness checks are likely insufficient.",
                "severity_hint": "P1",
                "min_hits": 1,
            },
            "practical_impact": {
                "gap_code": "missing_practical_impact",
                "keywords": ["real-world", "practical", "deployment", "latency", "cost"],
                "description": "Practical impact evidence appears limited.",
                "severity_hint": "P2",
                "min_hits": 1,
            },
            "baseline_config_fairness": {
                "gap_code": "missing_baseline_fairness",
                "keywords": [
                    "fair comparison",
                    "matched settings",
                    "same hardware",
                    "same budget",
                    "same training budget",
                    "tuned baseline",
                ],
                "description": "Baseline fairness and matched experimental settings are not explicit enough.",
                "severity_hint": "P1",
                "min_hits": 1,
            },
            "workload_diversity": {
                "gap_code": "missing_workload_diversity",
                "keywords": [
                    "workload",
                    "benchmark",
                    "tpc",
                    "ycsb",
                    "query set",
                    "oltp",
                    "olap",
                    "trace",
                ],
                "description": "Workload diversity looks insufficient for database/system-style evaluation.",
                "severity_hint": "P1",
                "min_hits": 2,
            },
            "scalability_evaluation": {
                "gap_code": "missing_scalability_evaluation",
                "keywords": [
                    "scalability",
                    "scale-out",
                    "scale up",
                    "data size",
                    "cluster size",
                    "number of nodes",
                    "strong scaling",
                    "weak scaling",
                ],
                "description": "Scalability evaluation appears weak or missing.",
                "severity_hint": "P1",
                "min_hits": 1,
            },
            "efficiency_tradeoff_reporting": {
                "gap_code": "missing_efficiency_tradeoff",
                "keywords": [
                    "latency",
                    "throughput",
                    "qps",
                    "runtime",
                    "memory",
                    "cpu",
                    "overhead",
                    "cost",
                ],
                "description": "Efficiency and cost/performance trade-off reporting appears insufficient.",
                "severity_hint": "P1",
                "min_hits": 2,
            },
            "system_setting_reproducibility": {
                "gap_code": "missing_system_setting_reproducibility",
                "keywords": [
                    "hardware",
                    "gpu",
                    "cpu",
                    "ram",
                    "db version",
                    "database version",
                    "configuration",
                    "parameter setting",
                    "environment",
                ],
                "description": "System settings and environment details are not reproducible enough.",
                "severity_hint": "P1",
                "min_hits": 2,
            },
            "ethics_limitations": {
                "gap_code": "missing_ethics_limitations",
                "keywords": ["ethics", "societal impact", "bias", "safety"],
                "description": "Ethics or societal limitations discussion appears missing.",
                "severity_hint": "P2",
                "min_hits": 1,
            },
            "contribution_alignment": {
                "gap_code": "missing_contribution_alignment",
                "keywords": ["contribution", "we propose", "novel", "positioning"],
                "description": "Contribution positioning and novelty alignment may be unclear.",
                "severity_hint": "P1",
                "min_hits": 2,
            },
            "clarity": {
                "gap_code": "missing_clarity_support",
                "keywords": ["notation", "definition", "algorithm", "pseudo-code"],
                "description": "Presentation clarity signals are weak for the target venue.",
                "severity_hint": "P2",
                "min_hits": 1,
            },
            "top_venue_related_work_coverage": {
                "gap_code": "missing_top_venue_related_work_coverage",
                "keywords": ["related work", "references", "state-of-the-art"],
                "description": "Related work appears to under-cover recent top-venue papers relevant to this topic.",
                "severity_hint": "P1",
                "min_hits": 1,
                "min_citation_top_venue": 4,
                "min_citation_top_venue_recent": 2,
            },
        }

        spec = default_specs.get(normalized, {}).copy()
        if isinstance(check_specs, dict):
            venue_spec = check_specs.get(normalized)
            if isinstance(venue_spec, dict):
                spec.update(venue_spec)

        if not spec:
            spec = {
                "check_name": normalized,
                "gap_code": f"missing_{normalized}",
                "keywords": [normalized.replace("_", " ")],
                "description": f"Evidence for required check '{check_name}' appears insufficient.",
                "severity_hint": "P2",
                "min_hits": 1,
            }

        keywords = spec.get("keywords", [])
        if not isinstance(keywords, list) or not keywords:
            keywords = [normalized.replace("_", " ")]
        keywords = [str(x).strip().lower() for x in keywords if str(x).strip()]

        hits, distinct_sections = self._count_hits(raw_text, passages, keywords)
        min_hits = int(spec.get("min_hits", 1) or 1)
        min_distinct_sections = int(spec.get("min_distinct_sections", 0) or 0)

        stats = citation_graph.get("stats", {}) if isinstance(citation_graph, dict) else {}
        outgoing_count = int(stats.get("outgoing_count", 0) or 0)
        baseline_like_count = int(stats.get("baseline_like_reference_count", 0) or 0)
        top_venue_count = int(stats.get("top_venue_reference_count", 0) or 0)
        recent_top_venue_count = int(stats.get("recent_top_venue_reference_count", 0) or 0)

        min_citation_outgoing = int(spec.get("min_citation_outgoing", 0) or 0)
        min_citation_baseline_like = int(spec.get("min_citation_baseline_like", 0) or 0)
        min_citation_top_venue = int(spec.get("min_citation_top_venue", 0) or 0)
        min_citation_top_venue_recent = int(spec.get("min_citation_top_venue_recent", 0) or 0)

        passed = hits >= min_hits
        passed = passed and distinct_sections >= min_distinct_sections
        passed = passed and outgoing_count >= min_citation_outgoing
        passed = passed and baseline_like_count >= min_citation_baseline_like
        passed = passed and top_venue_count >= min_citation_top_venue
        passed = passed and recent_top_venue_count >= min_citation_top_venue_recent

        return {
            "passed": passed,
            "check_name": normalized,
            "gap_code": str(spec.get("gap_code") or f"missing_{normalized}"),
            "keywords": keywords,
            "description": str(
                spec.get("description")
                or f"Evidence for required check '{check_name}' appears insufficient."
            ),
            "severity_hint": str(spec.get("severity_hint") or "P2"),
            "hits": hits,
            "distinct_sections": distinct_sections,
            "thresholds": {
                "min_hits": min_hits,
                "min_distinct_sections": min_distinct_sections,
                "min_citation_outgoing": min_citation_outgoing,
                "min_citation_baseline_like": min_citation_baseline_like,
                "min_citation_top_venue": min_citation_top_venue,
                "min_citation_top_venue_recent": min_citation_top_venue_recent,
            },
            "citation_stats": {
                "outgoing_count": outgoing_count,
                "baseline_like_reference_count": baseline_like_count,
                "top_venue_reference_count": top_venue_count,
                "recent_top_venue_reference_count": recent_top_venue_count,
            },
        }

    @staticmethod
    def _count_hits(raw_text: str, passages: list[dict], keywords: list[str]) -> tuple[int, int]:
        hits = 0
        sections: set[str] = set()
        for keyword in keywords:
            if keyword in raw_text:
                hits += 1

        # Add coverage from top passages to reduce false negatives when raw_text is noisy.
        for passage in passages[:120]:
            text = str(passage.get("text") or "").lower()
            if any(k in text for k in keywords):
                hits += 1
                section = str(passage.get("section") or "").strip().lower()
                if section:
                    sections.add(section)
        return hits, len(sections)

    @staticmethod
    def _probe_refs(passages: list[dict], keywords: list[str]) -> list[EvidenceRef]:
        refs: list[EvidenceRef] = []
        for passage in passages:
            text = str(passage.get("text") or "")
            if any(k in text.lower() for k in keywords):
                refs.append(
                    EvidenceRef(
                        section=str(passage.get("section") or "unknown"),
                        passage_id=str(passage.get("id") or "unknown"),
                        excerpt=text[:180],
                    )
                )
            if len(refs) >= 2:
                break

        # If nothing matched, include first paragraph as anchor so report has traceability.
        if not refs and passages:
            top = passages[0]
            refs.append(
                EvidenceRef(
                    section=str(top.get("section") or "unknown"),
                    passage_id=str(top.get("id") or "unknown"),
                    excerpt=str(top.get("text") or "")[:180],
                )
            )
        return refs

    @staticmethod
    def _severity_rank(severity: str) -> int:
        order = {"P0": 0, "P1": 1, "P2": 2}
        return order.get(str(severity).upper(), 3)

    def _collect_citation_gaps(self, citation_graph: dict, passages: list[dict]) -> list[dict]:
        if not isinstance(citation_graph, dict):
            return []

        stats = citation_graph.get("stats", {}) if isinstance(citation_graph.get("stats"), dict) else {}
        outgoing_count = int(stats.get("outgoing_count", 0) or 0)
        incoming_count = int(stats.get("incoming_count", 0) or 0)
        baseline_like_count = int(stats.get("baseline_like_reference_count", 0) or 0)
        source = str(citation_graph.get("source", "none"))
        top_venue_count = int(stats.get("top_venue_reference_count", 0) or 0)
        recent_top_venue_count = int(stats.get("recent_top_venue_reference_count", 0) or 0)
        novelty_signal_score = float(stats.get("novelty_signal_score", 0.0) or 0.0)
        content_novelty_score = float(stats.get("content_novelty_score", 0.0) or 0.0)

        gaps: list[dict] = []

        if outgoing_count < 10:
            refs = self._citation_refs(citation_graph, relation="outgoing")
            if not refs:
                refs = self._probe_refs(passages, ["related work", "references", "baseline", "sota"])
            gaps.append(
                GapItem(
                    code="missing_reference_coverage",
                    severity_hint="P1",
                    description=(
                        "Citation coverage appears shallow; references may be insufficient to position novelty and baselines."
                    ),
                    evidence_refs=refs,
                ).model_dump()
            )

        if outgoing_count >= 8 and (top_venue_count < 3 or recent_top_venue_count < 2):
            refs = self._citation_refs(citation_graph, relation="outgoing")
            if not refs:
                refs = self._probe_refs(passages, ["related work", "references", "state-of-the-art", "benchmark"])
            gaps.append(
                GapItem(
                    code="missing_top_venue_recent_coverage",
                    severity_hint="P2",
                    description=(
                        "Reference list seems to under-cover recent top-venue work; novelty positioning may be incomplete."
                    ),
                    evidence_refs=refs,
                ).model_dump()
            )

        if outgoing_count >= 6 and baseline_like_count < 2:
            refs = self._citation_refs(citation_graph, relation="outgoing")
            if not refs:
                refs = self._probe_refs(passages, ["baseline", "compare", "state-of-the-art"])
            gaps.append(
                GapItem(
                    code="missing_baseline_citation_coverage",
                    severity_hint="P1",
                    description="Cited work appears to under-cover strong baseline or benchmark papers.",
                    evidence_refs=refs,
                ).model_dump()
            )

        paper_year = citation_graph.get("paper", {}).get("year")
        try:
            paper_year = int(paper_year)
        except (TypeError, ValueError):
            paper_year = None

        current_year = datetime.now().year
        if source in {"semantic_scholar", "hybrid", "semantic_scholar_search_only"}:
            if paper_year is not None and current_year - paper_year >= 2 and incoming_count == 0:
                if novelty_signal_score < 0.42 and content_novelty_score < 0.55:
                    refs = self._citation_refs(citation_graph, relation="incoming")
                    if not refs:
                        refs = self._probe_refs(passages, ["contribution", "novel", "state of the art"])
                    gaps.append(
                        GapItem(
                            code="weak_novelty_signal_from_citations",
                            severity_hint="P2",
                            description=(
                                "Incoming citation signal is weak for a non-recent paper, and paper-content novelty signals are also limited; novelty impact may need stronger justification."
                            ),
                            evidence_refs=refs,
                        ).model_dump()
                    )

        return gaps

    @staticmethod
    def _citation_refs(citation_graph: dict, relation: str, limit: int = 2) -> list[EvidenceRef]:
        key = "outgoing_references" if relation == "outgoing" else "incoming_citations"
        items = citation_graph.get(key, [])
        if not isinstance(items, list):
            return []
        refs: list[EvidenceRef] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            refs.append(
                EvidenceRef(
                    section="citation_graph",
                    passage_id=f"{relation}:{idx}",
                    excerpt=title[:180],
                )
            )
            if len(refs) >= limit:
                break
        return refs
