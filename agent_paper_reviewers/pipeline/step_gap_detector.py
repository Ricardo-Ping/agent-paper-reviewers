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
        required_checks = ctx.artifacts["venue_profile"]["profile"].get("required_checks", [])
        alignments = ctx.artifacts["claim_evidence_matrix"]["alignments"]
        citation_graph = ctx.artifacts.get("citation_graph", {})

        gaps: list[dict] = []
        seen_codes: set[str] = set()

        for check_name in required_checks:
            outcome = self._evaluate_required_check(check_name, raw_text, passages)
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
            "alignment_strength_distribution": dict(
                Counter(a.get("strength", "None") for a in alignments)
            ),
            "citation_graph_summary": citation_graph.get("stats", {}),
        }
        ctx.artifacts["gaps"] = payload
        ctx.dump_json("artifacts/gaps.json", payload)

    def _evaluate_required_check(self, check_name: str, raw_text: str, passages: list[dict]) -> dict:
        normalized = str(check_name).strip().lower()

        check_specs = {
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
        }

        spec = check_specs.get(normalized)
        if spec is None:
            spec = {
                "gap_code": f"missing_{normalized}",
                "keywords": [normalized.replace("_", " ")],
                "description": f"Evidence for required check '{check_name}' appears insufficient.",
                "severity_hint": "P2",
                "min_hits": 1,
            }

        hits = self._count_hits(raw_text, passages, spec["keywords"])
        passed = hits >= int(spec["min_hits"])
        return {
            "passed": passed,
            "gap_code": spec["gap_code"],
            "keywords": spec["keywords"],
            "description": spec["description"],
            "severity_hint": spec["severity_hint"],
            "hits": hits,
        }

    @staticmethod
    def _count_hits(raw_text: str, passages: list[dict], keywords: list[str]) -> int:
        hits = 0
        for keyword in keywords:
            if keyword in raw_text:
                hits += 1

        # Add coverage from top passages to reduce false negatives when raw_text is noisy.
        for passage in passages[:120]:
            text = str(passage.get("text") or "").lower()
            if any(k in text for k in keywords):
                hits += 1
        return hits

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
                refs = self._citation_refs(citation_graph, relation="incoming")
                if not refs:
                    refs = self._probe_refs(passages, ["contribution", "novel", "state of the art"])
                gaps.append(
                    GapItem(
                        code="weak_novelty_signal_from_citations",
                        severity_hint="P2",
                        description=(
                            "Incoming citation signal is weak for a non-recent paper; novelty impact may need stronger justification."
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
