from __future__ import annotations

from collections import Counter
from datetime import datetime
import re
from typing import Any

from ..executors.base import ExecutorAdapter
from ..models import EvidenceRef, GapItem, TaskSpec
from .base import PipelineContext, PipelineStep


class GapDetectorStep(PipelineStep):
    name = "GapDetector"

    def __init__(self, executor: ExecutorAdapter | None = None) -> None:
        self.executor = executor

    def run(self, ctx: PipelineContext) -> None:
        raw_text = ctx.artifacts["paper_structured"].get("raw_text", "").lower()
        paper_structured = ctx.artifacts.get("paper_structured", {})
        passages = ctx.artifacts["evidence_index"].get("passages", [])
        venue_profile = ctx.artifacts["venue_profile"]["profile"]
        required_checks = venue_profile.get("required_checks", [])
        required_check_specs = venue_profile.get("required_check_specs", {})
        alignments = ctx.artifacts["claim_evidence_matrix"]["alignments"]
        citation_graph = ctx.artifacts.get("citation_graph", {})

        rule_gaps: list[dict] = []
        seen_codes: set[str] = set()
        check_outcomes: list[dict] = []

        for check_name in required_checks:
            outcome = self._evaluate_required_check(
                check_name,
                raw_text,
                passages,
                paper_structured=paper_structured,
                citation_graph=citation_graph,
                check_specs=required_check_specs,
                ctx=ctx,
            )
            check_outcomes.append(outcome)
            if outcome["passed"]:
                continue

            gap_code = outcome["gap_code"]
            if gap_code in seen_codes:
                continue
            seen_codes.add(gap_code)

            refs = outcome.get("evidence_refs")
            if not isinstance(refs, list) or not refs:
                refs = self._probe_refs(passages, outcome["keywords"])
            rule_gaps.append(
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
            rule_gaps.append(
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
            rule_gaps.append(
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
        rule_gaps.extend(citation_gaps)

        semantic_gaps = self._detect_semantic_gaps_with_executor(
            ctx=ctx,
            paper_structured=paper_structured if isinstance(paper_structured, dict) else {},
            required_checks=required_checks if isinstance(required_checks, list) else [],
            required_check_specs=required_check_specs if isinstance(required_check_specs, dict) else {},
            alignments=alignments if isinstance(alignments, list) else [],
            rule_gaps=rule_gaps,
            passages=passages if isinstance(passages, list) else [],
        )

        merged_candidates, merge_meta = self._merge_semantic_and_rule_gaps(
            semantic_gaps=semantic_gaps if isinstance(semantic_gaps, list) else [],
            rule_gaps=rule_gaps if isinstance(rule_gaps, list) else [],
        )

        # De-duplicate by description and keep deterministic ordering.
        dedup: list[dict] = []
        seen_desc = set()
        for gap in merged_candidates:
            key = (gap["code"], gap["description"])
            if key in seen_desc:
                continue
            seen_desc.add(key)
            dedup.append(gap)

        dedup.sort(key=lambda x: (self._severity_rank(x["severity_hint"]), x["code"]))

        payload = {
            "gaps": dedup,
            "source": merge_meta.get("source", "rule_fallback"),
            "semantic_primary_used": bool(merge_meta.get("semantic_primary_used", False)),
            "rule_fallback_used": bool(merge_meta.get("rule_fallback_used", True)),
            "semantic_gaps_count": int(merge_meta.get("semantic_gaps_count", 0)),
            "rule_gaps_count": int(merge_meta.get("rule_gaps_count", 0)),
            "guardrail_rule_gaps_count": int(merge_meta.get("guardrail_rule_gaps_count", 0)),
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

    @staticmethod
    def _merge_semantic_and_rule_gaps(
        *,
        semantic_gaps: list[dict],
        rule_gaps: list[dict],
    ) -> tuple[list[dict], dict]:
        semantic_rows = [x for x in semantic_gaps if isinstance(x, dict)]
        rule_rows = [x for x in rule_gaps if isinstance(x, dict)]
        if not semantic_rows:
            return rule_rows, {
                "source": "rule_fallback",
                "semantic_primary_used": False,
                "rule_fallback_used": True,
                "semantic_gaps_count": 0,
                "rule_gaps_count": len(rule_rows),
                "guardrail_rule_gaps_count": len(rule_rows),
            }

        out: list[dict] = list(semantic_rows)
        used_codes = {
            str(x.get("code", "")).strip().lower()
            for x in semantic_rows
            if str(x.get("code", "")).strip()
        }
        guardrail_codes = {"claim_evidence_contradiction", "weak_claim_alignment"}
        guardrail_added = 0

        # Keep rule guardrails only when they catch high-severity contradictions
        # or preserve minimal deterministic safety.
        for row in rule_rows:
            code = str(row.get("code", "")).strip().lower()
            if not code or code in used_codes:
                continue
            sev = str(row.get("severity_hint", "P2")).strip().upper()
            if code in guardrail_codes or sev == "P0":
                patched = dict(row)
                patched["source"] = patched.get("source", "rule_guardrail")
                out.append(patched)
                used_codes.add(code)
                guardrail_added += 1

        # If semantic output is too sparse, backfill strongest rule gaps.
        if len(semantic_rows) < 3:
            severity_rank = {"P0": 0, "P1": 1, "P2": 2}
            ranked = sorted(
                [
                    x for x in rule_rows
                    if str(x.get("code", "")).strip().lower() not in used_codes
                ],
                key=lambda r: (
                    severity_rank.get(str(r.get("severity_hint", "P2")).strip().upper(), 9),
                    str(r.get("code", "")),
                ),
            )
            for row in ranked[: 3 - len(semantic_rows)]:
                patched = dict(row)
                patched["source"] = patched.get("source", "rule_backfill")
                out.append(patched)
                guardrail_added += 1

        return out, {
            "source": "executor_primary+rule_guardrails",
            "semantic_primary_used": True,
            "rule_fallback_used": False,
            "semantic_gaps_count": len(semantic_rows),
            "rule_gaps_count": len(rule_rows),
            "guardrail_rule_gaps_count": guardrail_added,
        }

    def _detect_semantic_gaps_with_executor(
        self,
        *,
        ctx: PipelineContext,
        paper_structured: dict,
        required_checks: list[str],
        required_check_specs: dict,
        alignments: list[dict],
        rule_gaps: list[dict],
        passages: list[dict],
    ) -> list[dict]:
        if self.executor is None:
            return []

        sections = paper_structured.get("sections", []) if isinstance(paper_structured, dict) else []
        section_briefs: list[dict] = []
        if isinstance(sections, list):
            for sec in sections[:12]:
                if not isinstance(sec, dict):
                    continue
                section_briefs.append(
                    {
                        "section_id": str(sec.get("section_id", "")),
                        "name": str(sec.get("name", "")),
                        "text": str(sec.get("text", ""))[:600],
                    }
                )

        passage_locator: dict[str, dict] = {}
        for row in passages[:260]:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("id", "")).strip()
            if not pid:
                continue
            passage_locator[pid] = {
                "section": str(row.get("section", "")).strip(),
                "text": str(row.get("text", ""))[:320],
            }

        weak_claims: list[dict] = []
        for row in alignments[:12]:
            if not isinstance(row, dict):
                continue
            if str(row.get("strength", "")).strip().lower() not in {"weak", "none"}:
                continue
            weak_claims.append(
                {
                    "claim_id": str(row.get("claim_id", "")),
                    "claim_text": str(row.get("claim_text", ""))[:260],
                    "score": float(row.get("score", 0.0) or 0.0),
                }
            )

        spec_task = TaskSpec(
            task_type="gap_detection_agent",
            prompt=(
                "Identify paper-specific reject-critical gaps from a strict reviewer perspective. "
                "Do not repeat generic advice; tie every gap to this paper context and anchors."
            ),
            context={
                "venue": {"name": ctx.input_data.venue.name, "year": ctx.input_data.venue.year},
                "required_checks": required_checks,
                "required_check_specs": required_check_specs,
                "section_briefs": section_briefs,
                "weak_claims": weak_claims,
                "rule_gaps": rule_gaps[:8],
                "passage_locator": passage_locator,
            },
            output_schema={
                "gaps": [
                    {
                        "code": "string",
                        "severity_hint": "P0|P1|P2",
                        "specific_description": "string",
                        "evidence_passage_ids": ["S001_para0"],
                        "fix_action": "string",
                    }
                ],
                "summary": "string",
            },
            model_profile="judge",
        )

        result = self.executor.execute(spec_task)
        for warning in result.warnings:
            if "api_key_missing_use_fallback" in warning:
                continue
            ctx.add_qa_issue(f"gap_detector_agent_warning:{warning}")
        if not result.ok:
            return []

        payload: Any = result.output
        if isinstance(payload, dict) and isinstance(payload.get("response"), dict):
            payload = payload.get("response")
        if not isinstance(payload, dict):
            return []

        rows = payload.get("gaps", [])
        if not isinstance(rows, list) or not rows:
            return []

        by_pid = {
            str(row.get("id", "")).strip(): row
            for row in passages
            if isinstance(row, dict) and str(row.get("id", "")).strip()
        }
        out: list[dict] = []
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            code = str(row.get("code", "")).strip().lower()
            if not code:
                continue
            severity_hint = str(row.get("severity_hint", "P1")).strip().upper()
            if severity_hint not in {"P0", "P1", "P2"}:
                severity_hint = "P1"
            desc = str(row.get("specific_description", row.get("description", ""))).strip()
            if not desc:
                continue
            evidence_refs: list[EvidenceRef] = []
            pids = row.get("evidence_passage_ids", [])
            if isinstance(pids, list):
                for pid in pids[:3]:
                    p = by_pid.get(str(pid).strip())
                    if not isinstance(p, dict):
                        continue
                    evidence_refs.append(self._evidence_ref_from_passage(p, excerpt=str(p.get("text", ""))[:180]))
            if not evidence_refs:
                evidence_refs = self._probe_refs(passages, [code.replace("_", " ")])
            item = GapItem(
                code=code,
                severity_hint=severity_hint,
                description=desc,
                evidence_refs=evidence_refs,
            ).model_dump()
            fix_action = str(row.get("fix_action", "")).strip()
            if fix_action:
                item["fix_action"] = fix_action
            item["source"] = "agent_semantic"
            out.append(item)
        return out

    def _evaluate_required_check(
        self,
        check_name: str,
        raw_text: str,
        passages: list[dict],
        *,
        paper_structured: dict | None = None,
        citation_graph: dict | None = None,
        check_specs: dict | None = None,
        ctx: PipelineContext | None = None,
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
            "section_length_ratio": {
                "gap_code": "section_ratio_imbalance",
                "keywords": ["introduction", "method", "experiments", "discussion"],
                "description": (
                    "Section length balance appears misaligned with common venue writing structure "
                    "(introduction/method/experiments/discussion)."
                ),
                "severity_hint": "P2",
                "section_ratio_targets": {
                    "introduction": 0.18,
                    "method": 0.32,
                    "experiments": 0.34,
                    "discussion": 0.16,
                },
                "section_ratio_tolerance": 0.1,
                "section_ratio_min_total_words": 900,
                "section_ratio_min_bucket_words": 80,
                "section_aliases": {
                    "introduction": ["introduction", "background", "motivation", "overview"],
                    "method": ["method", "approach", "model", "architecture", "methodology", "framework", "system"],
                    "experiments": ["experiments", "evaluation", "results", "benchmark", "empirical"],
                    "discussion": ["discussion", "analysis", "conclusion", "limitations", "error analysis", "future work"],
                },
            },
            "terminology_consistency": {
                "gap_code": "terminology_inconsistency",
                "keywords": ["terminology", "notation", "acronym", "consistency"],
                "description": (
                    "Technical terminology appears inconsistent across sections "
                    "(term variants, acronym expansion drift, or unstable naming)."
                ),
                "severity_hint": "P2",
                "terminology_min_mentions": 2,
                "terminology_min_variant_hits": 1,
                "terminology_exempt_terms": ["sota", "llm", "gpu", "cpu"],
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

        if normalized in {"statistical_significance", "significance_reporting"}:
            return self._evaluate_statistical_significance_check(
                check_name=normalized,
                spec=spec,
                raw_text=raw_text,
                passages=passages,
                ctx=ctx,
            )

        if normalized == "section_length_ratio":
            return self._evaluate_section_length_ratio_check(
                spec=spec,
                paper_structured=paper_structured,
                passages=passages,
            )
        if normalized == "terminology_consistency":
            return self._evaluate_terminology_consistency_check(
                spec=spec,
                paper_structured=paper_structured,
                passages=passages,
            )

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

    def _evaluate_statistical_significance_check(
        self,
        *,
        check_name: str,
        spec: dict,
        raw_text: str,
        passages: list[dict],
        ctx: PipelineContext | None = None,
    ) -> dict:
        keywords = spec.get("keywords", [])
        if not isinstance(keywords, list) or not keywords:
            keywords = ["p-value", "confidence interval", "std", "variance", "significance", "t-test"]
        keywords = [str(x).strip().lower() for x in keywords if str(x).strip()]

        regex_payload = self._detect_statistical_signals_regex(raw_text=raw_text, passages=passages)
        llm_payload = self._detect_statistical_signals_with_executor(
            check_name=check_name,
            spec=spec,
            passages=passages,
            ctx=ctx,
        )

        core_signals = ["mean_std", "p_value", "confidence_interval"]
        aux_signals = ["seed_reporting", "test_name"]
        all_signals = core_signals + aux_signals

        final_signals: dict[str, bool] = {
            signal: bool(regex_payload["signals"].get(signal, False))
            for signal in all_signals
        }
        if isinstance(llm_payload, dict):
            llm_signals = llm_payload.get("signals", {})
            if isinstance(llm_signals, dict):
                for signal in all_signals:
                    final_signals[signal] = final_signals.get(signal, False) or self._to_bool(
                        llm_signals.get(signal)
                    )

        default_min_required = 2 if check_name == "statistical_significance" else 1
        min_required_signals = int(spec.get("min_required_signals", default_min_required) or default_min_required)
        min_required_signals = max(1, min(3, min_required_signals))
        require_seed_reporting = bool(spec.get("require_seed_reporting", False))
        require_test_name_or_pvalue = bool(spec.get("require_test_name_or_pvalue", False))

        required_signals_raw = spec.get("required_signals", [])
        required_signals: list[str] = []
        if isinstance(required_signals_raw, list):
            for value in required_signals_raw:
                key = str(value).strip().lower()
                if key in all_signals and key not in required_signals:
                    required_signals.append(key)

        core_hits = sum(1 for signal in core_signals if final_signals.get(signal, False))
        aux_hits = sum(1 for signal in aux_signals if final_signals.get(signal, False))
        passed = core_hits >= min_required_signals
        if required_signals:
            passed = passed and all(final_signals.get(signal, False) for signal in required_signals)
        if require_seed_reporting:
            passed = passed and final_signals.get("seed_reporting", False)
        if require_test_name_or_pvalue:
            passed = passed and (
                final_signals.get("test_name", False) or final_signals.get("p_value", False)
            )

        signal_refs: dict[str, list[EvidenceRef]] = regex_payload.get("signal_refs", {})
        merged_refs: list[EvidenceRef] = []
        for signal in core_signals:
            rows = signal_refs.get(signal, []) if isinstance(signal_refs, dict) else []
            if isinstance(rows, list) and rows:
                merged_refs.extend(rows[:1])
        if isinstance(llm_payload, dict):
            llm_refs = llm_payload.get("evidence_refs", [])
            if isinstance(llm_refs, list):
                merged_refs.extend([x for x in llm_refs if isinstance(x, EvidenceRef)])
        merged_refs = self._dedupe_evidence_refs(merged_refs, limit=3)
        if not merged_refs and passages:
            merged_refs = self._probe_refs(passages, keywords)
        evidence_ref_payload = [ref.model_dump() for ref in merged_refs]

        missing_core = [self._signal_display_name(signal) for signal in core_signals if not final_signals.get(signal, False)]
        if passed:
            present = [self._signal_display_name(signal) for signal in core_signals if final_signals.get(signal, False)]
            description = (
                "Statistical reporting signals are detected: " + ", ".join(present) + "."
            )
        else:
            pieces: list[str] = []
            if missing_core:
                pieces.append("missing " + ", ".join(missing_core))
            if require_seed_reporting and not final_signals.get("seed_reporting", False):
                pieces.append("missing multi-seed reporting")
            if require_test_name_or_pvalue and not (
                final_signals.get("test_name", False) or final_signals.get("p_value", False)
            ):
                pieces.append("missing explicit test name or p-value")
            suffix = "; ".join(pieces) if pieces else "statistical evidence is still weak."
            description = f"Statistical significance evidence appears incomplete: {suffix}."

        distinct_sections = len(
            {
                str(ref.section).strip().lower()
                for ref in merged_refs
                if str(ref.section).strip()
            }
        )

        return {
            "passed": passed,
            "check_name": check_name,
            "gap_code": str(spec.get("gap_code") or "missing_significance"),
            "keywords": keywords,
            "description": description,
            "severity_hint": str(spec.get("severity_hint") or "P1"),
            "hits": core_hits + aux_hits,
            "distinct_sections": distinct_sections,
            "thresholds": {
                "min_required_signals": min_required_signals,
                "required_signals": required_signals,
                "require_seed_reporting": require_seed_reporting,
                "require_test_name_or_pvalue": require_test_name_or_pvalue,
            },
            "statistical_detection": {
                "signals": final_signals,
                "missing_core_signals": [
                    signal for signal in core_signals if not final_signals.get(signal, False)
                ],
                "regex": {
                    "signals": regex_payload.get("signals", {}),
                    "raw_match_counts": regex_payload.get("raw_match_counts", {}),
                    "passage_match_counts": regex_payload.get("passage_match_counts", {}),
                    "passage_hits": regex_payload.get("passage_hits", {}),
                },
                "llm": (
                    {
                        "used": True,
                        "signals": llm_payload.get("signals", {}) if isinstance(llm_payload, dict) else {},
                        "confidence": llm_payload.get("confidence", 0.0) if isinstance(llm_payload, dict) else 0.0,
                        "matched_passage_ids": llm_payload.get("matched_passage_ids", {}) if isinstance(llm_payload, dict) else {},
                        "rationale": llm_payload.get("rationale", "") if isinstance(llm_payload, dict) else "",
                    }
                    if isinstance(llm_payload, dict)
                    else {"used": False}
                ),
            },
            "evidence_refs": evidence_ref_payload,
        }

    def _detect_statistical_signals_regex(
        self,
        *,
        raw_text: str,
        passages: list[dict],
    ) -> dict:
        patterns = self._statistical_signal_patterns()
        signal_names = list(patterns.keys())

        signals: dict[str, bool] = {name: False for name in signal_names}
        raw_match_counts: dict[str, int] = {name: 0 for name in signal_names}
        passage_match_counts: dict[str, int] = {name: 0 for name in signal_names}
        passage_hits: dict[str, list[str]] = {name: [] for name in signal_names}
        signal_refs: dict[str, list[EvidenceRef]] = {name: [] for name in signal_names}

        for name, regexes in patterns.items():
            count = 0
            for rgx in regexes:
                count += len(rgx.findall(raw_text))
            raw_match_counts[name] = count
            if count > 0:
                signals[name] = True

        for passage in passages[:260]:
            text = str(passage.get("text") or "")
            lowered = text.lower()
            pid = str(passage.get("id") or "").strip()
            for name, regexes in patterns.items():
                matched = any(rgx.search(lowered) for rgx in regexes)
                if not matched:
                    continue
                signals[name] = True
                passage_match_counts[name] += 1
                if pid and pid not in passage_hits[name]:
                    passage_hits[name].append(pid)
                if len(signal_refs[name]) < 2:
                    signal_refs[name].append(
                        self._evidence_ref_from_passage(passage, excerpt=text[:180])
                    )

        return {
            "signals": signals,
            "raw_match_counts": raw_match_counts,
            "passage_match_counts": passage_match_counts,
            "passage_hits": passage_hits,
            "signal_refs": signal_refs,
        }

    def _detect_statistical_signals_with_executor(
        self,
        *,
        check_name: str,
        spec: dict[str, Any],
        passages: list[dict],
        ctx: PipelineContext | None = None,
    ) -> dict | None:
        if self.executor is None:
            return None

        candidate_rows = self._select_statistical_candidate_passages(passages)
        if not candidate_rows:
            return None

        spec_task = TaskSpec(
            task_type="statistical_significance_detection",
            prompt=(
                "Detect statistical reporting signals from paper passages. "
                "Judge only from provided text and return JSON."
            ),
            context={
                "check_name": check_name,
                "description": str(spec.get("description", "")),
                "candidate_passages": candidate_rows,
                "signals_to_detect": [
                    "mean_std",
                    "p_value",
                    "confidence_interval",
                    "seed_reporting",
                    "test_name",
                ],
            },
            output_schema={
                "signals": {
                    "mean_std": True,
                    "p_value": False,
                    "confidence_interval": False,
                    "seed_reporting": False,
                    "test_name": False,
                },
                "matched_passage_ids": {
                    "mean_std": ["S001_para0"],
                    "p_value": [],
                    "confidence_interval": [],
                    "seed_reporting": [],
                    "test_name": [],
                },
                "confidence": 0.0,
                "rationale": "string",
            },
            model_profile="extract",
        )

        result = self.executor.execute(spec_task)
        for warning in result.warnings:
            if ctx is not None:
                ctx.add_qa_issue(f"gap_detector_statistical_executor_warning:{warning}")
        if not result.ok:
            return None

        payload: Any = result.output
        if isinstance(payload, dict) and isinstance(payload.get("response"), dict):
            payload = payload.get("response")
        if not isinstance(payload, dict):
            return None

        raw_signals = payload.get("signals", {})
        if not isinstance(raw_signals, dict):
            return None
        expected_keys = {"mean_std", "p_value", "confidence_interval", "seed_reporting", "test_name"}
        if not any(key in raw_signals for key in expected_keys):
            return None

        normalized_signals = {
            "mean_std": self._to_bool(raw_signals.get("mean_std")),
            "p_value": self._to_bool(raw_signals.get("p_value")),
            "confidence_interval": self._to_bool(raw_signals.get("confidence_interval")),
            "seed_reporting": self._to_bool(raw_signals.get("seed_reporting")),
            "test_name": self._to_bool(raw_signals.get("test_name")),
        }

        matched_ids_raw = payload.get("matched_passage_ids", {})
        matched_ids: dict[str, list[str]] = {}
        if isinstance(matched_ids_raw, dict):
            for key, value in matched_ids_raw.items():
                if not isinstance(value, list):
                    continue
                matched_ids[str(key).strip().lower()] = [
                    str(x).strip() for x in value if str(x).strip()
                ][:4]

        by_pid = {
            str(row.get("id") or "").strip(): row
            for row in passages
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        refs: list[EvidenceRef] = []
        for signal in ("mean_std", "p_value", "confidence_interval", "seed_reporting", "test_name"):
            for pid in matched_ids.get(signal, []):
                row = by_pid.get(pid)
                if not isinstance(row, dict):
                    continue
                refs.append(self._evidence_ref_from_passage(row, excerpt=str(row.get("text") or "")[:180]))
                if len(refs) >= 3:
                    break
            if len(refs) >= 3:
                break

        try:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "signals": normalized_signals,
            "matched_passage_ids": matched_ids,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "rationale": str(payload.get("rationale", "")).strip(),
            "evidence_refs": refs,
        }

    @staticmethod
    def _statistical_signal_patterns() -> dict[str, list[re.Pattern[str]]]:
        return {
            "mean_std": [
                re.compile(r"\b\d+(?:\.\d+)?\s*(?:±|\+/-)\s*\d+(?:\.\d+)?\b", re.IGNORECASE),
                re.compile(r"\b(mean|avg|average)\b.{0,24}\b(std|sd|stdev|standard deviation|variance)\b", re.IGNORECASE),
                re.compile(r"\bmean\s*(?:\+/-|±|\\pm)\s*(?:std|sd|stdev)\b", re.IGNORECASE),
            ],
            "p_value": [
                re.compile(r"\bp\s*(?:<|<=|=|>|>=)\s*0?\.\d+\b", re.IGNORECASE),
                re.compile(r"\bp-?\s*value(?:s)?\b", re.IGNORECASE),
            ],
            "confidence_interval": [
                re.compile(r"\b(?:95|99)\s*%\s*(?:confidence interval|ci)\b", re.IGNORECASE),
                re.compile(r"\bconfidence interval(?:s)?\b", re.IGNORECASE),
                re.compile(r"\bci\s*[\[\(]\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*[\]\)]", re.IGNORECASE),
            ],
            "seed_reporting": [
                re.compile(r"\b\d+\s*seeds?\b", re.IGNORECASE),
                re.compile(r"\bmulti[-\s]?seed\b|\bmultiple\s+seeds\b", re.IGNORECASE),
            ],
            "test_name": [
                re.compile(r"\b(t-?test|paired t-?test|wilcoxon|mann[-\s]?whitney|anova|bootstrap)\b", re.IGNORECASE),
                re.compile(r"\bsignificance\s+test(?:s)?\b", re.IGNORECASE),
            ],
        }

    @staticmethod
    def _select_statistical_candidate_passages(passages: list[dict], limit: int = 24) -> list[dict]:
        candidates: list[tuple[int, dict]] = []
        cues = [
            "significance",
            "p-value",
            "confidence interval",
            "mean",
            "std",
            "seed",
            "t-test",
            "table",
            "figure",
        ]
        for row in passages[:300]:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            section = str(row.get("section") or "").strip().lower()
            kind = str(row.get("kind") or "").strip().lower()
            lowered = text.lower()
            score = 0
            if any(x in section for x in ["experiment", "result", "analysis", "evaluation"]):
                score += 3
            if any(x in kind for x in ["table", "figure"]):
                score += 2
            cue_hits = sum(1 for cue in cues if cue in lowered)
            score += cue_hits
            if score <= 0:
                continue
            candidates.append(
                (
                    score,
                    {
                        "id": str(row.get("id") or ""),
                        "section": str(row.get("section") or ""),
                        "text": text[:420],
                    },
                )
            )

        candidates.sort(key=lambda x: (-x[0], str(x[1].get("id", ""))))
        out: list[dict] = []
        seen: set[str] = set()
        for _, row in candidates:
            pid = str(row.get("id") or "").strip()
            if pid and pid in seen:
                continue
            if pid:
                seen.add(pid)
            out.append(row)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _dedupe_evidence_refs(refs: list[EvidenceRef], limit: int = 3) -> list[EvidenceRef]:
        out: list[EvidenceRef] = []
        seen: set[str] = set()
        for ref in refs:
            if not isinstance(ref, EvidenceRef):
                continue
            key = f"{ref.section}|{ref.passage_id}"
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _signal_display_name(signal: str) -> str:
        mapping = {
            "mean_std": "mean+/-std",
            "p_value": "p-value",
            "confidence_interval": "confidence interval",
            "seed_reporting": "seed reporting",
            "test_name": "test name",
        }
        return mapping.get(signal, signal.replace("_", " "))

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "present", "detected"}:
            return True
        return False

    def _evaluate_section_length_ratio_check(
        self,
        *,
        spec: dict,
        paper_structured: dict | None,
        passages: list[dict],
    ) -> dict:
        keywords = spec.get("keywords", ["introduction", "method", "experiments", "discussion"])
        if not isinstance(keywords, list) or not keywords:
            keywords = ["introduction", "method", "experiments", "discussion"]
        keywords = [str(x).strip().lower() for x in keywords if str(x).strip()]

        default_aliases = {
            "introduction": ["introduction", "background", "motivation", "overview"],
            "method": ["method", "approach", "model", "architecture", "methodology", "framework", "system"],
            "experiments": ["experiments", "evaluation", "results", "benchmark", "empirical"],
            "discussion": ["discussion", "analysis", "conclusion", "limitations", "error analysis", "future work"],
        }
        aliases = spec.get("section_aliases", {})
        if not isinstance(aliases, dict):
            aliases = {}
        merged_aliases: dict[str, list[str]] = {}
        for bucket, defaults in default_aliases.items():
            row = aliases.get(bucket, defaults)
            if not isinstance(row, list) or not row:
                row = defaults
            merged_aliases[bucket] = [str(x).strip().lower() for x in row if str(x).strip()]

        raw_targets = spec.get("section_ratio_targets", {})
        if not isinstance(raw_targets, dict):
            raw_targets = {}
        targets: dict[str, float] = {}
        for bucket in ("introduction", "method", "experiments", "discussion"):
            value = raw_targets.get(bucket, 0.0)
            try:
                targets[bucket] = max(0.0, float(value))
            except (TypeError, ValueError):
                targets[bucket] = 0.0
        if sum(targets.values()) <= 0:
            targets = {"introduction": 0.18, "method": 0.32, "experiments": 0.34, "discussion": 0.16}

        tolerance = float(spec.get("section_ratio_tolerance", 0.1) or 0.1)
        tolerance = max(0.02, min(0.35, tolerance))
        min_total_words = int(spec.get("section_ratio_min_total_words", 900) or 900)
        min_bucket_words = int(spec.get("section_ratio_min_bucket_words", 80) or 80)

        section_rows = []
        if isinstance(paper_structured, dict):
            rows = paper_structured.get("sections", [])
            if isinstance(rows, list):
                section_rows = [x for x in rows if isinstance(x, dict)]

        bucket_words = {bucket: 0 for bucket in targets}
        bucket_sections: dict[str, list[dict]] = {bucket: [] for bucket in targets}
        total_words = 0

        for section in section_rows:
            text = str(section.get("text") or "")
            words = self._word_count(text)
            if words <= 0:
                continue
            total_words += words
            bucket = self._match_section_bucket(str(section.get("name") or ""), merged_aliases)
            if bucket is None:
                continue
            bucket_words[bucket] += words
            bucket_sections[bucket].append(section)

        if total_words < min_total_words:
            return {
                "passed": True,
                "check_name": "section_length_ratio",
                "gap_code": str(spec.get("gap_code") or "section_ratio_imbalance"),
                "keywords": keywords,
                "description": str(spec.get("description") or "Section length ratio check skipped for short draft."),
                "severity_hint": str(spec.get("severity_hint") or "P2"),
                "hits": 1,
                "distinct_sections": 1,
                "thresholds": {
                    "min_total_words": min_total_words,
                    "section_ratio_tolerance": tolerance,
                    "section_ratio_targets": targets,
                    "section_ratio_min_bucket_words": min_bucket_words,
                },
                "section_ratio": {
                    "total_words": total_words,
                    "targets": targets,
                    "actual": {k: 0.0 for k in targets},
                    "deviation": {k: 0.0 for k in targets},
                    "status": "skipped_short_draft",
                },
            }

        actual_ratios = {
            bucket: (bucket_words[bucket] / total_words if total_words > 0 else 0.0)
            for bucket in targets
        }
        deviations = {
            bucket: round(abs(actual_ratios[bucket] - targets[bucket]), 4)
            for bucket in targets
        }
        failing_buckets = [
            bucket
            for bucket in targets
            if deviations[bucket] > tolerance
            and (bucket_words[bucket] >= min_bucket_words or targets[bucket] >= 0.1)
        ]
        passed = len(failing_buckets) == 0

        if passed:
            description = (
                "Section length ratio is within tolerance for introduction/method/experiments/discussion."
            )
        else:
            parts = []
            for bucket in failing_buckets:
                parts.append(
                    f"{bucket}={actual_ratios[bucket]:.2f} (target {targets[bucket]:.2f}, Δ{deviations[bucket]:.2f})"
                )
            description = "Section length ratio imbalance detected: " + "; ".join(parts)

        evidence_refs: list[EvidenceRef] = []
        if not passed:
            for bucket in failing_buckets:
                rows = bucket_sections.get(bucket, [])
                if not rows:
                    continue
                top = max(rows, key=lambda x: self._word_count(str(x.get("text") or "")))
                evidence_refs.append(self._evidence_ref_from_section(top))
            if not evidence_refs and passages:
                evidence_refs = self._probe_refs(passages, keywords)
        evidence_ref_payload = [ref.model_dump() for ref in evidence_refs]

        return {
            "passed": passed,
            "check_name": "section_length_ratio",
            "gap_code": str(spec.get("gap_code") or "section_ratio_imbalance"),
            "keywords": keywords,
            "description": description,
            "severity_hint": str(spec.get("severity_hint") or "P2"),
            "hits": max(1, len(section_rows)),
            "distinct_sections": len([k for k, v in bucket_words.items() if v > 0]),
            "thresholds": {
                "min_total_words": min_total_words,
                "section_ratio_tolerance": tolerance,
                "section_ratio_targets": targets,
                "section_ratio_min_bucket_words": min_bucket_words,
            },
            "section_ratio": {
                "total_words": total_words,
                "targets": {k: round(v, 4) for k, v in targets.items()},
                "actual": {k: round(v, 4) for k, v in actual_ratios.items()},
                "deviation": deviations,
                "failing_buckets": failing_buckets,
            },
            "evidence_refs": evidence_ref_payload,
        }

    def _evaluate_terminology_consistency_check(
        self,
        *,
        spec: dict,
        paper_structured: dict | None,
        passages: list[dict],
    ) -> dict:
        keywords = spec.get("keywords", ["terminology", "notation", "acronym", "consistency"])
        if not isinstance(keywords, list) or not keywords:
            keywords = ["terminology", "notation", "acronym", "consistency"]
        keywords = [str(x).strip().lower() for x in keywords if str(x).strip()]

        min_mentions = int(spec.get("terminology_min_mentions", 2) or 2)
        min_variant_hits = int(spec.get("terminology_min_variant_hits", 1) or 1)
        min_mentions = max(1, min_mentions)
        min_variant_hits = max(1, min_variant_hits)

        exempt_raw = spec.get("terminology_exempt_terms", [])
        exempt_terms = set()
        if isinstance(exempt_raw, list):
            exempt_terms = {
                self._normalize_term_text(str(x))
                for x in exempt_raw
                if str(x).strip()
            }

        section_rows: list[dict] = []
        if isinstance(paper_structured, dict):
            rows = paper_structured.get("sections", [])
            if isinstance(rows, list):
                section_rows = [x for x in rows if isinstance(x, dict)]

        inconsistencies = self._extract_terminology_inconsistencies(
            section_rows=section_rows,
            min_mentions=min_mentions,
            min_variant_hits=min_variant_hits,
            exempt_terms=exempt_terms,
        )
        passed = len(inconsistencies) == 0

        if passed:
            description = "Technical terminology appears consistent across major sections."
        else:
            top = inconsistencies[:3]
            parts = []
            for item in top:
                if item["type"] == "acronym_expansion_drift":
                    parts.append(
                        f"{item['acronym']} maps to multiple expansions: {', '.join(item['variants'][:3])}"
                    )
                else:
                    parts.append(
                        f"Concept '{item['concept_key']}' uses mixed forms: {', '.join(item['variants'][:3])}"
                    )
            description = "Terminology consistency issues detected: " + "; ".join(parts)

        evidence_refs: list[EvidenceRef] = []
        for item in inconsistencies[:3]:
            refs = self._find_refs_for_term_fragments(passages, item.get("variants", []))
            for ref in refs:
                evidence_refs.append(ref)
                if len(evidence_refs) >= 3:
                    break
            if len(evidence_refs) >= 3:
                break
        if not evidence_refs and passages and not passed:
            evidence_refs = self._probe_refs(passages, keywords)

        evidence_ref_payload = [ref.model_dump() for ref in evidence_refs]
        return {
            "passed": passed,
            "check_name": "terminology_consistency",
            "gap_code": str(spec.get("gap_code") or "terminology_inconsistency"),
            "keywords": keywords,
            "description": description,
            "severity_hint": str(spec.get("severity_hint") or "P2"),
            "hits": len(inconsistencies),
            "distinct_sections": len(section_rows),
            "thresholds": {
                "terminology_min_mentions": min_mentions,
                "terminology_min_variant_hits": min_variant_hits,
            },
            "terminology_consistency": {
                "issue_count": len(inconsistencies),
                "items": inconsistencies[:8],
            },
            "evidence_refs": evidence_ref_payload,
        }

    def _extract_terminology_inconsistencies(
        self,
        *,
        section_rows: list[dict],
        min_mentions: int,
        min_variant_hits: int,
        exempt_terms: set[str],
    ) -> list[dict]:
        acronym_to_expansions: dict[str, dict[str, int]] = {}
        concept_forms: dict[str, dict[str, int]] = {}

        acronym_first_pattern = re.compile(r"\b([A-Z]{2,})\s*\(([^)]+)\)")
        expansion_first_pattern = re.compile(
            r"\b([A-Z][A-Za-z0-9]+(?:[\s\-][A-Z][A-Za-z0-9]+){1,5})\s*\(([A-Z]{2,})\)"
        )
        term_pattern = re.compile(r"\b([A-Za-z][A-Za-z0-9]+(?:[\-\s][A-Za-z0-9]+){1,3})\b")

        for section in section_rows:
            text = str(section.get("text") or "")
            if not text:
                continue

            for match in expansion_first_pattern.finditer(text):
                expansion = self._normalize_term_text(match.group(1))
                acronym = match.group(2).upper()
                if expansion and expansion not in exempt_terms:
                    row = acronym_to_expansions.setdefault(acronym, {})
                    row[expansion] = row.get(expansion, 0) + 1

            for match in acronym_first_pattern.finditer(text):
                acronym = match.group(1).upper()
                expansion = self._normalize_term_text(match.group(2))
                if expansion and expansion not in exempt_terms and 2 <= len(expansion.split()) <= 7:
                    row = acronym_to_expansions.setdefault(acronym, {})
                    row[expansion] = row.get(expansion, 0) + 1

            for match in term_pattern.finditer(text):
                raw_form = str(match.group(1)).strip()
                normalized = self._normalize_term_text(raw_form)
                if not normalized or normalized in exempt_terms:
                    continue
                if len(normalized) < 7:
                    continue
                token_count = len(normalized.split())
                if token_count < 2:
                    continue
                key = self._term_key(normalized)
                row = concept_forms.setdefault(key, {})
                row[raw_form] = row.get(raw_form, 0) + 1

        issues: list[dict] = []
        for acronym, expansions in acronym_to_expansions.items():
            variants = [
                exp
                for exp, count in expansions.items()
                if count >= min_variant_hits and exp not in exempt_terms
            ]
            total = sum(expansions.values())
            if total >= min_mentions and len(variants) >= 2:
                issues.append(
                    {
                        "type": "acronym_expansion_drift",
                        "acronym": acronym,
                        "concept_key": acronym.lower(),
                        "variants": sorted(variants)[:6],
                        "counts": {k: int(v) for k, v in sorted(expansions.items(), key=lambda x: (-x[1], x[0]))[:6]},
                    }
                )

        for concept_key, forms in concept_forms.items():
            total = sum(forms.values())
            variants = [form for form, count in forms.items() if count >= min_variant_hits]
            if total < min_mentions or len(variants) < 2:
                continue
            if len({self._normalize_term_text(v) for v in variants}) < 2:
                continue
            issues.append(
                {
                    "type": "term_variant_mismatch",
                    "acronym": "",
                    "concept_key": concept_key,
                    "variants": sorted(variants)[:6],
                    "counts": {k: int(v) for k, v in sorted(forms.items(), key=lambda x: (-x[1], x[0]))[:6]},
                }
            )

        # Keep deterministic and prioritize stronger evidence (more mentions).
        issues.sort(
            key=lambda x: (
                0 if x.get("type") == "acronym_expansion_drift" else 1,
                -sum(int(v) for v in x.get("counts", {}).values()),
                str(x.get("concept_key", "")),
            )
        )
        return issues

    @staticmethod
    def _normalize_term_text(text: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9\-\s]", " ", str(text or ""))
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        return cleaned

    @staticmethod
    def _term_key(normalized_text: str) -> str:
        return re.sub(r"[\s\-_]+", "", str(normalized_text or "").lower())

    @staticmethod
    def _find_refs_for_term_fragments(passages: list[dict], fragments: list[str]) -> list[EvidenceRef]:
        refs: list[EvidenceRef] = []
        seen: set[str] = set()
        for fragment in fragments:
            frag = str(fragment or "").strip().lower()
            if not frag:
                continue
            for passage in passages:
                text = str(passage.get("text") or "")
                if frag and frag in text.lower():
                    pid = str(passage.get("id") or "")
                    if pid in seen:
                        continue
                    seen.add(pid)
                    refs.append(GapDetectorStep._evidence_ref_from_passage(passage, excerpt=text[:180]))
                    break
            if len(refs) >= 2:
                break
        return refs

    @staticmethod
    def _match_section_bucket(section_name: str, aliases: dict[str, list[str]]) -> str | None:
        name = str(section_name or "").strip().lower()
        if not name:
            return None
        for bucket in ("introduction", "method", "experiments", "discussion"):
            for alias in aliases.get(bucket, []):
                if alias and alias in name:
                    return bucket
        return None

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"\b[\w\-]+\b", str(text or "")))

    @staticmethod
    def _evidence_ref_from_section(section: dict) -> EvidenceRef:
        text = str(section.get("text") or "")
        section_id = str(section.get("section_id") or "")
        section_index = int(section.get("section_index", 0) or 0)
        name = str(section.get("name") or "unknown")
        return EvidenceRef(
            section=name,
            passage_id=f"{section_id}_ratio" if section_id else f"{name}_ratio",
            excerpt=text[:180],
            section_id=section_id,
            section_index=section_index,
            page=0,
            kind="section_ratio_anchor",
            anchor_label=name,
            anchor_type="section",
            locator={"source": "section_ratio_check", "section_id": section_id, "section_index": section_index},
        )

    @staticmethod
    def _probe_refs(passages: list[dict], keywords: list[str]) -> list[EvidenceRef]:
        refs: list[EvidenceRef] = []
        for passage in passages:
            text = str(passage.get("text") or "")
            if any(k in text.lower() for k in keywords):
                refs.append(GapDetectorStep._evidence_ref_from_passage(passage, excerpt=text[:180]))
            if len(refs) >= 2:
                break

        # If nothing matched, include first paragraph as anchor so report has traceability.
        if not refs and passages:
            top = passages[0]
            refs.append(
                GapDetectorStep._evidence_ref_from_passage(
                    top,
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
        stance_counts = stats.get("outgoing_stance_counts", {}) if isinstance(stats.get("outgoing_stance_counts", {}), dict) else {}
        support_ratio = float(stats.get("outgoing_support_ratio", 0.0) or 0.0)
        challenge_ratio = float(stats.get("outgoing_challenge_ratio", 0.0) or 0.0)
        stance_context_coverage = float(stats.get("outgoing_stance_context_coverage_ratio", 0.0) or 0.0)

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

        challenge_count = int(stance_counts.get("challenging", 0) or 0)
        support_count = int(stance_counts.get("supporting", 0) or 0)
        if outgoing_count >= 6 and stance_context_coverage >= 0.2:
            if challenge_count >= 3 and challenge_ratio > support_ratio + 0.2:
                refs = self._citation_refs(citation_graph, relation="outgoing", stance="challenging")
                if not refs:
                    refs = self._citation_refs(citation_graph, relation="outgoing")
                gaps.append(
                    GapItem(
                        code="citation_context_challenge_dominant",
                        severity_hint="P2",
                        description=(
                            "Citation context suggests challenge-heavy positioning (more critical comparisons than supporting context). "
                            "Clarify novelty boundary and fairness framing to avoid over-claim interpretation."
                        ),
                        evidence_refs=refs,
                    ).model_dump()
                )
            elif support_count <= 1 and challenge_count >= 1 and challenge_ratio >= 0.28:
                refs = self._citation_refs(citation_graph, relation="outgoing", stance="challenging")
                if not refs:
                    refs = self._citation_refs(citation_graph, relation="outgoing")
                gaps.append(
                    GapItem(
                        code="citation_context_low_support_signal",
                        severity_hint="P2",
                        description=(
                            "Citation context has weak explicit support signal. Add citations that directly validate assumptions "
                            "or reported trends to improve argumentative balance."
                        ),
                        evidence_refs=refs,
                    ).model_dump()
                )

        return gaps

    @staticmethod
    def _citation_refs(citation_graph: dict, relation: str, limit: int = 2, stance: str = "") -> list[EvidenceRef]:
        key = "outgoing_references" if relation == "outgoing" else "incoming_citations"
        items = citation_graph.get(key, [])
        if not isinstance(items, list):
            return []
        refs: list[EvidenceRef] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            if stance:
                item_stance = str(item.get("citation_stance", "")).strip().lower()
                if item_stance != stance.strip().lower():
                    continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            stance_tag = str(item.get("citation_stance", "")).strip().lower()
            prefix = f"[{stance_tag}] " if stance_tag else ""
            refs.append(
                EvidenceRef(
                    section="citation_graph",
                    passage_id=f"{relation}:{idx}",
                    excerpt=f"{prefix}{title}"[:180],
                )
            )
            if len(refs) >= limit:
                break
        return refs

    @staticmethod
    def _evidence_ref_from_passage(passage: dict, *, excerpt: str) -> EvidenceRef:
        locator = passage.get("locator", {})
        if not isinstance(locator, dict):
            locator = {}
        return EvidenceRef(
            section=str(passage.get("section") or "unknown"),
            passage_id=str(passage.get("id") or "unknown"),
            excerpt=str(excerpt or "")[:180],
            section_id=str(passage.get("section_id", "") or ""),
            section_index=int(passage.get("section_index", 0) or 0),
            page=int(passage.get("page", 0) or 0),
            kind=str(passage.get("kind", "") or ""),
            anchor_label=str(passage.get("anchor_label", "") or ""),
            anchor_type=str(passage.get("anchor_type", "") or ""),
            locator=locator,
        )
