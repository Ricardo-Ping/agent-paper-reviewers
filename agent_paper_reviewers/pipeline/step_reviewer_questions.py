from __future__ import annotations

import json
import re

from ..executors.base import ExecutorAdapter
from ..models import TaskSpec
from .base import PipelineContext, PipelineStep


class ReviewerQuestionSimulatorStep(PipelineStep):
    name = "ReviewerQuestionSimulator"

    def __init__(self, executor: ExecutorAdapter | None = None) -> None:
        self.executor = executor

    def run(self, ctx: PipelineContext) -> None:
        venue = str(ctx.input_data.venue.name or "")
        year = int(ctx.input_data.venue.year or 0)
        manuscript_stage = ctx.input_data.review_context.manuscript_stage.value
        reviewer_comments = [
            {
                "review_id": str(x.review_id or "").strip() or f"R{idx}",
                "concern": str(x.concern or "").strip(),
            }
            for idx, x in enumerate(ctx.input_data.review_context.reviewer_comments, start=1)
            if str(x.concern or "").strip()
        ]
        gaps = ctx.artifacts.get("gaps", {}).get("gaps", [])
        risks = ctx.artifacts.get("risk_ranking", {}).get("risks", [])
        alignments = ctx.artifacts.get("claim_evidence_matrix", {}).get("alignments", [])
        venue_profile = ctx.artifacts.get("venue_profile", {}).get("profile", {})
        paper_structured = ctx.artifacts.get("paper_structured", {})

        questions = self._simulate_with_executor(
            ctx=ctx,
            venue=venue,
            year=year,
            gaps=gaps,
            risks=risks,
            alignments=alignments,
            venue_profile=venue_profile,
            paper_structured=paper_structured,
            manuscript_stage=manuscript_stage,
            reviewer_comments=reviewer_comments,
        )
        source = "executor"
        if questions is None:
            questions = self._simulate_rule_based(
                venue=venue,
                year=year,
                gaps=gaps if isinstance(gaps, list) else [],
                risks=risks if isinstance(risks, list) else [],
                alignments=alignments if isinstance(alignments, list) else [],
                paper_structured=paper_structured if isinstance(paper_structured, dict) else {},
                manuscript_stage=manuscript_stage,
                reviewer_comments=reviewer_comments,
            )
            source = "rule_fallback"
        else:
            fallback_questions = self._simulate_rule_based(
                venue=venue,
                year=year,
                gaps=gaps if isinstance(gaps, list) else [],
                risks=risks if isinstance(risks, list) else [],
                alignments=alignments if isinstance(alignments, list) else [],
                paper_structured=paper_structured if isinstance(paper_structured, dict) else {},
                manuscript_stage=manuscript_stage,
                reviewer_comments=reviewer_comments,
            )
            questions, enriched = self._merge_and_enforce_coverage(
                questions=questions,
                fallback_questions=fallback_questions,
                manuscript_stage=manuscript_stage,
            )
            if enriched:
                source = "executor_plus_rule_enrichment"

        questions = self._attach_evidence_anchors(
            questions=questions if isinstance(questions, list) else [],
            risks=risks if isinstance(risks, list) else [],
            alignments=alignments if isinstance(alignments, list) else [],
        )

        payload = {
            "venue": venue,
            "year": year,
            "manuscript_stage": manuscript_stage,
            "source": source,
            "questions": questions[:10],
        }
        ctx.artifacts["reviewer_questions"] = payload
        ctx.dump_json("artifacts/reviewer_questions.json", payload)

    def _simulate_with_executor(
        self,
        *,
        ctx: PipelineContext,
        venue: str,
        year: int,
        gaps: list,
        risks: list,
        alignments: list,
        venue_profile: dict,
        paper_structured: dict,
        manuscript_stage: str,
        reviewer_comments: list[dict],
    ) -> list[dict] | None:
        if self.executor is None:
            return None

        spec = TaskSpec(
            task_type="reviewer_question_simulation",
            prompt=(
                "You are simulating strict reviewer follow-up questions. "
                "Given current weaknesses, predict concrete questions reviewers are likely to ask next. "
                "Return JSON only."
            ),
            context={
                "venue": venue,
                "year": year,
                "manuscript_stage": manuscript_stage,
                "reviewer_comments": reviewer_comments[:8],
                "top_risks": risks[:8] if isinstance(risks, list) else [],
                "gaps": gaps[:12] if isinstance(gaps, list) else [],
                "weak_alignments": [
                    a
                    for a in (alignments if isinstance(alignments, list) else [])
                    if str(a.get("strength", "")).lower() in {"none", "weak"}
                ][:8],
                "venue_common_reject_reasons": venue_profile.get("common_reject_reasons", []),
                "paper_title": str(paper_structured.get("title", "")),
                "paper_hint_text": str(paper_structured.get("raw_text", ""))[:3000],
                "requirements": [
                    "Questions must be reviewer-style and specific, not generic.",
                    "Each question must include why it will be asked and what evidence author should prepare.",
                    "Prefer venue-specific language (e.g., SIGMOD/VLDB/ICDE for DB systems).",
                    "Ensure diversity: include both high- and medium-priority questions.",
                    "If manuscript_stage=meta_review_discussion, prioritize direct follow-up questions on reviewer concerns.",
                    "If manuscript_stage=initial_submission, prioritize predictive pre-submission challenge questions.",
                ],
            },
            output_schema={
                "questions": [
                    {
                        "priority": "high|medium|low",
                        "question_type": "stage_followup|evidence_closure|novelty_boundary|baseline_fairness|stats_rigor|ablation_attribution|reproducibility",
                        "reviewer_persona": "systems|theory|empirical|reproducibility",
                        "question": "string",
                        "why_this_will_be_asked": "string",
                        "trigger_gap_codes": ["string"],
                        "linked_risk_ids": ["RISK-001"],
                        "evidence_to_prepare": ["string"],
                        "suggested_response_strategy": "string",
                    }
                ]
            },
            model_profile="judge",
        )

        result = self.executor.execute(spec)
        for warning in result.warnings:
            ctx.add_qa_issue(f"reviewer_question_simulator_executor_warning:{warning}")
        if not result.ok:
            ctx.add_qa_issue("reviewer_question_simulator_executor_not_ok_use_rule_fallback")
            return None

        questions = self._normalize_questions(result.output)
        if not questions:
            ctx.add_qa_issue("reviewer_question_simulator_executor_output_invalid_use_rule_fallback")
            return None
        return questions

    @staticmethod
    def _normalize_questions(raw_output: object) -> list[dict]:
        data = raw_output
        if isinstance(raw_output, dict):
            if isinstance(raw_output.get("response"), dict):
                data = raw_output["response"]
            elif isinstance(raw_output.get("response"), str):
                parsed = ReviewerQuestionSimulatorStep._parse_json_like(raw_output["response"])
                if parsed is not None:
                    data = parsed
                else:
                    data = raw_output

        raw_questions = None
        if isinstance(data, dict):
            raw_questions = data.get("questions")
        elif isinstance(data, list):
            raw_questions = data
        if not isinstance(raw_questions, list):
            return []

        out: list[dict] = []
        seen_questions: set[str] = set()
        for idx, item in enumerate(raw_questions, start=1):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            if len(question) < 12:
                continue
            key = re.sub(r"\s+", " ", question.lower())
            if key in seen_questions:
                continue
            seen_questions.add(key)

            priority = str(item.get("priority", "medium")).strip().lower()
            if priority not in {"high", "medium", "low"}:
                priority = "medium"
            question_type = str(item.get("question_type", "evidence_closure")).strip().lower()
            if not question_type:
                question_type = "evidence_closure"
            reviewer_persona = ReviewerQuestionSimulatorStep._normalize_persona_slug(
                str(item.get("reviewer_persona", "empirical")).strip() or "empirical"
            )

            trigger_gap_codes = item.get("trigger_gap_codes", [])
            if not isinstance(trigger_gap_codes, list):
                trigger_gap_codes = []
            linked_risk_ids = item.get("linked_risk_ids", [])
            if not isinstance(linked_risk_ids, list):
                linked_risk_ids = []
            evidence_to_prepare = item.get("evidence_to_prepare", [])
            if not isinstance(evidence_to_prepare, list):
                evidence_to_prepare = []
            if not evidence_to_prepare:
                evidence_to_prepare = ["Prepare one concrete new evidence block linked to this question."]

            out.append(
                {
                    "id": f"RQ-{idx:03d}",
                    "priority": priority,
                    "question_type": question_type,
                    "reviewer_persona": reviewer_persona,
                    "question": question,
                    "why_this_will_be_asked": str(item.get("why_this_will_be_asked", "")).strip()
                    or "This question follows from detected weaknesses in the current draft.",
                    "trigger_gap_codes": [str(x).strip() for x in trigger_gap_codes if str(x).strip()],
                    "linked_risk_ids": [str(x).strip() for x in linked_risk_ids if str(x).strip()],
                    "evidence_to_prepare": [str(x).strip() for x in evidence_to_prepare if str(x).strip()][:4],
                    "suggested_response_strategy": str(item.get("suggested_response_strategy", "")).strip()
                    or "Answer with claim-linked numeric evidence and exact paper-change locations.",
                }
            )

        return out[:10]

    @staticmethod
    def _normalize_persona_slug(value: str) -> str:
        slug = str(value or "").strip().lower()
        mapping = {
            "methodology reviewer": "methodology_reviewer",
            "method reviewer": "methodology_reviewer",
            "methodology": "methodology_reviewer",
            "method": "methodology_reviewer",
            "empirical reviewer": "empirical_reviewer",
            "empirical": "empirical_reviewer",
            "theory reviewer": "theory_reviewer",
            "theoretical reviewer": "theory_reviewer",
            "theory": "theory_reviewer",
        }
        if slug in mapping:
            return mapping[slug]
        return slug or "empirical_reviewer"

    @classmethod
    def _attach_evidence_anchors(
        cls,
        *,
        questions: list[dict],
        risks: list[dict],
        alignments: list[dict],
    ) -> list[dict]:
        risk_anchor_map: dict[str, list[dict]] = {}
        for risk in risks:
            if not isinstance(risk, dict):
                continue
            risk_id = str(risk.get("id", "")).strip()
            if not risk_id:
                continue
            refs = cls._normalize_anchor_refs(risk.get("evidence_refs", []))
            if refs:
                risk_anchor_map[risk_id] = refs

        weak_alignment_refs: list[dict] = []
        contradiction_refs: list[dict] = []
        for row in alignments:
            if not isinstance(row, dict):
                continue
            strength = str(row.get("strength", "")).strip().lower()
            if strength in {"none", "weak"}:
                weak_alignment_refs.extend(cls._normalize_anchor_refs(row.get("evidence_refs", []))[:1])
            contradiction_refs.extend(
                cls._normalize_anchor_refs(row.get("contradictory_evidence_refs", []))[:1]
            )
        weak_alignment_refs = cls._dedupe_anchor_refs(weak_alignment_refs)[:3]
        contradiction_refs = cls._dedupe_anchor_refs(contradiction_refs)[:3]

        out: list[dict] = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            item = dict(q)
            linked_risk_ids = item.get("linked_risk_ids", [])
            if not isinstance(linked_risk_ids, list):
                linked_risk_ids = []
            trigger_codes = item.get("trigger_gap_codes", [])
            if not isinstance(trigger_codes, list):
                trigger_codes = []

            anchors: list[dict] = []
            for rid in linked_risk_ids:
                anchors.extend(risk_anchor_map.get(str(rid).strip(), []))

            lower_codes = {str(x).strip().lower() for x in trigger_codes if str(x).strip()}
            if "weak_claim_alignment" in lower_codes or "missing_significance" in lower_codes:
                anchors.extend(weak_alignment_refs)
            if "claim_evidence_contradiction" in lower_codes:
                anchors.extend(contradiction_refs)

            anchors = cls._dedupe_anchor_refs(anchors)[:3]
            item["evidence_anchor_refs"] = anchors
            item["evidence_anchor_hint"] = cls._anchor_hint(anchors)

            ev_prepare = item.get("evidence_to_prepare", [])
            if not isinstance(ev_prepare, list):
                ev_prepare = []
            anchor_hint = str(item.get("evidence_anchor_hint", "")).strip()
            if anchor_hint and not any("[see:" in str(x).lower() for x in ev_prepare):
                ev_prepare.append(f"Anchor pointers: {anchor_hint}")
            item["evidence_to_prepare"] = [str(x).strip() for x in ev_prepare if str(x).strip()][:5]
            out.append(item)
        return out

    @staticmethod
    def _normalize_anchor_refs(refs: object) -> list[dict]:
        if not isinstance(refs, list):
            return []
        out: list[dict] = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            section = str(ref.get("section", "")).strip()
            passage_id = str(ref.get("passage_id", "")).strip()
            if not section or not passage_id:
                continue
            try:
                page = int(ref.get("page", 0) or 0)
            except (TypeError, ValueError):
                page = 0
            out.append(
                {
                    "section": section,
                    "passage_id": passage_id,
                    "excerpt": str(ref.get("excerpt", "")).strip()[:220],
                    "page": page,
                    "anchor_label": str(ref.get("anchor_label", "")).strip(),
                }
            )
        return out

    @staticmethod
    def _dedupe_anchor_refs(refs: list[dict]) -> list[dict]:
        out: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            key = (
                str(ref.get("section", "")).strip().lower(),
                str(ref.get("passage_id", "")).strip().lower(),
            )
            if not all(key) or key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out

    @staticmethod
    def _anchor_hint(refs: list[dict]) -> str:
        hints: list[str] = []
        for ref in refs[:3]:
            if not isinstance(ref, dict):
                continue
            section = str(ref.get("section", "")).strip()
            passage_id = str(ref.get("passage_id", "")).strip()
            if not section or not passage_id:
                continue
            hints.append(f"[see: {section} -> {passage_id}]")
        return " ".join(hints)

    @staticmethod
    def _parse_json_like(text: str) -> object | None:
        s = str(text or "").strip()
        if not s:
            return None

        # 1) direct JSON parse
        try:
            return json.loads(s)
        except Exception:  # noqa: BLE001
            pass

        # 2) fenced markdown JSON block
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
        if fence:
            block = fence.group(1).strip()
            try:
                return json.loads(block)
            except Exception:  # noqa: BLE001
                pass

        # 3) extract first plausible JSON object/array segment
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = s.find(open_ch)
            if start < 0:
                continue
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = s[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except Exception:  # noqa: BLE001
                            break
        return None

    def _simulate_rule_based(
        self,
        *,
        venue: str,
        year: int,
        gaps: list[dict],
        risks: list[dict],
        alignments: list[dict],
        paper_structured: dict,
        manuscript_stage: str,
        reviewer_comments: list[dict],
    ) -> list[dict]:
        venue_slug = venue.strip().lower().replace("_", "-").replace(" ", "-")
        gap_codes = [str(g.get("code", "")).strip().lower() for g in gaps if isinstance(g, dict)]
        weak_claims = [
            str(a.get("claim_id", "")).strip()
            for a in alignments
            if isinstance(a, dict) and str(a.get("strength", "")).lower() in {"none", "weak"}
        ]

        out: list[dict] = []
        seen: set[str] = set()

        def add_question(item: dict) -> None:
            q = str(item.get("question", "")).strip()
            if not q:
                return
            key = re.sub(r"\s+", " ", q.lower())
            if key in seen:
                return
            seen.add(key)
            item["id"] = f"RQ-{len(out) + 1:03d}"
            out.append(item)

        for gap in gaps:
            if not isinstance(gap, dict):
                continue
            code = str(gap.get("code", "")).strip().lower()
            if not code:
                continue
            for q in self._templates_for_gap(code, venue_slug, weak_claims):
                q["trigger_gap_codes"] = [code]
                q["linked_risk_ids"] = self._linked_risk_ids(code, risks)
                add_question(q)

        for q in self._role_based_persona_questions(
            manuscript_stage=manuscript_stage,
            risks=risks,
            gap_codes=gap_codes,
            weak_claims=weak_claims,
            reviewer_comments=reviewer_comments,
            paper_text=str(paper_structured.get("raw_text", "")),
            venue_slug=venue_slug,
        ):
            add_question(q)

        for q in self._stage_specific_followups(
            manuscript_stage=manuscript_stage,
            reviewer_comments=reviewer_comments,
            risks=risks,
            weak_claims=weak_claims,
        ):
            add_question(q)

        # DB/system venues: add targeted follow-ups for system-level novelty.
        if venue_slug in {"sigmod", "vldb", "icde"}:
            for q in self._db_system_followups(
                paper_text=str(paper_structured.get("raw_text", "")),
                gap_codes=gap_codes,
            ):
                q["trigger_gap_codes"] = list(dict.fromkeys(q.get("trigger_gap_codes", []) + gap_codes[:1]))
                q["linked_risk_ids"] = self._linked_risk_ids("missing_baseline", risks)
                add_question(q)

        # If still too sparse, add one generic high-value follow-up.
        if not out:
            add_question(
                {
                    "priority": "high",
                    "question_type": "evidence_closure",
                    "reviewer_persona": "empirical",
                    "question": "Can you provide one direct numeric evidence block for each core claim under matched settings?",
                    "why_this_will_be_asked": "Reviewers penalize papers where claims are not mapped to concrete, reproducible evidence.",
                    "trigger_gap_codes": gap_codes[:2],
                    "linked_risk_ids": self._linked_risk_ids("weak_claim_alignment", risks),
                    "evidence_to_prepare": [
                        "Claim-to-evidence table with section/figure/table anchors.",
                        "Matched-setting baseline comparison and significance summary.",
                    ],
                    "suggested_response_strategy": "Answer claim-by-claim with explicit numbers and where each update appears in the paper.",
                }
            )

        # Priority sort.
        rank = {"high": 0, "medium": 1, "low": 2}
        out.sort(key=lambda x: (rank.get(str(x.get("priority", "medium")), 1), x.get("id", "")))

        return out[:10]

    @staticmethod
    def _templates_for_gap(code: str, venue_slug: str, weak_claims: list[str]) -> list[dict]:
        claim_hint = f" (especially {', '.join(weak_claims[:2])})" if weak_claims else ""
        mapping: dict[str, list[dict]] = {
            "missing_baseline": [
                {
                    "priority": "high",
                    "question_type": "baseline_fairness",
                    "reviewer_persona": "systems",
                    "question": "Which strongest baselines were compared under strictly matched data/compute settings, and what are the exact numeric deltas?",
                    "why_this_will_be_asked": "Weak baseline fairness is a common rejection trigger.",
                    "evidence_to_prepare": [
                        "Matched-budget comparison table with baseline tuning protocol.",
                        "Per-dataset numeric deltas and win/loss breakdown.",
                    ],
                    "suggested_response_strategy": "Show one fairness table and explicitly explain matched settings.",
                }
            ],
            "missing_significance": [
                {
                    "priority": "high",
                    "question_type": "stats_rigor",
                    "reviewer_persona": "empirical",
                    "question": f"Are the reported gains statistically significant{claim_hint}, with multi-seed mean/std and paired tests?",
                    "why_this_will_be_asked": "Single-run gains are often treated as unstable evidence.",
                    "evidence_to_prepare": [
                        "Multi-seed mean/std table for all primary metrics.",
                        "Paired significance tests and confidence intervals.",
                    ],
                    "suggested_response_strategy": "Answer with exact test setup, seeds, and p-values.",
                }
            ],
            "missing_ablation": [
                {
                    "priority": "medium",
                    "question_type": "ablation_attribution",
                    "reviewer_persona": "method",
                    "question": "Which component truly drives the gain, and do interaction ablations confirm this attribution?",
                    "why_this_will_be_asked": "Without controlled ablation, reviewers cannot trust contribution attribution.",
                    "evidence_to_prepare": [
                        "One-by-one component ablations.",
                        "Interaction ablations for key component pairs.",
                    ],
                    "suggested_response_strategy": "Provide causal interpretation tied to ablation numbers.",
                }
            ],
            "weak_claim_alignment": [
                {
                    "priority": "high",
                    "question_type": "evidence_closure",
                    "reviewer_persona": "meta-review",
                    "question": "Can each core claim be mapped to one direct evidence anchor (table/figure/section) without ambiguity?",
                    "why_this_will_be_asked": "Reviewer confidence drops quickly when claim-evidence mapping is indirect.",
                    "evidence_to_prepare": [
                        "Claim-evidence matrix with exact anchors.",
                        "Short explanation per claim on what evidence validates it.",
                    ],
                    "suggested_response_strategy": "Respond claim-by-claim, not paragraph-by-paragraph.",
                }
            ],
            "claim_evidence_contradiction": [
                {
                    "priority": "high",
                    "question_type": "stage_followup",
                    "reviewer_persona": "critical",
                    "question": "Some results appear to contradict your claim direction; where exactly is the claim scope corrected and reconciled?",
                    "why_this_will_be_asked": "Unresolved claim-result conflicts can directly cause reject decisions.",
                    "evidence_to_prepare": [
                        "Contradiction reconciliation table (claim -> conflicting result -> corrected interpretation).",
                        "Revised claim wording with scope conditions.",
                    ],
                    "suggested_response_strategy": "Acknowledge conflict first, then provide corrected scope and new analysis.",
                }
            ],
            "missing_reference_coverage": [
                {
                    "priority": "medium",
                    "question_type": "novelty_boundary",
                    "reviewer_persona": "related-work",
                    "question": "Which nearest prior works are most comparable, and where is the explicit novelty boundary stated?",
                    "why_this_will_be_asked": "Thin related-work positioning weakens novelty credibility.",
                    "evidence_to_prepare": [
                        "Nearest-neighbor prior-work comparison table.",
                        "Novelty boundary paragraph with direct references.",
                    ],
                    "suggested_response_strategy": "Use concrete comparisons instead of broad novelty claims.",
                }
            ],
            "missing_top_venue_related_work_coverage": [
                {
                    "priority": "medium",
                    "question_type": "novelty_boundary",
                    "reviewer_persona": "related-work",
                    "question": "How does this work differ from recent top-venue papers in the last 2-3 years on the same task?",
                    "why_this_will_be_asked": "Top-venue comparisons are expected for novelty positioning.",
                    "evidence_to_prepare": [
                        "Recent top-venue reference list with relevance notes.",
                        "Table: prior methods vs ours on assumptions and outcomes.",
                    ],
                    "suggested_response_strategy": "State concrete differences, not only citation additions.",
                }
            ],
        }

        rows = mapping.get(code, [])
        if venue_slug in {"sigmod", "vldb", "icde"} and code == "missing_baseline":
            rows = list(rows) + [
                {
                    "priority": "high",
                    "question_type": "baseline_fairness",
                    "reviewer_persona": "db-systems",
                    "question": "Are throughput/latency gains still valid across workload diversity and scale, not only one benchmark slice?",
                    "why_this_will_be_asked": "DB systems reviewers expect workload and scalability robustness.",
                    "evidence_to_prepare": [
                        "Workload-diversity table across OLTP/OLAP-like query sets.",
                        "Scale-up/scale-out curves with resource budget notes.",
                    ],
                    "suggested_response_strategy": "Quantify where gains hold and where they degrade.",
                }
            ]
        return rows

    @staticmethod
    def _db_system_followups(*, paper_text: str, gap_codes: list[str]) -> list[dict]:
        t = paper_text.lower()
        if "sql" not in t and "dialect" not in t and "query" not in t:
            return []

        rows: list[dict] = []
        if any(c in gap_codes for c in ["missing_baseline", "weak_claim_alignment", "missing_reference_coverage"]):
            rows.append(
                {
                    "priority": "high",
                    "question_type": "baseline_fairness",
                    "reviewer_persona": "db-systems",
                    "question": "What are the exact numeric comparisons against SQLGlot (and at least one non-LLM SQL parser) under the same workload and hardware budget?",
                    "why_this_will_be_asked": "System-level novelty claims in SQL translation are often challenged by direct tool-level comparisons.",
                    "trigger_gap_codes": ["missing_baseline"],
                    "evidence_to_prepare": [
                        "Comparison table: success rate / exact match / execution accuracy vs SQLGlot and non-LLM parser.",
                        "Matched workload and hardware budget protocol.",
                    ],
                    "suggested_response_strategy": "Show compute-normalized numbers and explicit fairness constraints.",
                }
            )
        rows.append(
            {
                "priority": "medium",
                "question_type": "stage_followup",
                "reviewer_persona": "empirical",
                "question": "Why is evaluation limited to one LLM family, and how stable are results across alternative LLM backbones?",
                "why_this_will_be_asked": "Reviewer confidence drops when generalization across model choices is unclear.",
                "trigger_gap_codes": ["weak_claim_alignment"],
                "evidence_to_prepare": [
                    "Cross-backbone robustness table (at least one additional LLM family).",
                    "Error profile comparison across backbones and dialect pairs.",
                ],
                "suggested_response_strategy": "Answer with cross-model robustness and failure pattern differences.",
            }
        )
        return rows

    @staticmethod
    def _stage_specific_followups(
        *,
        manuscript_stage: str,
        reviewer_comments: list[dict],
        risks: list[dict],
        weak_claims: list[str],
    ) -> list[dict]:
        rows: list[dict] = []
        weak_hint = f" (especially {', '.join(weak_claims[:2])})" if weak_claims else ""

        if manuscript_stage == "initial_submission":
            rows.extend(
                [
                    {
                        "priority": "high",
                        "question_type": "reproducibility",
                        "reviewer_persona": "reproducibility",
                        "question": "Which exact reproducibility assets (code, scripts, seeds, hardware/runtime config) can be released at submission time?",
                        "why_this_will_be_asked": "Top venues increasingly reject papers with weak reproducibility details.",
                        "trigger_gap_codes": ["missing_reproducibility"],
                        "linked_risk_ids": ReviewerQuestionSimulatorStep._linked_risk_ids("weak_claim_alignment", risks),
                        "evidence_to_prepare": [
                            "Reproducibility checklist with links/paths for code, data, and scripts.",
                            "Runtime/environment table with seeds and deterministic settings.",
                        ],
                        "suggested_response_strategy": "Provide a concrete release plan and map each artifact to the corresponding experiment section.",
                    },
                    {
                        "priority": "medium",
                        "question_type": "novelty_boundary",
                        "reviewer_persona": "related-work",
                        "question": "What is the precise novelty boundary versus nearest prior work, and where are the non-overlapping contributions stated?",
                        "why_this_will_be_asked": "Novelty claims that are too broad are frequently challenged before acceptance.",
                        "trigger_gap_codes": ["missing_reference_coverage"],
                        "linked_risk_ids": ReviewerQuestionSimulatorStep._linked_risk_ids("missing_reference_coverage", risks),
                        "evidence_to_prepare": [
                            "One table comparing assumptions, setting, and gains against 3-5 nearest papers.",
                            "One paragraph with explicit non-overlap contribution boundaries.",
                        ],
                        "suggested_response_strategy": "Use concrete pairwise comparisons instead of broad statements like 'first' or 'state of the art'.",
                    },
                    {
                        "priority": "medium",
                        "question_type": "evidence_closure",
                        "reviewer_persona": "empirical",
                        "question": f"Which failure cases remain and how do they affect the headline claims{weak_hint}?",
                        "why_this_will_be_asked": "Pre-submission reviewers often probe negative cases to test claim robustness.",
                        "trigger_gap_codes": ["weak_claim_alignment"],
                        "linked_risk_ids": ReviewerQuestionSimulatorStep._linked_risk_ids("weak_claim_alignment", risks),
                        "evidence_to_prepare": [
                            "Failure-case table with error categories and frequency.",
                            "Scope-corrected claim wording tied to failure analysis.",
                        ],
                        "suggested_response_strategy": "Acknowledge boundaries explicitly and quantify where the method underperforms.",
                    },
                ]
            )
            return rows

        if manuscript_stage in {"meta_review_discussion", "rejected_after_reviews"}:
            for idx, item in enumerate(reviewer_comments[:5], start=1):
                concern = str(item.get("concern", "")).strip()
                review_id = str(item.get("review_id", "")).strip() or f"R{idx}"
                if not concern:
                    continue
                inferred_codes = ReviewerQuestionSimulatorStep._infer_gap_codes_from_concern(concern)
                linked: list[str] = []
                for code in inferred_codes:
                    linked.extend(ReviewerQuestionSimulatorStep._linked_risk_ids(code, risks))
                linked = list(dict.fromkeys([x for x in linked if x]))[:3]
                rows.append(
                    {
                        "priority": "high",
                        "question_type": "stage_followup",
                        "reviewer_persona": "meta-review",
                        "question": f"Reviewer {review_id} raised '{concern}'. What new numeric evidence and exact paper edits directly close this concern?",
                        "why_this_will_be_asked": "Discussion-stage decisions depend on whether each explicit reviewer concern is concretely closed.",
                        "trigger_gap_codes": inferred_codes,
                        "linked_risk_ids": linked,
                        "evidence_to_prepare": [
                            f"Before/after rebuttal mapping for {review_id}: concern -> evidence -> paper change location.",
                            "One concise numeric evidence block that directly answers this concern.",
                        ],
                        "suggested_response_strategy": "Respond concern-by-concern with one evidence anchor and one exact revision location.",
                    }
                )

            rows.append(
                {
                    "priority": "medium",
                    "question_type": "evidence_closure",
                    "reviewer_persona": "critical",
                    "question": "Which rebuttal claims still rely on promises instead of completed evidence, and what is the completion deadline?",
                    "why_this_will_be_asked": "Meta reviewers discount responses that promise future work without concrete updates.",
                    "trigger_gap_codes": ["weak_claim_alignment"],
                    "linked_risk_ids": ReviewerQuestionSimulatorStep._linked_risk_ids("weak_claim_alignment", risks),
                    "evidence_to_prepare": [
                        "Checklist of completed vs promised items with status and proof.",
                        "Updated claim-evidence matrix including newly added results.",
                    ],
                    "suggested_response_strategy": "Separate completed evidence from future plans and prioritize completed evidence in the rebuttal narrative.",
                }
            )
        return rows

    @staticmethod
    def _role_based_persona_questions(
        *,
        manuscript_stage: str,
        risks: list[dict],
        gap_codes: list[str],
        weak_claims: list[str],
        reviewer_comments: list[dict],
        paper_text: str,
        venue_slug: str,
    ) -> list[dict]:
        risk_ids_stats = ReviewerQuestionSimulatorStep._linked_risk_ids("missing_significance", risks)
        risk_ids_align = ReviewerQuestionSimulatorStep._linked_risk_ids("weak_claim_alignment", risks)
        risk_ids_novelty = ReviewerQuestionSimulatorStep._linked_risk_ids("missing_reference_coverage", risks)
        weak_hint = f" (especially {', '.join(weak_claims[:2])})" if weak_claims else ""
        in_db = venue_slug in {"sigmod", "vldb", "icde"}

        methodology_question = (
            "Which methodological assumptions are essential for each claimed gain, and where is causal attribution isolated component-by-component?"
        )
        if manuscript_stage != "initial_submission" and reviewer_comments:
            top = str(reviewer_comments[0].get("concern", "")).strip()
            if top:
                methodology_question = (
                    f"For reviewer concern '{top}', which methodological assumptions are revised, and which ablation/controlled analyses isolate causality?"
                )
        if in_db:
            methodology_question = (
                "How does the system architecture isolate causality between parsing/optimization modules and end-to-end SQL translation gains?"
            )

        empirical_question = (
            f"What is the strict empirical validation protocol{weak_hint}: matched baselines, multi-seed significance, and robustness across settings?"
        )
        if manuscript_stage == "meta_review_discussion":
            empirical_question = (
                "Which new numbers directly answer each reviewer concern, and are they backed by multi-seed significance under matched settings?"
            )
        elif manuscript_stage == "rejected_after_reviews":
            empirical_question = (
                "Which reject-trigger metrics are newly fixed, and what exact before/after numbers prove empirical closure?"
            )

        theory_question = (
            "What is the precise theoretical boundary of the method (assumptions, failure regime, and non-guaranteed cases), and where is it explicitly stated?"
        )
        if in_db:
            theory_question = (
                "Under what query/workload assumptions does the method hold, and where are theoretical failure boundaries quantified?"
            )
        if manuscript_stage != "initial_submission" and reviewer_comments:
            theory_question = (
                "Which claim statements are too strong under current assumptions, and how will scope-corrected theoretical boundaries be written?"
            )

        return [
            {
                "priority": "high",
                "question_type": "methodology_rigor",
                "reviewer_persona": "methodology_reviewer",
                "question": methodology_question,
                "why_this_will_be_asked": (
                    "Methodology reviewers focus on whether contribution attribution is causally justified rather than correlational."
                ),
                "trigger_gap_codes": list(dict.fromkeys((gap_codes[:2] or []) + ["missing_ablation"])),
                "linked_risk_ids": list(dict.fromkeys((risk_ids_align + risk_ids_stats)[:3])),
                "evidence_to_prepare": [
                    "Component-level and interaction ablations tied to each methodological claim.",
                    "One assumption-to-evidence table: assumption -> test -> observed effect.",
                ],
                "suggested_response_strategy": (
                    "Answer in causal order: assumption, intervention, measured effect, and scope boundary."
                ),
            },
            {
                "priority": "high",
                "question_type": "empirical_validation",
                "reviewer_persona": "empirical_reviewer",
                "question": empirical_question,
                "why_this_will_be_asked": (
                    "Empirical reviewers prioritize reproducible quantitative closure with fair baselines and significance."
                ),
                "trigger_gap_codes": list(dict.fromkeys((gap_codes[:2] or []) + ["missing_significance"])),
                "linked_risk_ids": list(dict.fromkeys((risk_ids_stats + risk_ids_align)[:3])),
                "evidence_to_prepare": [
                    "Matched-setting baseline table with per-dataset deltas.",
                    "Multi-seed mean/std, confidence intervals, and paired tests.",
                ],
                "suggested_response_strategy": (
                    "Lead with exact numbers and test setup; avoid qualitative-only claims."
                ),
            },
            {
                "priority": "medium",
                "question_type": "theory_soundness",
                "reviewer_persona": "theory_reviewer",
                "question": theory_question,
                "why_this_will_be_asked": (
                    "Theory reviewers probe assumption validity, boundary conditions, and whether claims exceed proven scope."
                ),
                "trigger_gap_codes": list(dict.fromkeys((gap_codes[:2] or []) + ["missing_reference_coverage"])),
                "linked_risk_ids": list(dict.fromkeys((risk_ids_novelty + risk_ids_align)[:3])),
                "evidence_to_prepare": [
                    "Explicit assumption list and counter-example/failure regime discussion.",
                    "Scope-corrected claim wording aligned with theoretical boundaries.",
                ],
                "suggested_response_strategy": (
                    "State formal scope first, then map each claim to validated assumptions."
                ),
            },
        ]

    @staticmethod
    def _infer_gap_codes_from_concern(concern: str) -> list[str]:
        t = concern.lower()
        out: list[str] = []
        if any(k in t for k in ["baseline", "comparison", "fair"]):
            out.append("missing_baseline")
        if any(k in t for k in ["significance", "statistical", "p-value", "seed"]):
            out.append("missing_significance")
        if "ablation" in t or "component" in t:
            out.append("missing_ablation")
        if any(k in t for k in ["reproduc", "code", "release", "artifact"]):
            out.append("missing_reproducibility")
        if any(k in t for k in ["related work", "citation", "novelty", "prior"]):
            out.append("missing_reference_coverage")
        if any(k in t for k in ["evidence", "support", "claim", "justify"]):
            out.append("weak_claim_alignment")
        if not out:
            out.append("weak_claim_alignment")
        return list(dict.fromkeys(out))

    @staticmethod
    def _coverage_targets(manuscript_stage: str) -> dict:
        mapping = {
            "initial_submission": {
                "min_total": 6,
                "min_priority": {"high": 2, "medium": 3},
                "required_personas": {
                    "methodology_reviewer",
                    "empirical_reviewer",
                    "theory_reviewer",
                },
            },
            "meta_review_discussion": {
                "min_total": 6,
                "min_priority": {"high": 3, "medium": 2},
                "required_personas": {
                    "methodology_reviewer",
                    "empirical_reviewer",
                    "theory_reviewer",
                    "meta-review",
                },
            },
            "rejected_after_reviews": {
                "min_total": 6,
                "min_priority": {"high": 3, "medium": 2},
                "required_personas": {
                    "methodology_reviewer",
                    "empirical_reviewer",
                    "theory_reviewer",
                    "critical",
                },
            },
        }
        return mapping.get(manuscript_stage, mapping["initial_submission"])

    def _merge_and_enforce_coverage(
        self,
        *,
        questions: list[dict],
        fallback_questions: list[dict],
        manuscript_stage: str,
    ) -> tuple[list[dict], bool]:
        if not isinstance(questions, list):
            questions = []
        if not isinstance(fallback_questions, list):
            fallback_questions = []

        def norm_key(item: dict) -> str:
            q = str(item.get("question", "")).strip().lower()
            return re.sub(r"\s+", " ", q)

        def normalize_item(item: dict) -> dict:
            priority = str(item.get("priority", "medium")).strip().lower()
            if priority not in {"high", "medium", "low"}:
                priority = "medium"
            return {
                "priority": priority,
                "question_type": str(item.get("question_type", "evidence_closure")).strip().lower()
                or "evidence_closure",
                "reviewer_persona": ReviewerQuestionSimulatorStep._normalize_persona_slug(
                    str(item.get("reviewer_persona", "empirical")).strip() or "empirical"
                ),
                "question": str(item.get("question", "")).strip(),
                "why_this_will_be_asked": str(item.get("why_this_will_be_asked", "")).strip()
                or "This question follows from detected weaknesses in the current draft.",
                "trigger_gap_codes": [
                    str(x).strip()
                    for x in item.get("trigger_gap_codes", [])
                    if str(x).strip()
                ]
                if isinstance(item.get("trigger_gap_codes", []), list)
                else [],
                "linked_risk_ids": [
                    str(x).strip()
                    for x in item.get("linked_risk_ids", [])
                    if str(x).strip()
                ]
                if isinstance(item.get("linked_risk_ids", []), list)
                else [],
                "evidence_to_prepare": [
                    str(x).strip()
                    for x in item.get("evidence_to_prepare", [])
                    if str(x).strip()
                ][:4]
                if isinstance(item.get("evidence_to_prepare", []), list)
                else [],
                "suggested_response_strategy": str(item.get("suggested_response_strategy", "")).strip()
                or "Answer with claim-linked numeric evidence and exact paper-change locations.",
            }

        merged: list[dict] = []
        seen: set[str] = set()
        for item in questions:
            if not isinstance(item, dict):
                continue
            key = norm_key(item)
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalize_item(item))

        fallback_pool: list[dict] = []
        for item in fallback_questions:
            if not isinstance(item, dict):
                continue
            key = norm_key(item)
            if not key or key in seen:
                continue
            fallback_pool.append(normalize_item(item))

        enriched = False
        policy = self._coverage_targets(manuscript_stage)
        min_total = int(policy.get("min_total", 6))
        min_priority = policy.get("min_priority", {})
        required_personas = set(policy.get("required_personas", set()))

        def current_priority_count(label: str) -> int:
            return sum(1 for q in merged if str(q.get("priority", "")).lower() == label)

        for label in ("high", "medium"):
            need = int(min_priority.get(label, 0))
            while current_priority_count(label) < need:
                idx = next(
                    (
                        i
                        for i, row in enumerate(fallback_pool)
                        if str(row.get("priority", "")).lower() == label
                    ),
                    -1,
                )
                if idx < 0:
                    break
                merged.append(fallback_pool.pop(idx))
                enriched = True

        existing_personas = {str(q.get("reviewer_persona", "")).strip() for q in merged}
        for persona in sorted(required_personas):
            if persona in existing_personas:
                continue
            idx = next(
                (
                    i
                    for i, row in enumerate(fallback_pool)
                    if str(row.get("reviewer_persona", "")).strip() == persona
                ),
                -1,
            )
            if idx < 0:
                continue
            merged.append(fallback_pool.pop(idx))
            existing_personas.add(persona)
            enriched = True

        while len(merged) < min_total and fallback_pool:
            merged.append(fallback_pool.pop(0))
            enriched = True

        rank = {"high": 0, "medium": 1, "low": 2}
        merged.sort(key=lambda x: (rank.get(str(x.get("priority", "medium")).lower(), 1), str(x.get("question", "")).lower()))
        out: list[dict] = []
        for idx, row in enumerate(merged[:10], start=1):
            item = dict(row)
            item["id"] = f"RQ-{idx:03d}"
            if not item.get("evidence_to_prepare"):
                item["evidence_to_prepare"] = ["Prepare one concrete new evidence block linked to this question."]
            out.append(item)
        return out, enriched

    @staticmethod
    def _linked_risk_ids(code: str, risks: list[dict]) -> list[str]:
        if not isinstance(risks, list):
            return []
        code = code.lower()
        keywords_map = {
            "missing_baseline": ["baseline", "fair", "comparison"],
            "missing_significance": ["significance", "statistical", "p-value"],
            "missing_ablation": ["ablation", "component"],
            "weak_claim_alignment": ["claim", "evidence", "support"],
            "claim_evidence_contradiction": ["contradiction", "conflict"],
            "missing_reference_coverage": ["reference", "related work", "citation"],
            "missing_top_venue_related_work_coverage": ["top-venue", "recent", "related work"],
        }
        keywords = keywords_map.get(code, [])
        out: list[str] = []
        for risk in risks:
            if not isinstance(risk, dict):
                continue
            rid = str(risk.get("id", "")).strip()
            reason = str(risk.get("reason", "")).lower()
            if not rid:
                continue
            if any(k in reason for k in keywords):
                out.append(rid)
        return out[:3]
