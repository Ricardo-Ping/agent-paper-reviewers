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
            )
            source = "rule_fallback"

        payload = {
            "venue": venue,
            "year": year,
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
                ],
            },
            output_schema={
                "questions": [
                    {
                        "priority": "high|medium|low",
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
                    "reviewer_persona": str(item.get("reviewer_persona", "empirical")).strip() or "empirical",
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
