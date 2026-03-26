from __future__ import annotations

from difflib import SequenceMatcher
import re

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency fallback
    fuzz = None

from ..models import ClaimAlignment, EvidenceRef
from ..services.embedding import cosine_similarity, encode_texts
from .base import PipelineContext, PipelineStep


class ClaimEvidenceAlignerStep(PipelineStep):
    name = "ClaimEvidenceAligner"

    def run(self, ctx: PipelineContext) -> None:
        claims = ctx.artifacts["claims_normalized"]["claims"]
        passages = ctx.artifacts["evidence_index"]["passages"]
        evidence_vectors = ctx.artifacts.get("evidence_vectors", {})
        domain_profile = self._domain_profile(
            paper_structured=ctx.artifacts.get("paper_structured", {}),
            claims=claims,
        )

        passage_embeddings = [evidence_vectors.get(p.get("id", ""), []) for p in passages]
        if not passage_embeddings or not all(isinstance(v, list) and v for v in passage_embeddings):
            passage_embeddings, _ = encode_texts(p.get("text", "") for p in passages)

        claim_queries = [self._claim_query(claim) for claim in claims]
        claim_embeddings, _ = encode_texts(claim_queries)

        matrix: list[dict] = []
        for claim, claim_vec in zip(claims, claim_embeddings):
            claim_profile = self._claim_profile(claim)
            scored = []
            contradiction_rows = []
            for passage, passage_vec in zip(passages, passage_embeddings):
                quality = float(passage.get("quality_score", 0.5) or 0.5)
                kind = str(passage.get("kind", "paragraph"))
                section = str(passage.get("section", ""))

                # Hard skip noisy low-quality narrative passages.
                if kind in {"paragraph", "figure_table_mention"} and quality < 0.2:
                    continue

                semantic = cosine_similarity(claim_vec, passage_vec)
                lexical = self._lexical_score(claim["claim_text"], passage["text"])
                semantic_weight, lexical_weight = self._similarity_weights(
                    claim_text=str(claim.get("claim_text", "")),
                    passage_text=str(passage.get("text", "")),
                    claim_profile=claim_profile,
                    domain_profile=domain_profile,
                )
                section_prior = self._section_prior(claim_profile, section)
                kind_prior = self._kind_prior(claim_profile, kind)
                contradiction = self._contradiction_score(claim, claim_profile, str(passage.get("text", "")))

                base = (
                    semantic_weight * semantic
                    + lexical_weight * lexical
                    + 0.1 * section_prior
                    + 0.08 * kind_prior
                )
                quality_factor = 0.65 + 0.5 * quality
                support_score = round(base * quality_factor * (1.0 - 0.6 * contradiction), 4)

                if contradiction >= 0.45:
                    contradiction_rows.append(
                        (
                            contradiction,
                            semantic,
                            lexical,
                            section_prior,
                            kind_prior,
                            quality,
                            passage,
                        )
                    )
                if support_score > 0.1:
                    scored.append(
                        (
                            support_score,
                            semantic,
                            lexical,
                            semantic_weight,
                            lexical_weight,
                            section_prior,
                            kind_prior,
                            quality,
                            contradiction,
                            passage,
                        )
                    )

            scored.sort(key=lambda x: x[0], reverse=True)
            top = self._select_top_evidence(scored, claim_profile)
            top_score = top[0][0] if top else 0.0
            contradiction_candidates = sorted(
                contradiction_rows,
                key=lambda x: float(x[0]),
                reverse=True,
            )
            contradiction_score = float(contradiction_candidates[0][0]) if contradiction_candidates else 0.0
            strength = self._strength(top_score)
            strength = self._adjust_strength_for_contradiction(strength, contradiction_score)

            refs = [self._support_evidence_ref(item) for item in top]
            contradictory_refs = [
                self._contradiction_evidence_ref(item) for item in contradiction_candidates[:2]
            ]
            confidence_summary = self._evidence_confidence_summary(refs, contradictory_refs)

            record = ClaimAlignment(
                claim_id=claim["claim_id"],
                claim_text=claim["claim_text"],
                strength=strength,
                score=round(top_score, 3),
                evidence_refs=refs,
            )
            row = record.model_dump()
            row["claim_type"] = str(claim.get("claim_type", "novelty"))
            row["contradiction_detected"] = bool(contradictory_refs)
            row["contradiction_score"] = round(contradiction_score, 3)
            row["contradictory_evidence_refs"] = [r.model_dump() for r in contradictory_refs]
            row["contradiction_summary"] = self._contradiction_summary(claim, contradiction_score, contradictory_refs)
            row["evidence_confidence"] = confidence_summary
            row["conflict_alert"] = bool(
                row["contradiction_detected"]
                or any(bool(ref.conflict_alert) for ref in refs)
            )
            row["diagnostics"] = {
                "claim_profile": claim_profile,
                "selected_sections": [str(item[9].get("section", "")) for item in top],
                "selected_kinds": [str(item[9].get("kind", "")) for item in top],
                "avg_quality": round(sum(item[7] for item in top) / len(top), 3) if top else 0.0,
                "contradiction_candidates": len(contradiction_candidates),
                "domain_profile": domain_profile,
            }
            if top:
                row["score_breakdown"] = {
                    "semantic": round(top[0][1], 3),
                    "lexical": round(top[0][2], 3),
                    "weights": {
                        "semantic": round(top[0][3], 3),
                        "lexical": round(top[0][4], 3),
                        "section_prior": 0.1,
                        "kind_prior": 0.08,
                    },
                    "section_prior": round(top[0][5], 3),
                    "kind_prior": round(top[0][6], 3),
                    "quality": round(top[0][7], 3),
                    "contradiction_penalty": round(top[0][8], 3),
                }
            matrix.append(row)

        payload = {
            "alignments": matrix,
            "weighting_policy": {
                "name": "adaptive_semantic_lexical",
                "core_similarity_weight": 0.82,
                "section_prior_weight": 0.1,
                "kind_prior_weight": 0.08,
                "domain_profile": domain_profile,
            },
            "evidence_confidence_policy": {
                "support": {"strong_threshold": 0.75, "medium_threshold": 0.5},
                "contradiction": {"strong_threshold": 0.7, "medium_threshold": 0.48},
                "labels": ["Strong", "Medium", "Weak"],
            },
        }
        ctx.artifacts["claim_evidence_matrix"] = payload
        ctx.dump_json("artifacts/claim_evidence_matrix.json", payload)

    @staticmethod
    def _claim_query(claim: dict) -> str:
        parts = [
            str(claim.get("claim_text", "")),
            str(claim.get("verifiable_claim", "")),
            str(claim.get("success_criteria", "")),
        ]
        return "\n".join(p for p in parts if p).strip()

    @staticmethod
    def _lexical_score(a: str, b: str) -> float:
        if fuzz is not None:
            return fuzz.token_set_ratio(a, b) / 100.0
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _token_set(text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-_]{2,}", text.lower())
        stop = {
            "this",
            "that",
            "with",
            "from",
            "using",
            "method",
            "approach",
            "model",
            "paper",
            "results",
            "show",
        }
        return {t for t in tokens if t not in stop}

    @staticmethod
    def _domain_profile(paper_structured: dict, claims: list[dict]) -> dict:
        title = str((paper_structured or {}).get("title", ""))
        raw_text = str((paper_structured or {}).get("raw_text", ""))[:12000]
        claim_text = " ".join(str(c.get("claim_text", "")) for c in claims if isinstance(c, dict))
        blob = f"{title}\n{claim_text}\n{raw_text}".lower()

        sql_keywords = [
            "sql",
            "dialect",
            "query",
            "database",
            "dbms",
            "parser",
            "ast",
            "schema",
            "execution",
            "benchmark",
        ]
        sql_hits = sum(1 for k in sql_keywords if k in blob)
        is_sql_domain = sql_hits >= 2

        technical_tokens = re.findall(r"\b[a-z]+(?:-[a-z]+)+\b|\b[A-Z]{2,}\b", f"{title} {claim_text}")
        technical_density = len(technical_tokens) / max(1, len(re.findall(r"\w+", f"{title} {claim_text}")))
        return {
            "is_sql_domain": bool(is_sql_domain),
            "sql_keyword_hits": sql_hits,
            "technical_density": round(min(1.0, technical_density), 3),
        }

    def _similarity_weights(
        self,
        *,
        claim_text: str,
        passage_text: str,
        claim_profile: dict,
        domain_profile: dict,
    ) -> tuple[float, float]:
        claim_tokens = self._token_set(claim_text)
        passage_tokens = self._token_set(passage_text)

        jaccard = 0.0
        if claim_tokens and passage_tokens:
            jaccard = len(claim_tokens & passage_tokens) / max(1, len(claim_tokens | passage_tokens))

        # Core similarity (semantic + lexical) kept at 0.82; split adaptively.
        semantic_share = 0.85
        if jaccard >= 0.35:
            semantic_share = 0.74
        elif jaccard < 0.12:
            semantic_share = 0.93

        if bool(domain_profile.get("is_sql_domain")):
            # SQL/dialect narratives often use heterogeneous terminology.
            semantic_share += 0.03

        if bool(claim_profile.get("empirical")) and jaccard >= 0.18:
            semantic_share -= 0.04
        if str(claim_profile.get("claim_type", "")) == "statistical" and jaccard >= 0.14:
            semantic_share -= 0.03

        semantic_share = max(0.7, min(0.96, semantic_share))
        semantic_weight = round(0.82 * semantic_share, 4)
        lexical_weight = round(0.82 - semantic_weight, 4)
        return semantic_weight, lexical_weight

    @staticmethod
    def _claim_profile(claim: dict) -> dict:
        claim_type = str(claim.get("claim_type", "novelty")).lower()
        blob = " ".join(
            [
                str(claim.get("claim_text", "")),
                str(claim.get("verifiable_claim", "")),
                str(claim.get("success_criteria", "")),
            ]
        ).lower()

        empirical_keywords = [
            "improve",
            "outperform",
            "accuracy",
            "f1",
            "bleu",
            "latency",
            "efficiency",
            "benchmark",
            "result",
            "score",
        ]
        empirical = claim_type in {"baseline", "ablation", "statistical"} or any(k in blob for k in empirical_keywords)
        reproducibility = claim_type == "reproducibility" or "reproduc" in blob
        expects_higher = any(k in blob for k in ["outperform", "higher", "increase", "improve", "gain"])
        expects_lower = any(k in blob for k in ["lower", "reduce", "smaller", "less", "decrease"])
        metric_groups = {
            "higher_better": [k for k in ["accuracy", "f1", "bleu", "auc", "precision", "recall", "throughput", "success rate"] if k in blob],
            "lower_better": [k for k in ["error", "latency", "runtime", "time", "cost", "memory"] if k in blob],
        }
        return {
            "claim_type": claim_type,
            "empirical": empirical,
            "reproducibility": reproducibility,
            "expects_higher": expects_higher,
            "expects_lower": expects_lower,
            "metric_groups": metric_groups,
        }

    @staticmethod
    def _contradiction_score(claim: dict, claim_profile: dict, passage_text: str) -> float:
        claim_text = (
            " ".join(
                [
                    str(claim.get("claim_text", "")),
                    str(claim.get("verifiable_claim", "")),
                    str(claim.get("success_criteria", "")),
                ]
            )
            .lower()
            .strip()
        )
        passage = passage_text.lower()
        if not passage:
            return 0.0

        higher_metrics = set(claim_profile.get("metric_groups", {}).get("higher_better", []) or [])
        lower_metrics = set(claim_profile.get("metric_groups", {}).get("lower_better", []) or [])

        metric_overlap = 0
        for m in higher_metrics | lower_metrics:
            if m and m in passage:
                metric_overlap += 1

        contradiction = 0.0
        if claim_profile.get("expects_higher") or higher_metrics:
            if any(
                k in passage
                for k in [
                    "worse",
                    "underperform",
                    "lower than",
                    "inferior",
                    "drop",
                    "decrease",
                    "behind",
                    "regression",
                    "cannot outperform",
                ]
            ):
                contradiction += 0.5
        if claim_profile.get("expects_lower") or lower_metrics:
            if any(
                k in passage
                for k in ["higher than", "increase", "larger", "more than", "slower", "more latency", "higher error"]
            ):
                contradiction += 0.5

        if str(claim_profile.get("claim_type", "")) == "statistical":
            if any(k in passage for k in ["not significant", "no significant", "p > 0.05", "insignificant"]):
                contradiction += 0.65

        if any(k in claim_text for k in ["all baselines", "every baseline", "all methods"]):
            if any(k in passage for k in ["not all", "some baseline", "fails on", "except"]):
                contradiction += 0.45

        numeric_conflict = ClaimEvidenceAlignerStep._numeric_conflict_bonus(claim_profile, passage)
        if contradiction > 0.0:
            contradiction += min(0.2, 0.08 * metric_overlap)
        if numeric_conflict > 0:
            contradiction += numeric_conflict + min(0.12, 0.05 * metric_overlap)
        return max(0.0, min(1.0, round(contradiction, 4)))

    @staticmethod
    def _numeric_conflict_bonus(claim_profile: dict, passage_text: str) -> float:
        t = passage_text.lower()
        ours = ClaimEvidenceAlignerStep._extract_named_number(t, ["ours", "our method", "proposed"])
        baseline = ClaimEvidenceAlignerStep._extract_named_number(t, ["baseline", "sota", "prior", "existing"])
        if ours is None or baseline is None:
            return 0.0
        if claim_profile.get("expects_higher") and ours < baseline:
            return 0.35
        if claim_profile.get("expects_lower") and ours > baseline:
            return 0.35
        return 0.0

    @staticmethod
    def _extract_named_number(text: str, anchors: list[str]) -> float | None:
        for anchor in anchors:
            pattern = re.escape(anchor) + r"[^0-9\-+]{0,20}([-+]?\d+(?:\.\d+)?)"
            m = re.search(pattern, text)
            if not m:
                continue
            try:
                return float(m.group(1))
            except Exception:  # noqa: BLE001
                continue
        return None

    @staticmethod
    def _adjust_strength_for_contradiction(strength: str, contradiction_score: float) -> str:
        order = ["None", "Weak", "Medium", "Strong"]
        idx = order.index(strength) if strength in order else 0
        if contradiction_score >= 0.75:
            idx = max(0, idx - 2)
        elif contradiction_score >= 0.5:
            idx = max(0, idx - 1)
        return order[idx]

    @staticmethod
    def _contradiction_summary(claim: dict, contradiction_score: float, refs: list[EvidenceRef]) -> str:
        if contradiction_score < 0.45 or not refs:
            return ""
        claim_id = str(claim.get("claim_id", "claim"))
        top = refs[0]
        return (
            f"Potential contradiction detected for {claim_id}: "
            f"evidence in {top.section}/{top.passage_id} may conflict with the claim direction."
        )

    @staticmethod
    def _section_prior(claim_profile: dict, section: str) -> float:
        sec = (section or "").lower()
        empirical = bool(claim_profile.get("empirical"))
        reproducibility = bool(claim_profile.get("reproducibility"))

        if empirical:
            if sec in {"experiments", "results", "ablation", "analysis", "figures_tables"}:
                return 1.0
            if sec in {"method", "approach", "model"}:
                return 0.4
            if sec in {"abstract", "introduction", "conclusion"}:
                return 0.45
            return 0.35

        if reproducibility:
            if sec in {"appendix", "method", "experiments", "analysis"}:
                return 1.0
            return 0.45

        # novelty-like claims
        if sec in {"abstract", "introduction", "related work", "method", "conclusion"}:
            return 1.0
        if sec in {"experiments", "results"}:
            return 0.65
        return 0.45

    @staticmethod
    def _kind_prior(claim_profile: dict, kind: str) -> float:
        empirical = bool(claim_profile.get("empirical"))
        reproducibility = bool(claim_profile.get("reproducibility"))
        k = (kind or "").lower()

        if empirical:
            if k in {"table_data", "table_content", "figure_content"}:
                return 1.0
            if k == "figure_table_mention":
                return 0.75
            return 0.6

        if reproducibility:
            if k in {"paragraph", "table_content", "figure_table_mention"}:
                return 0.9
            return 0.6

        if k == "paragraph":
            return 1.0
        if k in {"figure_table_mention", "table_content"}:
            return 0.7
        return 0.55

    def _select_top_evidence(self, scored: list[tuple], claim_profile: dict) -> list[tuple]:
        if not scored:
            return []

        top = scored[:6]
        # Diversity pass: try to keep at least two different sections in final refs.
        selected: list[tuple] = []
        seen_sections: set[str] = set()
        for item in top:
            section = str(item[9].get("section", ""))
            if section not in seen_sections or len(selected) < 2:
                selected.append(item)
                seen_sections.add(section)
            if len(selected) >= 3:
                break
        if not selected:
            selected = top[:3]

        # Empirical claims: force one anchor from experiments/results/ablation if candidate is close enough.
        if claim_profile.get("empirical"):
            empirical_sections = {"experiments", "results", "ablation", "analysis", "figures_tables"}
            has_empirical = any(str(x[9].get("section", "")).lower() in empirical_sections for x in selected)
            if not has_empirical:
                best_score = selected[0][0] if selected else 0.0
                for cand in top:
                    sec = str(cand[9].get("section", "")).lower()
                    if sec in empirical_sections and cand[0] >= best_score - 0.1:
                        if selected:
                            selected[-1] = cand
                        else:
                            selected.append(cand)
                        break

        selected.sort(key=lambda x: x[0], reverse=True)
        return selected[:3]

    @staticmethod
    def _strength(score: float) -> str:
        if score >= 0.72:
            return "Strong"
        if score >= 0.54:
            return "Medium"
        if score >= 0.36:
            return "Weak"
        return "None"

    def _support_evidence_ref(self, item: tuple) -> EvidenceRef:
        (
            support_score,
            semantic,
            lexical,
            _semantic_weight,
            _lexical_weight,
            section_prior,
            kind_prior,
            quality,
            contradiction,
            passage,
        ) = item
        confidence_score = self._support_confidence_score(
            support_score=float(support_score),
            semantic=float(semantic),
            lexical=float(lexical),
            section_prior=float(section_prior),
            kind_prior=float(kind_prior),
            quality=float(quality),
            contradiction=float(contradiction),
        )
        confidence_level = self._confidence_level(confidence_score, strong=0.75, medium=0.5)
        conflict_alert = float(contradiction) >= 0.45
        conflict_reason = (
            f"Potential directional conflict detected in this support evidence (contradiction={float(contradiction):.3f})."
            if conflict_alert
            else ""
        )
        return self._build_evidence_ref(
            passage=passage,
            excerpt=str(passage.get("text", ""))[:220],
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            conflict_alert=conflict_alert,
            conflict_reason=conflict_reason,
            relation="support",
        )

    def _contradiction_evidence_ref(self, item: tuple) -> EvidenceRef:
        (
            contradiction,
            semantic,
            lexical,
            section_prior,
            kind_prior,
            quality,
            passage,
        ) = item
        confidence_score = self._contradiction_confidence_score(
            contradiction=float(contradiction),
            semantic=float(semantic),
            lexical=float(lexical),
            section_prior=float(section_prior),
            kind_prior=float(kind_prior),
            quality=float(quality),
        )
        confidence_level = self._confidence_level(confidence_score, strong=0.7, medium=0.48)
        return self._build_evidence_ref(
            passage=passage,
            excerpt=str(passage.get("text", ""))[:220],
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            conflict_alert=True,
            conflict_reason=(
                f"Contradiction evidence against claim direction (contradiction={float(contradiction):.3f})."
            ),
            relation="contradiction",
        )

    @staticmethod
    def _support_confidence_score(
        *,
        support_score: float,
        semantic: float,
        lexical: float,
        section_prior: float,
        kind_prior: float,
        quality: float,
        contradiction: float,
    ) -> float:
        raw = (
            0.52 * support_score
            + 0.16 * quality
            + 0.12 * max(semantic, lexical)
            + 0.1 * section_prior
            + 0.1 * kind_prior
        )
        raw *= 1.0 - 0.35 * max(0.0, min(1.0, contradiction))
        return round(ClaimEvidenceAlignerStep._clip01(raw), 3)

    @staticmethod
    def _contradiction_confidence_score(
        *,
        contradiction: float,
        semantic: float,
        lexical: float,
        section_prior: float,
        kind_prior: float,
        quality: float,
    ) -> float:
        raw = (
            0.6 * contradiction
            + 0.14 * quality
            + 0.12 * max(semantic, lexical)
            + 0.08 * section_prior
            + 0.06 * kind_prior
        )
        return round(ClaimEvidenceAlignerStep._clip01(raw), 3)

    @staticmethod
    def _confidence_level(score: float, *, strong: float, medium: float) -> str:
        if score >= strong:
            return "Strong"
        if score >= medium:
            return "Medium"
        return "Weak"

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _build_evidence_ref(
        *,
        passage: dict,
        excerpt: str,
        confidence_level: str,
        confidence_score: float,
        conflict_alert: bool,
        conflict_reason: str,
        relation: str,
    ) -> EvidenceRef:
        locator = passage.get("locator", {})
        if not isinstance(locator, dict):
            locator = {}
        return EvidenceRef(
            section=str(passage.get("section", "unknown")),
            passage_id=str(passage.get("id", "unknown")),
            excerpt=str(excerpt or "")[:220],
            section_id=str(passage.get("section_id", "") or ""),
            section_index=int(passage.get("section_index", 0) or 0),
            page=int(passage.get("page", 0) or 0),
            kind=str(passage.get("kind", "") or ""),
            anchor_label=str(passage.get("anchor_label", "") or ""),
            anchor_type=str(passage.get("anchor_type", "") or ""),
            locator=locator,
            confidence_level=confidence_level,
            confidence_score=round(ClaimEvidenceAlignerStep._clip01(confidence_score), 3),
            conflict_alert=bool(conflict_alert),
            conflict_reason=str(conflict_reason or ""),
            relation=str(relation or "support"),
        )

    @staticmethod
    def _evidence_confidence_summary(
        support_refs: list[EvidenceRef], contradiction_refs: list[EvidenceRef]
    ) -> dict:
        def _count_levels(rows: list[EvidenceRef]) -> dict[str, int]:
            levels = {"Strong": 0, "Medium": 0, "Weak": 0}
            for ref in rows:
                lv = str(ref.confidence_level or "Weak")
                if lv not in levels:
                    lv = "Weak"
                levels[lv] += 1
            return levels

        support_counts = _count_levels(support_refs)
        contradiction_counts = _count_levels(contradiction_refs)
        support_scores = [float(ref.confidence_score or 0.0) for ref in support_refs]
        contradiction_scores = [float(ref.confidence_score or 0.0) for ref in contradiction_refs]
        return {
            "support": {
                "counts": support_counts,
                "max_confidence_score": round(max(support_scores), 3) if support_scores else 0.0,
                "min_confidence_score": round(min(support_scores), 3) if support_scores else 0.0,
            },
            "contradiction": {
                "counts": contradiction_counts,
                "max_confidence_score": round(max(contradiction_scores), 3) if contradiction_scores else 0.0,
                "min_confidence_score": round(min(contradiction_scores), 3) if contradiction_scores else 0.0,
            },
            "conflict_alert": bool(contradiction_refs),
        }
