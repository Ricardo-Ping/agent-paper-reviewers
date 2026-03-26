from __future__ import annotations

import re

from ..services.embedding import encode_texts
from .base import PipelineContext, PipelineStep


class EvidenceIndexerStep(PipelineStep):
    name = "EvidenceIndexer"

    def run(self, ctx: PipelineContext) -> None:
        structured = ctx.artifacts["paper_structured"]
        sections = structured.get("sections", [])

        nodes: list[dict] = []
        for sec_idx, sec in enumerate(sections):
            section_name = str(sec.get("name") or f"section_{sec_idx}").strip().lower()
            section_id = str(sec.get("section_id") or f"S{sec_idx + 1:03d}").strip()
            section_index = int(sec.get("section_index") or (sec_idx + 1))
            section_text = str(sec.get("text") or "")
            for para_idx, paragraph in enumerate(self._split_paragraphs(section_text)):
                node_id = f"{section_id}_para{para_idx}"
                clean = paragraph.strip()
                if not clean:
                    continue
                quality_score, quality_flags, is_noisy = self._quality_profile(
                    clean,
                    section=section_name,
                    kind="paragraph",
                )
                if is_noisy:
                    continue
                nodes.append(
                    {
                        "id": node_id,
                        "section_id": section_id,
                        "section_index": section_index,
                        "section": section_name,
                        "kind": "paragraph",
                        "text": clean[:5000],
                        "page": self._estimate_page(structured, clean),
                        "quality_score": quality_score,
                        "quality_flags": quality_flags,
                    }
                )

                mentions = self._extract_figure_table_mentions(clean)
                for mt_idx, mention in enumerate(mentions):
                    mt_score, mt_flags, mt_noisy = self._quality_profile(
                        mention,
                        section=section_name,
                        kind="figure_table_mention",
                    )
                    if mt_noisy:
                        continue
                    nodes.append(
                        {
                            "id": f"{node_id}_mt{mt_idx}",
                            "section_id": section_id,
                            "section_index": section_index,
                            "section": section_name,
                            "kind": "figure_table_mention",
                            "text": mention[:5000],
                            "page": self._estimate_page(structured, mention),
                            "quality_score": mt_score,
                            "quality_flags": mt_flags,
                        }
                    )

        nodes.extend(self._extract_page_visual_content_nodes(structured))
        nodes = self._dedupe_nodes(nodes)

        if not nodes:
            # Never leave downstream steps without evidence candidates.
            fallback_text = structured.get("raw_text", "")[:5000]
            q_score, q_flags, _ = self._quality_profile(
                fallback_text,
                section="body",
                kind="paragraph",
            )
            nodes.append(
                {
                    "id": "S001_para0",
                    "section_id": "S001",
                    "section_index": 1,
                    "section": "body",
                    "kind": "paragraph",
                    "text": fallback_text,
                    "page": 1,
                    "quality_score": q_score,
                    "quality_flags": q_flags,
                }
            )

        vectors, embed_backend = encode_texts(node["text"] for node in nodes)
        evidence_vectors: dict[str, list[float]] = {}
        for node, vec in zip(nodes, vectors):
            evidence_vectors[node["id"]] = vec
            node["embedding_dim"] = len(vec)

        passages = [
            {
                "id": node["id"],
                "section_id": node.get("section_id", ""),
                "section_index": node.get("section_index", 0),
                "section": node["section"],
                "text": node["text"],
                "kind": node["kind"],
                "page": node["page"],
                "embedding_dim": node["embedding_dim"],
                "quality_score": node.get("quality_score", 0.5),
                "quality_flags": node.get("quality_flags", []),
            }
            for node in nodes
        ]
        passage_locator = {
            str(p["id"]): {
                "section_id": p.get("section_id", ""),
                "section_index": p.get("section_index", 0),
                "section": p.get("section", ""),
                "page": p.get("page", 1),
                "kind": p.get("kind", ""),
            }
            for p in passages
        }

        payload = {
            "passages": passages,
            "passage_locator": passage_locator,
            "nodes": nodes,
            "passage_count": len(passages),
            "index_backend": "in_memory_semantic_vector",
            "embedding_backend": embed_backend,
            "embedding_dim": len(next(iter(evidence_vectors.values()))) if evidence_vectors else 0,
        }

        ctx.artifacts["evidence_index"] = payload
        ctx.artifacts["evidence_vectors"] = evidence_vectors
        ctx.dump_json("artifacts/evidence_index.json", payload)

    @staticmethod
    def _dedupe_nodes(nodes: list[dict]) -> list[dict]:
        out: list[dict] = []
        seen: set[tuple[str, str, int, str]] = set()
        for node in nodes:
            key = (
                str(node.get("kind", "")),
                str(node.get("section", "")),
                int(node.get("page", 1) or 1),
                re.sub(r"\s+", " ", str(node.get("text", "")).strip().lower())[:320],
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(node)
        return out

    @staticmethod
    def _quality_profile(text: str, *, section: str, kind: str) -> tuple[float, list[str], bool]:
        tokens = [t for t in re.split(r"\s+", text.strip()) if t]
        total_tokens = len(tokens)
        if total_tokens == 0:
            return 0.0, ["empty"], True

        alpha_tokens = sum(1 for t in tokens if re.search(r"[A-Za-z]", t))
        numeric_tokens = sum(1 for t in tokens if re.fullmatch(r"[-+]?[\d\.,%]+", t) is not None)
        alpha_ratio = alpha_tokens / max(1, total_tokens)
        numeric_ratio = numeric_tokens / max(1, total_tokens)

        line_break_like = text.count("|") + text.count("\t")
        long_number_runs = len(re.findall(r"(?:\d+\s+){4,}\d+", text))

        flags: list[str] = []
        score = 1.0
        if alpha_ratio < 0.35:
            flags.append("low_alpha_ratio")
            score -= 0.25
        if numeric_ratio > 0.35:
            flags.append("high_numeric_ratio")
            score -= 0.25
        if line_break_like >= 3:
            flags.append("table_like_layout")
            score -= 0.15
        if long_number_runs > 0:
            flags.append("number_run_pattern")
            score -= 0.2
        if len(text) < 35:
            flags.append("too_short")
            score -= 0.1
        if section in {"experiments", "results", "ablation", "analysis"}:
            score += 0.08
        if kind in {"table_data", "figure_content", "figure_table_mention"}:
            score += 0.05

        score = max(0.0, min(1.0, round(score, 3)))

        # Strong noise rule for paragraphs/mentions (keep table_data nodes for numeric evidence).
        is_noisy = (
            kind in {"paragraph", "figure_table_mention"}
            and (
                (alpha_tokens < 4 and numeric_tokens >= 4)
                or (numeric_ratio > 0.55 and alpha_ratio < 0.28)
                or ("number_run_pattern" in flags and alpha_ratio < 0.4)
            )
        )
        return score, flags, is_noisy

    @staticmethod
    def _split_paragraphs(section_text: str) -> list[str]:
        blocks = re.split(r"\n\s*\n+", section_text)
        paragraphs: list[str] = []
        for block in blocks:
            clean = re.sub(r"\s+", " ", block).strip()
            if not clean:
                continue
            if len(clean) <= 1200:
                paragraphs.append(clean)
                continue

            # Chunk overly long paragraphs into sentence windows for retrievability.
            sentences = re.split(r"(?<=[.!?])\s+", clean)
            chunk: list[str] = []
            chunk_len = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if chunk_len + len(sentence) > 900 and chunk:
                    paragraphs.append(" ".join(chunk).strip())
                    chunk = [sentence]
                    chunk_len = len(sentence)
                else:
                    chunk.append(sentence)
                    chunk_len += len(sentence)
            if chunk:
                paragraphs.append(" ".join(chunk).strip())

        return paragraphs

    @staticmethod
    def _extract_figure_table_mentions(text: str) -> list[str]:
        mentions: list[str] = []
        patterns = [
            r"(table\s+\d+[\w\-\.]*)",
            r"(tab\.\s*\d+[\w\-\.]*)",
            r"(figure\s+\d+[\w\-\.]*)",
            r"(fig\.\s*\d+[\w\-\.]*)",
        ]
        lowered = text.lower()
        for pattern in patterns:
            for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
                span_start = max(0, match.start() - 80)
                span_end = min(len(text), match.end() + 180)
                snippet = text[span_start:span_end].strip()
                if snippet and snippet not in mentions:
                    mentions.append(snippet)
        return mentions[:3]

    def _extract_page_visual_content_nodes(self, structured: dict) -> list[dict]:
        pages = structured.get("pages") or []
        if not isinstance(pages, list):
            return []

        nodes: list[dict] = []
        for page in pages:
            page_no = self._coerce_page(page.get("page"))
            page_text = str(page.get("text") or "")
            if page_text.strip():
                nodes.extend(self._extract_caption_and_context_nodes(page_no, page_text))
            tables = page.get("tables") or []
            if isinstance(tables, list) and tables:
                nodes.extend(self._extract_table_data_nodes(page_no, tables))
        return nodes

    def _extract_caption_and_context_nodes(self, page_no: int, page_text: str) -> list[dict]:
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        if not lines:
            return []

        nodes: list[dict] = []
        caption_idx = 0
        for i, line in enumerate(lines):
            m = re.match(
                r"^(figure|fig\.?|table|tab\.?)\s*(\d+[A-Za-z\-\.]*)\s*[:\.\-]?\s*(.*)$",
                line,
                flags=re.IGNORECASE,
            )
            if not m:
                continue

            label = m.group(1).lower()
            number = m.group(2)
            title_tail = m.group(3).strip()
            context_lines = []
            for j in range(i + 1, min(len(lines), i + 7)):
                nxt = lines[j]
                if re.match(r"^(figure|fig\.?|table|tab\.?)\s*\d+", nxt, flags=re.IGNORECASE):
                    break
                if len(nxt.split()) <= 2:
                    continue
                context_lines.append(nxt)

            block_text = f"{line} {' '.join(context_lines[:4])}".strip()
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?%?", block_text)
            if numbers:
                block_text += f" Key numbers: {', '.join(numbers[:12])}"

            kind = "figure_content" if label.startswith("fig") else "table_content"
            anchor = f"{'fig' if label.startswith('fig') else 'table'}_{number}"
            caption_idx += 1
            q_score, q_flags, _ = self._quality_profile(
                block_text,
                section="figures_tables",
                kind=kind,
            )
            nodes.append(
                {
                    "id": f"p{page_no}_{anchor}_cap{caption_idx}",
                    "section_id": f"P{page_no:03d}",
                    "section_index": 0,
                    "section": "figures_tables",
                    "kind": kind,
                    "text": block_text[:5000],
                    "page": page_no,
                    "quality_score": q_score,
                    "quality_flags": q_flags,
                }
            )

            if title_tail:
                t_score, t_flags, _ = self._quality_profile(
                    title_tail,
                    section="figures_tables",
                    kind=f"{kind}_title",
                )
                nodes.append(
                    {
                        "id": f"p{page_no}_{anchor}_title{caption_idx}",
                        "section_id": f"P{page_no:03d}",
                        "section_index": 0,
                        "section": "figures_tables",
                        "kind": f"{kind}_title",
                        "text": title_tail[:5000],
                        "page": page_no,
                        "quality_score": t_score,
                        "quality_flags": t_flags,
                    }
                )
        return nodes

    def _extract_table_data_nodes(self, page_no: int, tables: list) -> list[dict]:
        nodes: list[dict] = []
        for t_idx, table in enumerate(tables, start=1):
            if not isinstance(table, list):
                continue

            row_strings: list[str] = []
            numeric_values: list[str] = []
            for row in table[:20]:
                if not isinstance(row, list):
                    continue
                cells = [re.sub(r"\s+", " ", str(cell or "").strip()) for cell in row]
                if not any(cells):
                    continue
                cleaned = [c for c in cells if c]
                if not cleaned:
                    continue
                row_strings.append(" | ".join(cleaned[:12]))
                for cell in cleaned[:12]:
                    numeric_values.extend(re.findall(r"[-+]?\d+(?:\.\d+)?%?", cell))

            if not row_strings:
                continue

            text = f"Table data p{page_no} t{t_idx}: " + " ; ".join(row_strings[:12])
            if numeric_values:
                text += f" Key numbers: {', '.join(numeric_values[:20])}"
            q_score, q_flags, _ = self._quality_profile(
                text,
                section="figures_tables",
                kind="table_data",
            )

            nodes.append(
                {
                    "id": f"p{page_no}_table_data_{t_idx}",
                    "section_id": f"P{page_no:03d}",
                    "section_index": 0,
                    "section": "figures_tables",
                    "kind": "table_data",
                    "text": text[:5000],
                    "page": page_no,
                    "quality_score": q_score,
                    "quality_flags": q_flags,
                }
            )
        return nodes

    @staticmethod
    def _coerce_page(value: object) -> int:
        try:
            return int(value)
        except Exception:  # noqa: BLE001
            return 1

    @staticmethod
    def _estimate_page(structured: dict, text: str) -> int:
        pages = structured.get("pages") or []
        if not pages:
            return 1
        query = text[:220].lower()
        for page in pages:
            page_text = str(page.get("text") or "").lower()
            if query and query in page_text:
                try:
                    return int(page.get("page", 1))
                except Exception:  # noqa: BLE001
                    return 1
        return 1
