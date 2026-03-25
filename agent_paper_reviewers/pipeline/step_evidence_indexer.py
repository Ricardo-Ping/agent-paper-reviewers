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
            section_text = str(sec.get("text") or "")
            for para_idx, paragraph in enumerate(self._split_paragraphs(section_text)):
                node_id = f"sec{sec_idx}_para{para_idx}"
                clean = paragraph.strip()
                if not clean:
                    continue
                nodes.append(
                    {
                        "id": node_id,
                        "section": section_name,
                        "kind": "paragraph",
                        "text": clean[:5000],
                        "page": self._estimate_page(structured, clean),
                    }
                )

                mentions = self._extract_figure_table_mentions(clean)
                for mt_idx, mention in enumerate(mentions):
                    nodes.append(
                        {
                            "id": f"{node_id}_mt{mt_idx}",
                            "section": section_name,
                            "kind": "figure_table_mention",
                            "text": mention[:5000],
                            "page": self._estimate_page(structured, mention),
                        }
                    )

        if not nodes:
            # Never leave downstream steps without evidence candidates.
            nodes.append(
                {
                    "id": "sec0_para0",
                    "section": "body",
                    "kind": "paragraph",
                    "text": structured.get("raw_text", "")[:5000],
                    "page": 1,
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
                "section": node["section"],
                "text": node["text"],
                "kind": node["kind"],
                "page": node["page"],
                "embedding_dim": node["embedding_dim"],
            }
            for node in nodes
        ]

        payload = {
            "passages": passages,
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
