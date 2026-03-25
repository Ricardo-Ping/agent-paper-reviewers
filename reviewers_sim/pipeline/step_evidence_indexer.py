from __future__ import annotations

from ..services.paper_parser import split_passages
from .base import PipelineContext, PipelineStep


class EvidenceIndexerStep(PipelineStep):
    name = "EvidenceIndexer"

    def run(self, ctx: PipelineContext) -> None:
        structured = ctx.artifacts["paper_structured"]
        passages = split_passages(structured)
        payload = {
            "passages": passages,
            "passage_count": len(passages),
            "index_backend": "in_memory_token_overlap",
        }
        ctx.artifacts["evidence_index"] = payload
        ctx.dump_json("artifacts/evidence_index.json", payload)
