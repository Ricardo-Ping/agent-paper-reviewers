from __future__ import annotations

from ..services.citation_graph import build_citation_graph
from .base import PipelineContext, PipelineStep


class CitationGraphStep(PipelineStep):
    name = "CitationGraph"

    def run(self, ctx: PipelineContext) -> None:
        structured = ctx.artifacts["paper_structured"]
        graph = build_citation_graph(structured)

        for warning in graph.get("warnings", []):
            if self._should_emit_warning(graph, warning):
                ctx.add_qa_issue(f"citation_graph_warning:{warning}")

        ctx.artifacts["citation_graph"] = graph
        ctx.dump_json("artifacts/citation_graph.json", graph)

    @staticmethod
    def _should_emit_warning(graph: dict, warning: str) -> bool:
        source = str(graph.get("source") or "")
        outgoing_count = int(graph.get("stats", {}).get("outgoing_count", 0) or 0)

        # Soft-recoverable remote lookup warnings should not fail QA when we still
        # have usable local or hybrid citation evidence.
        soft_remote_warnings = (
            "semantic_scholar_status_429",
            "semantic_scholar_rate_limited",
            "semantic_scholar_no_search_result",
            "semantic_scholar_request_failed",
            "semantic_scholar_forbidden_check_api_key_or_quota",
            "semantic_scholar_status_503",
        )
        if warning.startswith(soft_remote_warnings) and source in {"local_only", "hybrid"} and outgoing_count > 0:
            return False
        return True
