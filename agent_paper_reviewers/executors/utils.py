from __future__ import annotations

import json
import re
from typing import Any

from ..models import TaskSpec


def build_llm_prompt(spec: TaskSpec) -> str:
    schema = json.dumps(spec.output_schema, ensure_ascii=False, indent=2)
    context = json.dumps(spec.context, ensure_ascii=False, indent=2)
    return (
        f"{spec.prompt}\n\n"
        f"[task_type]\n{spec.task_type}\n\n"
        f"[context]\n{context}\n\n"
        f"[output_schema]\n{schema}\n\n"
        "You must return a JSON object only. No markdown fence."
    )


def _extract_json_blob(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    obj = re.search(r"(\{[\s\S]*\})", text)
    if obj:
        try:
            parsed = json.loads(obj.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def normalize_output(spec: TaskSpec, text: str, raw: Any = None) -> dict[str, Any]:
    parsed = _extract_json_blob(text)
    if parsed is not None:
        return parsed

    stripped = text.strip()
    if spec.task_type == "translate_zh":
        return {"translated_text": stripped}
    if spec.task_type == "summarize":
        return {"summary": stripped}
    if spec.task_type == "score_similarity":
        try:
            return {"score": float(stripped)}
        except ValueError:
            return {"score": 0.0}

    out: dict[str, Any] = {"text": stripped}
    if raw is not None:
        out["raw"] = raw
    return out

