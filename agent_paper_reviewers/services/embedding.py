from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import re
from typing import Iterable

_SENTENCE_TRANSFORMER = None
_SENTENCE_TRANSFORMER_ERROR: str | None = None


def _get_sentence_transformer():
    global _SENTENCE_TRANSFORMER, _SENTENCE_TRANSFORMER_ERROR
    if _SENTENCE_TRANSFORMER is not None:
        return _SENTENCE_TRANSFORMER
    if _SENTENCE_TRANSFORMER_ERROR is not None:
        return None

    if importlib.util.find_spec("torch") is None:
        _SENTENCE_TRANSFORMER_ERROR = "torch_not_installed_use_hash_embedding"
        return None

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model_name = os.getenv(
            "AGENT_PAPER_REVIEWERS_EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        _SENTENCE_TRANSFORMER = SentenceTransformer(model_name)
        return _SENTENCE_TRANSFORMER
    except Exception as exc:  # noqa: BLE001
        _SENTENCE_TRANSFORMER_ERROR = str(exc)
        return None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _hash_embedding(text: str, dim: int = 384) -> list[float]:
    if not text.strip():
        return [0.0] * dim

    vec = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], byteorder="big", signed=False) % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return vec
    return [v / norm for v in vec]


def encode_texts(texts: Iterable[str]) -> tuple[list[list[float]], str]:
    text_list = list(texts)
    if not text_list:
        return [], "hash_embedding"

    model = _get_sentence_transformer()
    if model is not None:
        try:
            vectors = model.encode(text_list, normalize_embeddings=True)
            return [list(map(float, row)) for row in vectors], "sentence_transformers"
        except Exception:  # noqa: BLE001
            pass

    return [_hash_embedding(t) for t in text_list], "hash_embedding"


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)
