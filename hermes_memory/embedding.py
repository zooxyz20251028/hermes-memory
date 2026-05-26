"""Aliyun text-embedding-v4 client.

Async HTTP client with local caching.
Uses httpx for HTTP (ECC Python patterns).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import httpx
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class EmbeddingClient:
    """Client for Aliyun (Bailian) text-embedding-v4 API.

    Args:
        api_key: Aliyun API key.
        model: Model name (default: text-embedding-v4).
        base_url: API endpoint.
        max_retries: Max retries on failure.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._max_retries = max_retries
        self._cache: dict[str, list[float]] = {}
        self._client = httpx.AsyncClient(timeout=30.0)

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for a single text.

        Args:
            text: Input text (must not be empty).

        Returns:
            1024-dim float vector.

        Raises:
            ValueError: If text is empty.
            RuntimeError: If API call fails.
        """
        if not text.strip():
            raise ValueError("Input text must not be empty")

        text_stripped = text.strip()
        if text_stripped in self._cache:
            return self._cache[text_stripped]

        vectors = await self._call_api([text_stripped])
        result = vectors[0]
        self._cache[text_stripped] = result
        return result

    async def batch_embedding(self, texts: Sequence[str]) -> list[list[float]]:
        """Get embedding vectors for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of 1024-dim float vectors.
        """
        uncached: list[str] = []
        uncached_indices: list[int] = []
        results: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]

        for i, t in enumerate(texts):
            ts = t.strip()
            if ts in self._cache:
                results[i] = self._cache[ts]
            else:
                uncached.append(ts)
                uncached_indices.append(i)

        if uncached:
            vectors = await self._call_api(uncached)
            for idx, vec in zip(uncached_indices, vectors, strict=True):
                self._cache[texts[idx].strip()] = vec
                results[idx] = vec

        return results  # type: ignore[return-value]

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the Aliyun embedding API.

        Args:
            texts: Non-empty list of input texts.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: If the API returns an error.
        """
        payload = {
            "model": self._model,
            "input": {"texts": texts},
            "parameters": {"text_type": "query"},
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.post(
                    self._base_url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = await response.json()
                embeds = data["output"]["embeddings"]
                return [e["embedding"] for e in embeds]
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    import asyncio

                    await asyncio.sleep(2**attempt)

        raise RuntimeError(f"Embedding API error after {self._max_retries} retries") from last_error

    def cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in [0, 1]. Returns 0.0 for zero vectors.
        """
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        dot = float(np.dot(a_arr, b_arr))
        norm_a = float(np.linalg.norm(a_arr))
        norm_b = float(np.linalg.norm(b_arr))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def clear_cache(self) -> None:
        """Clear the local embedding cache."""
        self._cache.clear()
