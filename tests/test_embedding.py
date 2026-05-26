"""Tests for hermes_memory.embedding — Aliyun text-embedding-v4 client."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from hermes_memory.embedding import EmbeddingClient


class TestEmbeddingClient:
    """Embedding client — HTTP mocking for tests."""

    @pytest.mark.asyncio
    async def test_get_embedding_returns_1024d_vector(self):
        """get_embedding should return a list of 1024 floats."""
        mock_response = {
            "output": {"embeddings": [{"embedding": [0.1] * 1024}]},
        }
        client = EmbeddingClient(api_key="test-key", model="text-embedding-v4")
        with patch.object(client, "_client") as mock_http:
            mock_http.post = AsyncMock(return_value=MockResponse(200, mock_response))
            result = await client.get_embedding("Hello world")
        assert len(result) == 1024
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_get_embedding_caches_result(self):
        """Same text should return cached embedding without API call."""
        mock_response = {
            "output": {"embeddings": [{"embedding": [0.5] * 1024}]},
        }
        client = EmbeddingClient(api_key="test-key")
        with patch.object(client, "_client") as mock_http:
            mock_http.post = AsyncMock(return_value=MockResponse(200, mock_response))
            first = await client.get_embedding("cache test")
            second = await client.get_embedding("cache test")
        assert first == second
        assert mock_http.post.call_count == 1  # only one API call

    @pytest.mark.asyncio
    async def test_get_embedding_raises_on_api_error(self):
        """Non-200 response should raise RuntimeError."""
        client = EmbeddingClient(api_key="bad-key")
        with patch.object(client, "_client") as mock_http:
            mock_http.post = AsyncMock(return_value=MockResponse(400, {"error": "bad request"}))
            with pytest.raises(RuntimeError, match="Embedding API error"):
                await client.get_embedding("test")

    @pytest.mark.asyncio
    async def test_get_embedding_empty_text_raises(self):
        """Empty text should raise ValueError."""
        client = EmbeddingClient(api_key="test-key")
        with pytest.raises(ValueError, match="empty"):
            await client.get_embedding("")

    @pytest.mark.asyncio
    async def test_batch_embedding(self):
        """batch_embedding should return list of vectors."""
        mock_response = {
            "output": {
                "embeddings": [
                    {"embedding": [0.1] * 1024},
                    {"embedding": [0.2] * 1024},
                ]
            },
        }
        client = EmbeddingClient(api_key="test-key")
        with patch.object(client, "_client") as mock_http:
            mock_http.post = AsyncMock(return_value=MockResponse(200, mock_response))
            results = await client.batch_embedding(["text a", "text b"])
        assert len(results) == 2
        assert len(results[0]) == 1024
        assert len(results[1]) == 1024

    @pytest.mark.asyncio
    async def test_cosine_similarity(self):
        """cosine_similarity should return correct value."""
        client = EmbeddingClient(api_key="test-key")
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        c = [1.0, 0.0, 0.0]
        assert client.cosine_similarity(a, a) == pytest.approx(1.0)
        assert client.cosine_similarity(a, b) == pytest.approx(0.0)
        assert client.cosine_similarity(a, c) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_cosine_similarity_empty_vector(self):
        """Zero vector should return 0.0."""
        client = EmbeddingClient(api_key="test-key")
        zero = [0.0, 0.0, 0.0]
        assert client.cosine_similarity(zero, [1.0, 0.0, 0.0]) == pytest.approx(0.0)


class MockResponse:
    """Minimal mock for httpx Response."""

    def __init__(self, status_code: int, json_data: dict):
        self.status_code = status_code
        self._json = json_data

    async def json(self):
        return self._json

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
