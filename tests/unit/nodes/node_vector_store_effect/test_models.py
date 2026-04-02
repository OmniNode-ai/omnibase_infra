# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for vector store effect models."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_vector_store_effect.models import (
    ModelVectorStoreRequest,
    ModelVectorStoreResult,
)
from omnibase_infra.nodes.node_vector_store_effect.models.model_vector_document import (
    ModelVectorDocument,
)
from omnibase_infra.nodes.node_vector_store_effect.models.model_vector_search_hit import (
    ModelVectorSearchHit,
)


@pytest.mark.unit
class TestModelVectorDocument:
    def test_frozen(self) -> None:
        doc = ModelVectorDocument(text="hello")
        with pytest.raises(Exception):
            doc.text = "changed"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        with pytest.raises(Exception):
            ModelVectorDocument(text="hello", bogus="nope")  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        doc = ModelVectorDocument(text="hello")
        assert doc.metadata == {}
        assert doc.doc_id == ""


@pytest.mark.unit
class TestModelVectorStoreRequest:
    def test_upsert_request(self) -> None:
        req = ModelVectorStoreRequest(
            operation="upsert",
            documents=(ModelVectorDocument(text="doc1"),),
        )
        assert req.operation == "upsert"
        assert len(req.documents) == 1
        assert req.collection_name == "vectors"

    def test_search_request(self) -> None:
        req = ModelVectorStoreRequest(
            operation="search",
            query="find me",
            limit=5,
        )
        assert req.operation == "search"
        assert req.query == "find me"
        assert req.limit == 5

    def test_frozen(self) -> None:
        req = ModelVectorStoreRequest(operation="search", query="test")
        with pytest.raises(Exception):
            req.query = "changed"  # type: ignore[misc]

    def test_limit_bounds(self) -> None:
        with pytest.raises(Exception):
            ModelVectorStoreRequest(operation="search", query="test", limit=0)
        with pytest.raises(Exception):
            ModelVectorStoreRequest(operation="search", query="test", limit=200)


@pytest.mark.unit
class TestModelVectorSearchHit:
    def test_frozen(self) -> None:
        hit = ModelVectorSearchHit(doc_id="abc", score=0.95)
        with pytest.raises(Exception):
            hit.score = 0.5  # type: ignore[misc]


@pytest.mark.unit
class TestModelVectorStoreResult:
    def test_upsert_result(self) -> None:
        result = ModelVectorStoreResult(
            correlation_id=uuid4(),
            success=True,
            operation="upsert",
            upserted_ids=("id1", "id2"),
        )
        assert result.success is True
        assert len(result.upserted_ids) == 2

    def test_search_result(self) -> None:
        hit = ModelVectorSearchHit(doc_id="abc", score=0.95, text="found it")
        result = ModelVectorStoreResult(
            correlation_id=uuid4(),
            success=True,
            operation="search",
            hits=(hit,),
        )
        assert len(result.hits) == 1
        assert result.hits[0].text == "found it"

    def test_failure(self) -> None:
        result = ModelVectorStoreResult(
            correlation_id=uuid4(),
            success=False,
            error_message="connection refused",
        )
        assert result.success is False
