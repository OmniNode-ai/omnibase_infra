# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the decision embedding pipeline for Qdrant.

Tests ModelDecisionRecord creation, build_embedding_text formatting,
deterministic point ID generation, and DecisionEmbedder methods with
mocked Qdrant and embedding endpoints.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 12).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.services.session_registry.decision_embedder import (
    COLLECTION_NAME,
    VECTOR_DIMENSION,
    DecisionEmbedder,
    ModelDecisionRecord,
    _decision_point_id,
    build_embedding_text,
)


@pytest.fixture
def sample_record() -> ModelDecisionRecord:
    """Create a sample decision record for testing."""
    return ModelDecisionRecord(
        task_id="OMN-1234",
        session_id="sess-abc-123",
        decision_text="Use Kafka for async communication",
        context="Existing Redpanda infra supports this pattern",
        timestamp=datetime(2026, 3, 27, 12, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def sample_record_no_context() -> ModelDecisionRecord:
    """Create a sample decision record without context."""
    return ModelDecisionRecord(
        task_id="OMN-5678",
        session_id="sess-def-456",
        decision_text="Pin qdrant-client to v1.16+",
        context="",
        timestamp=datetime(2026, 3, 27, 14, 0, 0, tzinfo=UTC),
    )


@pytest.mark.unit
class TestModelDecisionRecord:
    """Tests for ModelDecisionRecord validation."""

    def test_valid_creation(self, sample_record: ModelDecisionRecord) -> None:
        assert sample_record.task_id == "OMN-1234"
        assert sample_record.session_id == "sess-abc-123"
        assert sample_record.decision_text == "Use Kafka for async communication"

    def test_frozen_immutability(self, sample_record: ModelDecisionRecord) -> None:
        with pytest.raises(Exception):  # ValidationError for frozen model
            sample_record.task_id = "OMN-9999"  # type: ignore[misc]

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(Exception):
            ModelDecisionRecord(
                task_id="OMN-1234",
                session_id="sess-abc",
                decision_text="test",
                timestamp=datetime.now(tz=UTC),
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_requires_task_id(self) -> None:
        with pytest.raises(Exception):
            ModelDecisionRecord(
                task_id="",
                session_id="sess-abc",
                decision_text="test",
                timestamp=datetime.now(tz=UTC),
            )

    def test_requires_decision_text(self) -> None:
        with pytest.raises(Exception):
            ModelDecisionRecord(
                task_id="OMN-1234",
                session_id="sess-abc",
                decision_text="",
                timestamp=datetime.now(tz=UTC),
            )


@pytest.mark.unit
class TestBuildEmbeddingText:
    """Tests for build_embedding_text formatting."""

    def test_with_context(self, sample_record: ModelDecisionRecord) -> None:
        text = build_embedding_text(sample_record)
        assert text == (
            "Task OMN-1234: Use Kafka for async communication. "
            "Context: Existing Redpanda infra supports this pattern"
        )

    def test_without_context(
        self, sample_record_no_context: ModelDecisionRecord
    ) -> None:
        text = build_embedding_text(sample_record_no_context)
        assert text == "Task OMN-5678: Pin qdrant-client to v1.16+"
        assert "Context:" not in text

    def test_preserves_task_id_prefix(self, sample_record: ModelDecisionRecord) -> None:
        text = build_embedding_text(sample_record)
        assert text.startswith("Task OMN-1234:")


@pytest.mark.unit
class TestDecisionPointId:
    """Tests for deterministic point ID generation."""

    def test_deterministic(self) -> None:
        id1 = _decision_point_id("OMN-1234", "Use Kafka")
        id2 = _decision_point_id("OMN-1234", "Use Kafka")
        assert id1 == id2

    def test_different_for_different_text(self) -> None:
        id1 = _decision_point_id("OMN-1234", "Use Kafka")
        id2 = _decision_point_id("OMN-1234", "Use RabbitMQ")
        assert id1 != id2

    def test_different_for_different_task(self) -> None:
        id1 = _decision_point_id("OMN-1234", "Use Kafka")
        id2 = _decision_point_id("OMN-5678", "Use Kafka")
        assert id1 != id2

    def test_returns_valid_uuid_string(self) -> None:
        point_id = _decision_point_id("OMN-1234", "test")
        # uuid5 returns a UUID with dashes
        parts = point_id.split("-")
        assert len(parts) == 5


@pytest.mark.unit
class TestDecisionEmbedderEnsureCollection:
    """Tests for DecisionEmbedder.ensure_collection."""

    @pytest.mark.asyncio
    async def test_creates_collection_when_missing(self) -> None:
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value = MagicMock(collections=[])

        with patch(
            "omnibase_infra.services.session_registry.decision_embedder.QdrantClient",
            return_value=mock_qdrant,
        ):
            embedder = DecisionEmbedder(qdrant_url="http://localhost:6333")
            await embedder.ensure_collection()

        mock_qdrant.create_collection.assert_called_once()
        call_kwargs = mock_qdrant.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == COLLECTION_NAME

    @pytest.mark.asyncio
    async def test_skips_creation_when_exists(self) -> None:
        existing = MagicMock()
        existing.name = COLLECTION_NAME
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value = MagicMock(collections=[existing])

        with patch(
            "omnibase_infra.services.session_registry.decision_embedder.QdrantClient",
            return_value=mock_qdrant,
        ):
            embedder = DecisionEmbedder(qdrant_url="http://localhost:6333")
            await embedder.ensure_collection()

        mock_qdrant.create_collection.assert_not_called()


@pytest.mark.unit
class TestDecisionEmbedderEmbedAndUpsert:
    """Tests for DecisionEmbedder.embed_and_upsert."""

    @pytest.mark.asyncio
    async def test_upserts_to_qdrant(self, sample_record: ModelDecisionRecord) -> None:
        mock_qdrant = MagicMock()
        fake_vector = [0.1] * VECTOR_DIMENSION

        with (
            patch(
                "omnibase_infra.services.session_registry.decision_embedder.QdrantClient",
                return_value=mock_qdrant,
            ),
            patch.object(
                DecisionEmbedder,
                "_get_embedding",
                new_callable=AsyncMock,
                return_value=fake_vector,
            ),
        ):
            embedder = DecisionEmbedder(qdrant_url="http://localhost:6333")
            point_id = await embedder.embed_and_upsert(sample_record)

        assert point_id is not None
        mock_qdrant.upsert.assert_called_once()
        call_kwargs = mock_qdrant.upsert.call_args
        assert call_kwargs.kwargs["collection_name"] == COLLECTION_NAME

    @pytest.mark.asyncio
    async def test_idempotent_point_id(
        self, sample_record: ModelDecisionRecord
    ) -> None:
        mock_qdrant = MagicMock()
        fake_vector = [0.1] * VECTOR_DIMENSION

        with (
            patch(
                "omnibase_infra.services.session_registry.decision_embedder.QdrantClient",
                return_value=mock_qdrant,
            ),
            patch.object(
                DecisionEmbedder,
                "_get_embedding",
                new_callable=AsyncMock,
                return_value=fake_vector,
            ),
        ):
            embedder = DecisionEmbedder(qdrant_url="http://localhost:6333")
            id1 = await embedder.embed_and_upsert(sample_record)
            id2 = await embedder.embed_and_upsert(sample_record)

        assert id1 == id2

    @pytest.mark.asyncio
    async def test_payload_contains_all_fields(
        self, sample_record: ModelDecisionRecord
    ) -> None:
        mock_qdrant = MagicMock()
        fake_vector = [0.1] * VECTOR_DIMENSION

        with (
            patch(
                "omnibase_infra.services.session_registry.decision_embedder.QdrantClient",
                return_value=mock_qdrant,
            ),
            patch.object(
                DecisionEmbedder,
                "_get_embedding",
                new_callable=AsyncMock,
                return_value=fake_vector,
            ),
        ):
            embedder = DecisionEmbedder(qdrant_url="http://localhost:6333")
            await embedder.embed_and_upsert(sample_record)

        call_kwargs = mock_qdrant.upsert.call_args
        points = call_kwargs.kwargs["points"]
        payload = points[0].payload
        assert payload["task_id"] == "OMN-1234"
        assert payload["session_id"] == "sess-abc-123"
        assert payload["decision_text"] == "Use Kafka for async communication"
        assert payload["context"] == "Existing Redpanda infra supports this pattern"
        assert "timestamp" in payload
        assert "embedding_text" in payload


@pytest.mark.unit
class TestDecisionEmbedderSearchSimilar:
    """Tests for DecisionEmbedder.search_similar."""

    @pytest.mark.asyncio
    async def test_search_without_filter(self) -> None:
        mock_qdrant = MagicMock()
        mock_qdrant.query_points.return_value = MagicMock(points=[])
        fake_vector = [0.1] * VECTOR_DIMENSION

        with (
            patch(
                "omnibase_infra.services.session_registry.decision_embedder.QdrantClient",
                return_value=mock_qdrant,
            ),
            patch.object(
                DecisionEmbedder,
                "_get_embedding",
                new_callable=AsyncMock,
                return_value=fake_vector,
            ),
        ):
            embedder = DecisionEmbedder(qdrant_url="http://localhost:6333")
            results = await embedder.search_similar("Kafka usage")

        assert results == []
        mock_qdrant.query_points.assert_called_once()
        call_kwargs = mock_qdrant.query_points.call_args
        assert call_kwargs.kwargs["query_filter"] is None
        assert call_kwargs.kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_search_with_task_filter(self) -> None:
        mock_qdrant = MagicMock()
        mock_qdrant.query_points.return_value = MagicMock(points=[])
        fake_vector = [0.1] * VECTOR_DIMENSION

        with (
            patch(
                "omnibase_infra.services.session_registry.decision_embedder.QdrantClient",
                return_value=mock_qdrant,
            ),
            patch.object(
                DecisionEmbedder,
                "_get_embedding",
                new_callable=AsyncMock,
                return_value=fake_vector,
            ),
        ):
            embedder = DecisionEmbedder(qdrant_url="http://localhost:6333")
            await embedder.search_similar("Kafka", task_id_filter="OMN-1234")

        call_kwargs = mock_qdrant.query_points.call_args
        assert call_kwargs.kwargs["query_filter"] is not None
