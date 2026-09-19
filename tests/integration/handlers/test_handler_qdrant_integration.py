# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for HandlerQdrant against a real Qdrant server.

Selection in CI (OMN-18781). These tests do NOT skip gracefully in CI, because
a silent skip and a green run are indistinguishable in a junit summary:

    - the module carries ``pytest.mark.qdrant``, which the PR test splits
      deselect, so they are never collected-and-skipped on a pull request;
    - the service-backed job in ``.github/workflows/ci.yml`` starts a real
      Qdrant, exports ``QDRANT_URL`` and ``QDRANT_INTEGRATION_TESTS=1``, and
      runs them;
    - a CI job that selects them without that provisioning FAILS, naming the
      variable, rather than skipping.

Outside CI they skip, so a laptop with no Qdrant is not forced to stand one up.

``QDRANT_URL`` alone is never evidence that a server exists: ``tests/conftest.py``
supplies a localhost default for it so models can be instantiated offline.

Why this file was rewritten, not merely ungated. The suite these tests replace could not run at all, and had not been able to for
a long time. It called ``handler.execute(envelope)`` with hand-built envelopes
and treated ``describe()`` as synchronous. ``HandlerQdrant`` has neither: it
implements ``ProtocolVectorStoreHandler`` — ``store_embedding``,
``query_similar``, ``delete_embedding``, ``create_index``, ``health_check`` and
an async ``describe`` — and its constructor takes a ``ModelONEXContainer``.
The very first thing an ungated run produced was
``HandlerQdrant.__init__() missing 1 required positional argument: 'container'``.

That is the cost of a suite that skips: it keeps its shape while the thing it
claims to cover moves out from under it, and nothing says so. The coverage
below is written against the protocol the handler actually exposes today.

Run with infrastructure::

    $ QDRANT_URL=http://localhost:6333 QDRANT_INTEGRATION_TESTS=1 \\
        uv run pytest tests/integration/handlers/test_handler_qdrant_integration.py -v

Related tickets: OMN-1142 (original suite), OMN-18781 (this rewrite).
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums.enum_vector_distance_metric import EnumVectorDistanceMetric
from omnibase_core.enums.enum_vector_filter_operator import EnumVectorFilterOperator
from omnibase_core.models.common.model_schema_value import ModelSchemaValue
from omnibase_core.models.vector.model_embedding import ModelEmbedding
from omnibase_core.models.vector.model_vector_connection_config import (
    ModelVectorConnectionConfig,
)
from omnibase_core.models.vector.model_vector_metadata_filter import (
    ModelVectorMetadataFilter,
)
from omnibase_infra.errors import InfraConnectionError
from tests.helpers.service_env import require_service_env

if TYPE_CHECKING:
    from omnibase_infra.handlers import HandlerQdrant

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Four-dimensional vectors keep the assertions readable. The axis-aligned unit
# vectors below are mutually orthogonal, so cosine similarity separates them
# completely and a ranking assertion cannot pass by accident.
DIMENSION = 4
VECTOR_X: list[float] = [1.0, 0.0, 0.0, 0.0]
VECTOR_Y: list[float] = [0.0, 1.0, 0.0, 0.0]
VECTOR_Z: list[float] = [0.0, 0.0, 1.0, 0.0]

pytestmark = [
    pytest.mark.qdrant,
]


@pytest.fixture(autouse=True, scope="module")
def _require_qdrant() -> None:
    """Refuse to skip this suite silently once CI has selected it."""
    require_service_env(
        opt_in="QDRANT_INTEGRATION_TESTS",
        endpoint="QDRANT_URL",
        workflow=".github/workflows/ci.yml (service-integration-suites)",
        service="Qdrant",
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def connection_config() -> ModelVectorConnectionConfig:
    """Connection configuration pointing at the provisioned Qdrant."""
    return ModelVectorConnectionConfig(
        url=QDRANT_URL or "",
        api_key=QDRANT_API_KEY,
        timeout=30.0,
    )


@pytest.fixture
async def handler(
    connection_config: ModelVectorConnectionConfig,
) -> AsyncGenerator[HandlerQdrant, None]:
    """An initialized HandlerQdrant, shut down after the test."""
    from omnibase_infra.handlers import HandlerQdrant

    instance = HandlerQdrant(ModelONEXContainer(enable_service_registry=False))
    await instance.initialize(connection_config)
    try:
        yield instance
    finally:
        await instance.shutdown()


@pytest.fixture
def index_name() -> str:
    """A collection name unique to this test.

    Unique per test so a parallel run, or a leftover from a failed run, cannot
    make one test's assertions depend on another's data.
    """
    return f"test_collection_{uuid4().hex[:12]}"


@pytest.fixture
async def created_index(
    handler: HandlerQdrant,
    index_name: str,
) -> AsyncGenerator[str, None]:
    """A freshly created collection, deleted on every exit path."""
    await handler.create_index(
        index_name=index_name,
        dimension=DIMENSION,
        metric=EnumVectorDistanceMetric.COSINE.value,
    )
    try:
        yield index_name
    finally:
        await handler.delete_index(index_name)


# =============================================================================
# Handler metadata and health
# =============================================================================


class TestHandlerQdrantMetadata:
    """The handler describes itself, and reports on the server it is talking to."""

    @pytest.mark.asyncio
    async def test_describe_reports_the_vector_store_contract(
        self, handler: HandlerQdrant
    ) -> None:
        """describe() names the handler and the metrics it can actually serve."""
        metadata = await handler.describe()

        assert metadata.handler_type
        assert metadata.capabilities
        assert EnumVectorDistanceMetric.COSINE in metadata.supported_metrics

    @pytest.mark.asyncio
    async def test_health_check_reports_a_reachable_server(
        self, handler: HandlerQdrant
    ) -> None:
        """A live server reports healthy with a measured latency and no error.

        This is the assertion the whole job exists to make possible: it can only
        pass against a real Qdrant, and it fails rather than skips when there is
        not one.
        """
        health = await handler.health_check()

        assert health.healthy is True
        assert health.latency_ms >= 0
        assert health.last_error is None


# =============================================================================
# Collection lifecycle
# =============================================================================


class TestHandlerQdrantIndexLifecycle:
    """Collections are created and deleted against the real server."""

    @pytest.mark.asyncio
    async def test_create_index_reports_the_dimension_and_metric_it_created(
        self, handler: HandlerQdrant, index_name: str
    ) -> None:
        """The result echoes the geometry actually created, not the request."""
        result = await handler.create_index(
            index_name=index_name,
            dimension=DIMENSION,
            metric=EnumVectorDistanceMetric.COSINE.value,
        )
        try:
            assert result.success is True
            assert result.index_name == index_name
            assert result.dimension == DIMENSION
            assert result.metric is EnumVectorDistanceMetric.COSINE
        finally:
            await handler.delete_index(index_name)

    @pytest.mark.asyncio
    async def test_delete_index_reports_the_geometry_it_removed_and_the_index_is_gone(
        self, handler: HandlerQdrant, index_name: str
    ) -> None:
        """Deletion reports the real geometry, and absence is proven by a query.

        Two claims, and both needed a live server to state. The reported
        dimension and metric are read from the collection BEFORE it is removed
        — this method previously returned a placeholder ``dimension=0`` that the
        result model rejects, so it could never succeed at all (OMN-18781).
        Absence is then proven by asking the server for the collection again,
        rather than by ``health_check().indices``, whose value is cached inside
        the handler and would report the pre-delete inventory.
        """
        await handler.create_index(
            index_name=index_name,
            dimension=DIMENSION,
            metric=EnumVectorDistanceMetric.COSINE.value,
        )

        result = await handler.delete_index(index_name)

        assert result.success is True
        assert result.index_name == index_name
        assert result.dimension == DIMENSION
        assert result.metric is EnumVectorDistanceMetric.COSINE

        with pytest.raises(InfraConnectionError):
            await handler.query_similar(
                query_vector=VECTOR_X,
                top_k=1,
                index_name=index_name,
            )


# =============================================================================
# Vector operations
# =============================================================================


class TestHandlerQdrantVectorOperations:
    """Store, query and delete against a real collection."""

    @pytest.mark.asyncio
    async def test_stored_embedding_is_returned_by_a_similarity_query(
        self, handler: HandlerQdrant, created_index: str
    ) -> None:
        """The round trip: what was stored comes back, with its metadata."""
        embedding_id = str(uuid4())

        stored = await handler.store_embedding(
            embedding_id=embedding_id,
            vector=VECTOR_X,
            metadata={"kind": "unit-x"},
            index_name=created_index,
        )
        assert stored.success is True
        assert stored.embedding_id == embedding_id

        found = await handler.query_similar(
            query_vector=VECTOR_X,
            top_k=1,
            index_name=created_index,
            include_metadata=True,
        )

        assert found.total_results == 1
        hit = found.results[0]
        assert hit.id == embedding_id
        assert hit.score == pytest.approx(1.0, abs=1e-3)
        assert hit.metadata["kind"].get_string() == "unit-x"

    @pytest.mark.asyncio
    async def test_a_batch_is_ranked_by_similarity_to_the_query(
        self, handler: HandlerQdrant, created_index: str
    ) -> None:
        """Three orthogonal vectors rank deterministically against one of them."""
        ids = [str(uuid4()) for _ in range(3)]
        batch = [
            ModelEmbedding(
                id=ids[position],
                vector=vector,
                metadata={"axis": ModelSchemaValue.from_value(axis)},
            )
            for position, (vector, axis) in enumerate(
                ((VECTOR_X, "x"), (VECTOR_Y, "y"), (VECTOR_Z, "z"))
            )
        ]

        result = await handler.store_embeddings_batch(
            embeddings=batch,
            index_name=created_index,
        )
        assert result.success is True

        found = await handler.query_similar(
            query_vector=VECTOR_Y,
            top_k=3,
            index_name=created_index,
            include_metadata=True,
        )

        assert found.total_results == 3
        assert found.results[0].id == ids[1]
        assert found.results[0].score == pytest.approx(1.0, abs=1e-3)
        # The two orthogonal vectors score far below the match; the ranking is a
        # property of the server, so a handler that dropped the query vector
        # would fail here rather than returning an arbitrary order.
        assert found.results[0].score > found.results[1].score

    @pytest.mark.asyncio
    async def test_deleted_embedding_is_no_longer_returned(
        self, handler: HandlerQdrant, created_index: str
    ) -> None:
        """Delete is proven by a read-back, not by the delete call's own verdict."""
        embedding_id = str(uuid4())
        await handler.store_embedding(
            embedding_id=embedding_id,
            vector=VECTOR_X,
            metadata={"kind": "doomed"},
            index_name=created_index,
        )

        deleted = await handler.delete_embedding(
            embedding_id=embedding_id,
            index_name=created_index,
        )
        assert deleted.success is True
        assert deleted.deleted is True

        found = await handler.query_similar(
            query_vector=VECTOR_X,
            top_k=5,
            index_name=created_index,
        )
        assert [hit.id for hit in found.results] == []

    @pytest.mark.asyncio
    async def test_a_metadata_filter_narrows_the_result_set(
        self, handler: HandlerQdrant, created_index: str
    ) -> None:
        """Filtering is done by the server, so it must be exercised against one."""
        keep_id, drop_id = str(uuid4()), str(uuid4())
        await handler.store_embedding(
            embedding_id=keep_id,
            vector=VECTOR_X,
            metadata={"tenant": "keep"},
            index_name=created_index,
        )
        await handler.store_embedding(
            embedding_id=drop_id,
            vector=VECTOR_X,
            metadata={"tenant": "drop"},
            index_name=created_index,
        )

        found = await handler.query_similar(
            query_vector=VECTOR_X,
            top_k=5,
            index_name=created_index,
            filter_metadata=ModelVectorMetadataFilter(
                field="tenant",
                operator=EnumVectorFilterOperator.EQ,
                value=ModelSchemaValue.from_value("keep"),
            ),
            include_metadata=True,
        )

        assert [hit.id for hit in found.results] == [keep_id]

    @pytest.mark.asyncio
    async def test_vectors_are_returned_only_when_asked_for(
        self, handler: HandlerQdrant, created_index: str
    ) -> None:
        """include_vectors is a real round trip through the server's payload."""
        embedding_id = str(uuid4())
        await handler.store_embedding(
            embedding_id=embedding_id,
            vector=VECTOR_Z,
            metadata={"kind": "unit-z"},
            index_name=created_index,
        )

        without = await handler.query_similar(
            query_vector=VECTOR_Z,
            top_k=1,
            index_name=created_index,
            include_vectors=False,
        )
        assert without.results[0].vector is None

        with_vectors = await handler.query_similar(
            query_vector=VECTOR_Z,
            top_k=1,
            index_name=created_index,
            include_vectors=True,
        )
        assert with_vectors.results[0].vector == pytest.approx(VECTOR_Z)
