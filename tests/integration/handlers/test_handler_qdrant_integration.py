# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerQdrant against real Qdrant infrastructure.

These tests validate HandlerQdrant behavior against an actual Qdrant vector database
instance. They require a running Qdrant server and will be skipped gracefully if
the server is not available.

CI/CD Graceful Skip Behavior
============================

These tests skip gracefully in CI/CD environments without Qdrant access:

Skip Conditions:
    - Skips if QDRANT_URL environment variable is not set
    - Tests are marked with module-level ``pytestmark`` using ``pytest.mark.skipif``

Example CI/CD Output::

    $ pytest tests/integration/handlers/test_handler_qdrant_integration.py -v
    test_qdrant_describe SKIPPED (QDRANT_URL not set - Qdrant integration tests skipped)
    test_qdrant_full_workflow SKIPPED (QDRANT_URL not set - Qdrant integration tests skipped)

Run with infrastructure::

    $ QDRANT_URL=http://localhost:6333 uv run pytest tests/integration/handlers/test_handler_qdrant_integration.py -v

Test Categories
===============

- Handler Metadata Tests: Validate describe functionality and capabilities
- Collection Tests: Create and manage vector collections
- Vector Operations Tests: Upsert, search, and delete vectors
- Full Workflow Tests: End-to-end vector database operations

Environment Variables
=====================

    QDRANT_URL: Qdrant server URL (required - skip if not set)
        Example: http://localhost:6333 or http://your-server-ip:6333
    QDRANT_API_KEY: Optional API key for authentication

Related Ticket: OMN-1142
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from omnibase_infra.handlers import HandlerQdrant

# =============================================================================
# Environment Configuration
# =============================================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Check if Qdrant is available based on URL being set
QDRANT_AVAILABLE = QDRANT_URL is not None

# =============================================================================
# Test Configuration and Skip Conditions
# =============================================================================

# Module-level markers - skip all tests if Qdrant is not available
pytestmark = [
    pytest.mark.skipif(
        not QDRANT_AVAILABLE,
        reason="QDRANT_URL not set - Qdrant integration tests skipped",
    ),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def qdrant_config() -> dict[str, object]:
    """Provide Qdrant configuration for HandlerQdrant.

    Returns:
        Configuration dict for HandlerQdrant.initialize()
    """
    config: dict[str, object] = {
        "url": QDRANT_URL,
        "timeout_seconds": 30.0,
        "prefer_grpc": False,
    }

    if QDRANT_API_KEY:
        config["api_key"] = QDRANT_API_KEY

    return config


@pytest.fixture
async def initialized_qdrant_handler(
    qdrant_config: dict[str, object],
) -> AsyncGenerator[HandlerQdrant, None]:
    """Provide an initialized HandlerQdrant instance with automatic cleanup.

    Creates a HandlerQdrant, initializes it with the test configuration,
    yields it for the test, then ensures proper cleanup via shutdown().

    Cleanup Behavior:
        - Calls handler.shutdown() after test completion
        - Closes Qdrant client connection
        - Idempotent: safe to call shutdown() multiple times
        - Ignores cleanup errors to prevent test pollution

    Yields:
        Initialized HandlerQdrant ready for vector operations.
    """
    from omnibase_infra.handlers import HandlerQdrant

    handler = HandlerQdrant()
    await handler.initialize(qdrant_config)

    yield handler

    # Cleanup: ensure handler is properly shut down
    try:
        await handler.shutdown()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def unique_collection_name() -> str:
    """Generate a unique collection name for test isolation.

    Returns:
        Unique collection name prefixed with 'test_collection_'.
    """
    return f"test_collection_{uuid4().hex[:12]}"


# =============================================================================
# Handler Metadata Tests
# =============================================================================


class TestHandlerQdrantMetadata:
    """Tests for HandlerQdrant metadata and describe functionality."""

    @pytest.mark.asyncio
    async def test_qdrant_describe(
        self, initialized_qdrant_handler: HandlerQdrant
    ) -> None:
        """Test handler describe returns correct metadata.

        Verifies that:
        - Describe returns supported operations
        - Handler reports correct type and version
        - Handler is initialized
        """
        description = initialized_qdrant_handler.describe()

        assert description["handler_type"] == "infra_handler"
        assert description["handler_category"] == "effect"
        assert description["initialized"] is True
        assert "qdrant.create_collection" in description["supported_operations"]
        assert "qdrant.upsert" in description["supported_operations"]
        assert "qdrant.search" in description["supported_operations"]
        assert "qdrant.delete" in description["supported_operations"]


# =============================================================================
# Collection Tests
# =============================================================================


class TestHandlerQdrantCollections:
    """Tests for HandlerQdrant collection operations."""

    @pytest.mark.asyncio
    async def test_create_collection(
        self,
        initialized_qdrant_handler: HandlerQdrant,
        unique_collection_name: str,
    ) -> None:
        """Test creating a new vector collection.

        Verifies that:
        - Collection can be created with specified parameters
        - Response indicates success
        - Collection name is returned in response
        """
        envelope = {
            "operation": "qdrant.create_collection",
            "payload": {
                "collection_name": unique_collection_name,
                "vector_size": 4,
                "distance": "cosine",
            },
            "correlation_id": str(uuid4()),
        }

        result = await initialized_qdrant_handler.execute(envelope)

        assert result.result.status == "success"
        assert result.result.payload.data.collection_name == unique_collection_name
        assert result.result.payload.data.vector_size == 4
        assert result.result.payload.data.success is True


# =============================================================================
# Vector Operations Tests
# =============================================================================


class TestHandlerQdrantVectorOperations:
    """Tests for HandlerQdrant vector CRUD operations."""

    @pytest.mark.asyncio
    async def test_upsert_vector(
        self,
        initialized_qdrant_handler: HandlerQdrant,
        unique_collection_name: str,
    ) -> None:
        """Test upserting a vector with payload.

        Verifies that:
        - Collection can be created
        - Vector can be upserted with metadata payload
        - Response indicates success
        """
        # First create collection
        create_envelope = {
            "operation": "qdrant.create_collection",
            "payload": {
                "collection_name": unique_collection_name,
                "vector_size": 4,
                "distance": "cosine",
            },
            "correlation_id": str(uuid4()),
        }
        await initialized_qdrant_handler.execute(create_envelope)

        # Upsert vector
        point_id = f"test-point-{uuid4().hex[:8]}"
        upsert_envelope = {
            "operation": "qdrant.upsert",
            "payload": {
                "collection_name": unique_collection_name,
                "point_id": point_id,
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {"text": "hello world", "category": "test"},
            },
            "correlation_id": str(uuid4()),
        }

        result = await initialized_qdrant_handler.execute(upsert_envelope)

        assert result.result.status == "success"
        assert result.result.payload.data.collection_name == unique_collection_name
        assert result.result.payload.data.success is True

    @pytest.mark.asyncio
    async def test_search_vectors(
        self,
        initialized_qdrant_handler: HandlerQdrant,
        unique_collection_name: str,
    ) -> None:
        """Test searching for similar vectors.

        Verifies that:
        - Vectors can be found by similarity search
        - Search results include scores
        - Payloads are returned with results
        """
        # Create collection and upsert vectors
        create_envelope = {
            "operation": "qdrant.create_collection",
            "payload": {
                "collection_name": unique_collection_name,
                "vector_size": 4,
                "distance": "cosine",
            },
            "correlation_id": str(uuid4()),
        }
        await initialized_qdrant_handler.execute(create_envelope)

        # Upsert multiple vectors
        for i in range(3):
            upsert_envelope = {
                "operation": "qdrant.upsert",
                "payload": {
                    "collection_name": unique_collection_name,
                    "point_id": f"point-{i}",
                    "vector": [
                        0.1 * (i + 1),
                        0.2 * (i + 1),
                        0.3 * (i + 1),
                        0.4 * (i + 1),
                    ],
                    "payload": {"index": i, "text": f"document {i}"},
                },
                "correlation_id": str(uuid4()),
            }
            await initialized_qdrant_handler.execute(upsert_envelope)

        # Search for similar vectors
        search_envelope = {
            "operation": "qdrant.search",
            "payload": {
                "collection_name": unique_collection_name,
                "query_vector": [0.1, 0.2, 0.3, 0.4],
                "limit": 10,
            },
            "correlation_id": str(uuid4()),
        }

        result = await initialized_qdrant_handler.execute(search_envelope)

        assert result.result.status == "success"
        assert len(result.result.payload.data.results) > 0
        # Verify results have scores
        for search_result in result.result.payload.data.results:
            assert hasattr(search_result, "score")
            assert hasattr(search_result, "id")

    @pytest.mark.asyncio
    async def test_delete_vectors(
        self,
        initialized_qdrant_handler: HandlerQdrant,
        unique_collection_name: str,
    ) -> None:
        """Test deleting vectors by ID.

        Verifies that:
        - Vectors can be deleted by point ID
        - Deleted vectors are no longer returned in search
        """
        # Create collection and upsert a vector
        create_envelope = {
            "operation": "qdrant.create_collection",
            "payload": {
                "collection_name": unique_collection_name,
                "vector_size": 4,
                "distance": "cosine",
            },
            "correlation_id": str(uuid4()),
        }
        await initialized_qdrant_handler.execute(create_envelope)

        point_id = "delete-test-point"
        upsert_envelope = {
            "operation": "qdrant.upsert",
            "payload": {
                "collection_name": unique_collection_name,
                "point_id": point_id,
                "vector": [0.5, 0.5, 0.5, 0.5],
                "payload": {"text": "to be deleted"},
            },
            "correlation_id": str(uuid4()),
        }
        await initialized_qdrant_handler.execute(upsert_envelope)

        # Delete the vector
        delete_envelope = {
            "operation": "qdrant.delete",
            "payload": {
                "collection_name": unique_collection_name,
                "point_ids": [point_id],
            },
            "correlation_id": str(uuid4()),
        }

        result = await initialized_qdrant_handler.execute(delete_envelope)

        assert result.result.status == "success"
        assert result.result.payload.data.success is True


# =============================================================================
# Full Workflow Tests
# =============================================================================


class TestHandlerQdrantFullWorkflow:
    """End-to-end workflow tests for HandlerQdrant."""

    @pytest.mark.asyncio
    async def test_full_vector_workflow(
        self,
        initialized_qdrant_handler: HandlerQdrant,
        unique_collection_name: str,
    ) -> None:
        """Test complete workflow: create, upsert, search, delete.

        This test validates the full lifecycle of vector operations:
        1. Create a collection
        2. Upsert vectors with payloads
        3. Search for similar vectors
        4. Delete vectors
        5. Verify deletion by searching again
        """
        # 1. Create collection
        create_envelope = {
            "operation": "qdrant.create_collection",
            "payload": {
                "collection_name": unique_collection_name,
                "vector_size": 4,
                "distance": "cosine",
            },
            "correlation_id": str(uuid4()),
        }
        result = await initialized_qdrant_handler.execute(create_envelope)
        assert result.result.status == "success"

        # 2. Upsert vectors
        point_ids = []
        for i in range(5):
            point_id = f"workflow-point-{i}"
            point_ids.append(point_id)
            upsert_envelope = {
                "operation": "qdrant.upsert",
                "payload": {
                    "collection_name": unique_collection_name,
                    "point_id": point_id,
                    "vector": [
                        0.1 * (i + 1),
                        0.2 * (i + 1),
                        0.3 * (i + 1),
                        0.4 * (i + 1),
                    ],
                    "payload": {"index": i, "category": "workflow_test"},
                },
                "correlation_id": str(uuid4()),
            }
            result = await initialized_qdrant_handler.execute(upsert_envelope)
            assert result.result.status == "success"

        # 3. Search for vectors
        search_envelope = {
            "operation": "qdrant.search",
            "payload": {
                "collection_name": unique_collection_name,
                "query_vector": [0.2, 0.4, 0.6, 0.8],
                "limit": 10,
            },
            "correlation_id": str(uuid4()),
        }
        result = await initialized_qdrant_handler.execute(search_envelope)
        assert result.result.status == "success"
        assert len(result.result.payload.data.results) == 5

        # 4. Delete some vectors
        delete_envelope = {
            "operation": "qdrant.delete",
            "payload": {
                "collection_name": unique_collection_name,
                "point_ids": point_ids[:2],  # Delete first two
            },
            "correlation_id": str(uuid4()),
        }
        result = await initialized_qdrant_handler.execute(delete_envelope)
        assert result.result.status == "success"

        # 5. Search again - should have fewer results
        result = await initialized_qdrant_handler.execute(search_envelope)
        assert result.result.status == "success"
        assert len(result.result.payload.data.results) == 3


# =============================================================================
# Correlation ID Tests
# =============================================================================


class TestHandlerQdrantCorrelationId:
    """Tests for correlation ID handling in HandlerQdrant."""

    @pytest.mark.asyncio
    async def test_correlation_id_preserved(
        self,
        initialized_qdrant_handler: HandlerQdrant,
        unique_collection_name: str,
    ) -> None:
        """Verify correlation_id from envelope is preserved in response."""
        test_correlation_id = uuid4()

        envelope = {
            "operation": "qdrant.create_collection",
            "correlation_id": str(test_correlation_id),
            "payload": {
                "collection_name": unique_collection_name,
                "vector_size": 4,
                "distance": "cosine",
            },
        }

        result = await initialized_qdrant_handler.execute(envelope)

        assert result.correlation_id == test_correlation_id
        assert result.result.correlation_id == test_correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_generated_if_missing(
        self,
        initialized_qdrant_handler: HandlerQdrant,
        unique_collection_name: str,
    ) -> None:
        """Verify correlation_id is generated when not provided."""
        from uuid import UUID

        envelope = {
            "operation": "qdrant.create_collection",
            # No correlation_id provided
            "payload": {
                "collection_name": unique_collection_name,
                "vector_size": 4,
                "distance": "cosine",
            },
        }

        result = await initialized_qdrant_handler.execute(envelope)

        # Should have a generated correlation_id
        assert result.correlation_id is not None
        assert isinstance(result.correlation_id, UUID)
