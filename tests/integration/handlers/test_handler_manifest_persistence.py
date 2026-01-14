# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerManifestPersistence.

This module tests the manifest persistence handler which stores ModelExecutionManifest
objects to the filesystem with date-based partitioning, atomic writes, and query support.

Test Coverage:
    - TestCoreOperations: Basic store, retrieve, and query functionality
    - TestMetadataOnlyQuery: Lightweight metadata-only query mode
    - TestFileBackendSpecifics: Filesystem-specific behaviors (partitioning, atomicity)
    - TestErrorHandling: Error cases and edge conditions
    - TestHandlerLifecycle: Handler initialization and shutdown
    - TestQueryCombinations: Complex query scenarios with multiple filters
    - TestConcurrentWrites: Concurrent write safety verification

Circuit Breaker Tests (TestCircuitBreakerBehavior):
    The handler uses MixinAsyncCircuitBreaker for resilient I/O operations.
    Tests verify the following behaviors:
    - Circuit opens after threshold failures (default: 5)
    - Operations blocked when circuit is open (raises InfraUnavailableError)
    - Circuit resets on successful operation
    - Half-open state allows test request after timeout
    See: docs/patterns/circuit_breaker_implementation.md for implementation details.

Related:
    - OMN-1163: Manifest persistence handler implementation
    - src/omnibase_infra/handlers/handler_manifest_persistence.py

Note:
    These tests follow TDD principles - tests are written before the handler
    implementation. The handler should be implemented to make these tests pass.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import (
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.handlers.handler_manifest_persistence import (
    HandlerManifestPersistence,
)

# =============================================================================
# Helper Functions
# =============================================================================


def create_test_manifest(
    manifest_id: UUID | None = None,
    correlation_id: UUID | None = None,
    node_id: str = "test-node",
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Create a test manifest dict matching ModelExecutionManifest structure.

    Args:
        manifest_id: Optional manifest UUID. Generated if not provided.
        correlation_id: Optional correlation UUID.
        node_id: Node identifier for the manifest.
        created_at: Optional creation timestamp. Uses current time if not provided.

    Returns:
        Dict representing a minimal valid execution manifest.
    """
    return {
        "manifest_id": str(manifest_id or uuid4()),
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "correlation_id": str(correlation_id) if correlation_id else None,
        "node_identity": {
            "node_id": node_id,
            "node_type": "test",
        },
        "contract_identity": {
            "contract_id": "test-contract",
            "contract_version": "1.0.0",
        },
        "execution_context": {
            "environment": "test",
            "session_id": str(uuid4()),
        },
    }


def create_store_envelope(
    manifest: dict[str, object],
    correlation_id: UUID | None = None,
) -> dict[str, object]:
    """Create envelope for manifest.store operation.

    Args:
        manifest: The manifest dict to store.
        correlation_id: Optional correlation ID for tracing.

    Returns:
        Envelope dict for execute() method.
    """
    return {
        "id": str(uuid4()),
        "operation": "manifest.store",
        "payload": {"manifest": manifest},
        "correlation_id": str(correlation_id or uuid4()),
    }


def create_retrieve_envelope(
    manifest_id: str | UUID,
    correlation_id: UUID | None = None,
) -> dict[str, object]:
    """Create envelope for manifest.retrieve operation.

    Args:
        manifest_id: The manifest ID to retrieve.
        correlation_id: Optional correlation ID for tracing.

    Returns:
        Envelope dict for execute() method.
    """
    return {
        "id": str(uuid4()),
        "operation": "manifest.retrieve",
        "payload": {"manifest_id": str(manifest_id)},
        "correlation_id": str(correlation_id or uuid4()),
    }


def create_query_envelope(
    correlation_id: UUID | None = None,
    node_id: str | None = None,
    created_after: datetime | None = None,
    limit: int | None = None,
    metadata_only: bool = False,
    envelope_correlation_id: UUID | None = None,
) -> dict[str, object]:
    """Create envelope for manifest.query operation.

    Args:
        correlation_id: Filter by manifest correlation_id.
        node_id: Filter by node_id.
        created_after: Filter by creation time.
        limit: Maximum number of results.
        metadata_only: Return only summary metadata.
        envelope_correlation_id: Correlation ID for the envelope itself.

    Returns:
        Envelope dict for execute() method.
    """
    payload: dict[str, object] = {}
    if correlation_id is not None:
        payload["correlation_id"] = str(correlation_id)
    if node_id is not None:
        payload["node_id"] = node_id
    if created_after is not None:
        payload["created_after"] = created_after.isoformat()
    if limit is not None:
        payload["limit"] = limit
    if metadata_only:
        payload["metadata_only"] = metadata_only

    return {
        "id": str(uuid4()),
        "operation": "manifest.query",
        "payload": payload,
        "correlation_id": str(envelope_correlation_id or uuid4()),
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_storage_path(tmp_path: Path) -> Path:
    """Create temporary storage directory for tests.

    Args:
        tmp_path: pytest tmp_path fixture.

    Returns:
        Path to temporary manifest storage directory.
    """
    return tmp_path / "manifests"


@pytest.fixture
async def handler(
    temp_storage_path: Path,
) -> AsyncGenerator[HandlerManifestPersistence, None]:
    """Create and initialize handler with temp storage.

    Args:
        temp_storage_path: Temporary storage directory.

    Yields:
        Initialized HandlerManifestPersistence instance.
    """
    h = HandlerManifestPersistence()
    await h.initialize({"storage_path": str(temp_storage_path)})
    yield h
    await h.shutdown()


# =============================================================================
# TestCoreOperations
# =============================================================================


class TestCoreOperations:
    """Test core manifest persistence operations: store, retrieve, and query."""

    @pytest.mark.asyncio
    async def test_store_manifest_returns_manifest_id(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Store returns the manifest_id from the stored manifest.

        Validates that store operation returns the correct manifest_id
        and indicates successful creation.
        """
        manifest = create_test_manifest()
        manifest_id = manifest["manifest_id"]

        result = await handler.execute(create_store_envelope(manifest))

        assert result.result["status"] == "success"
        # Compare as strings since manifest_id might be serialized
        assert str(result.result["payload"]["manifest_id"]) == manifest_id
        assert result.result["payload"]["created"] is True

    @pytest.mark.asyncio
    async def test_retrieve_by_id_returns_manifest(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Retrieve by manifest_id returns the full manifest.

        Validates that a stored manifest can be retrieved by its ID
        and the content matches what was stored.
        """
        manifest = create_test_manifest()
        manifest_id = manifest["manifest_id"]

        # Store the manifest
        await handler.execute(create_store_envelope(manifest))

        # Retrieve by ID
        result = await handler.execute(create_retrieve_envelope(manifest_id))

        assert result.result["status"] == "success"
        assert result.result["payload"]["found"] is True
        retrieved_manifest = result.result["payload"]["manifest"]
        assert retrieved_manifest["manifest_id"] == manifest_id
        assert (
            retrieved_manifest["node_identity"]["node_id"]
            == manifest["node_identity"]["node_id"]
        )

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_returns_none(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Retrieve with unknown ID returns found=False, manifest=None.

        Validates that retrieving a non-existent manifest returns
        a graceful response instead of an error.
        """
        nonexistent_id = uuid4()

        result = await handler.execute(create_retrieve_envelope(nonexistent_id))

        assert result.result["status"] == "success"
        assert result.result["payload"]["found"] is False
        assert result.result["payload"]["manifest"] is None

    @pytest.mark.asyncio
    async def test_query_by_correlation_id(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Query filters by correlation_id correctly.

        Validates that manifests can be queried by their correlation_id
        and only matching manifests are returned.
        """
        target_correlation_id = uuid4()
        other_correlation_id = uuid4()

        # Store manifests with different correlation IDs
        manifest1 = create_test_manifest(
            correlation_id=target_correlation_id, node_id="node-1"
        )
        manifest2 = create_test_manifest(
            correlation_id=target_correlation_id, node_id="node-2"
        )
        manifest3 = create_test_manifest(
            correlation_id=other_correlation_id, node_id="node-3"
        )

        await handler.execute(create_store_envelope(manifest1))
        await handler.execute(create_store_envelope(manifest2))
        await handler.execute(create_store_envelope(manifest3))

        # Query by target correlation_id (uses manifest_data when metadata_only=False)
        result = await handler.execute(
            create_query_envelope(correlation_id=target_correlation_id)
        )

        assert result.result["status"] == "success"
        # When metadata_only=False (default), manifests are in manifest_data
        manifests = result.result["payload"]["manifest_data"]
        assert len(manifests) == 2
        manifest_ids = {m["manifest_id"] for m in manifests}
        assert manifest1["manifest_id"] in manifest_ids
        assert manifest2["manifest_id"] in manifest_ids
        assert manifest3["manifest_id"] not in manifest_ids

    @pytest.mark.asyncio
    async def test_query_by_node_id(self, handler: HandlerManifestPersistence) -> None:
        """Query filters by node_id correctly.

        Validates that manifests can be queried by their node_id
        and only matching manifests are returned.
        """
        target_node_id = "target-node"
        other_node_id = "other-node"

        # Store manifests with different node IDs
        manifest1 = create_test_manifest(node_id=target_node_id)
        manifest2 = create_test_manifest(node_id=target_node_id)
        manifest3 = create_test_manifest(node_id=other_node_id)

        await handler.execute(create_store_envelope(manifest1))
        await handler.execute(create_store_envelope(manifest2))
        await handler.execute(create_store_envelope(manifest3))

        # Query by target node_id
        result = await handler.execute(create_query_envelope(node_id=target_node_id))

        assert result.result["status"] == "success"
        # When metadata_only=False (default), manifests are in manifest_data
        manifests = result.result["payload"]["manifest_data"]
        assert len(manifests) == 2
        for m in manifests:
            assert m["node_identity"]["node_id"] == target_node_id

    @pytest.mark.asyncio
    async def test_query_by_date_range(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Query respects created_after filter.

        Validates that manifests can be filtered by creation time
        and only manifests created after the specified time are returned.
        """
        now = datetime.now(UTC)
        past = now - timedelta(hours=2)
        recent = now - timedelta(minutes=30)

        # Create manifests with different creation times
        old_manifest = create_test_manifest(node_id="old", created_at=past)
        new_manifest = create_test_manifest(node_id="new", created_at=recent)

        await handler.execute(create_store_envelope(old_manifest))
        await handler.execute(create_store_envelope(new_manifest))

        # Query for manifests created after 1 hour ago
        cutoff = now - timedelta(hours=1)
        result = await handler.execute(create_query_envelope(created_after=cutoff))

        assert result.result["status"] == "success"
        # When metadata_only=False (default), manifests are in manifest_data
        manifests = result.result["payload"]["manifest_data"]
        assert len(manifests) == 1
        assert manifests[0]["node_identity"]["node_id"] == "new"

    @pytest.mark.asyncio
    async def test_query_with_limit(self, handler: HandlerManifestPersistence) -> None:
        """Query respects limit parameter.

        Validates that the query limit parameter restricts
        the number of returned results.
        """
        # Store multiple manifests
        for i in range(5):
            manifest = create_test_manifest(node_id=f"node-{i}")
            await handler.execute(create_store_envelope(manifest))

        # Query with limit
        result = await handler.execute(create_query_envelope(limit=3))

        assert result.result["status"] == "success"
        # When metadata_only=False (default), manifests are in manifest_data
        manifests = result.result["payload"]["manifest_data"]
        assert len(manifests) == 3


# =============================================================================
# TestMetadataOnlyQuery
# =============================================================================


class TestMetadataOnlyQuery:
    """Test metadata-only query mode for lightweight manifest discovery."""

    @pytest.mark.asyncio
    async def test_query_metadata_only_returns_summary(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """metadata_only=true returns only id, created_at, correlation_id, node_id.

        Validates that metadata-only queries return a lightweight summary
        containing only essential identification fields.
        """
        manifest = create_test_manifest(
            correlation_id=uuid4(),
            node_id="summary-test-node",
        )
        await handler.execute(create_store_envelope(manifest))

        result = await handler.execute(create_query_envelope(metadata_only=True))

        assert result.result["status"] == "success"
        # When metadata_only=True, results are in manifests (list of metadata)
        manifests = result.result["payload"]["manifests"]
        assert len(manifests) == 1

        summary = manifests[0]
        # Should contain these fields (ModelManifestMetadata fields)
        assert "manifest_id" in summary
        assert "created_at" in summary
        assert "correlation_id" in summary
        assert "node_id" in summary
        # Compare as strings since manifest_id might be serialized
        assert str(summary["manifest_id"]) == manifest["manifest_id"]
        assert summary["node_id"] == "summary-test-node"

    @pytest.mark.asyncio
    async def test_query_metadata_only_excludes_full_manifest(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """metadata_only query result should not contain full manifest fields.

        Validates that metadata-only results exclude heavyweight fields
        like execution_context, contract_identity details, etc.
        """
        manifest = create_test_manifest()
        await handler.execute(create_store_envelope(manifest))

        result = await handler.execute(create_query_envelope(metadata_only=True))

        assert result.result["status"] == "success"
        # When metadata_only=True, results are in manifests (list of metadata)
        manifests = result.result["payload"]["manifests"]
        assert len(manifests) == 1

        summary = manifests[0]
        # Should NOT contain full manifest fields
        assert "execution_context" not in summary
        assert "contract_identity" not in summary
        assert "node_identity" not in summary  # Only node_id extracted


# =============================================================================
# TestFileBackendSpecifics
# =============================================================================


class TestFileBackendSpecifics:
    """Test filesystem-specific behaviors: partitioning, atomicity, JSON validity."""

    @pytest.mark.asyncio
    async def test_partitioned_directory_structure(
        self,
        handler: HandlerManifestPersistence,
        temp_storage_path: Path,
    ) -> None:
        """Files stored in year/month/day subdirectories.

        Validates that manifests are stored in a date-partitioned directory
        structure for efficient querying and archival.
        """
        manifest = create_test_manifest()
        await handler.execute(create_store_envelope(manifest))

        # Check that file exists in partitioned structure
        now = datetime.now(UTC)
        expected_dir = (
            temp_storage_path / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
        )

        assert expected_dir.exists(), (
            f"Expected partitioned directory {expected_dir} to exist"
        )

        # Find the manifest file
        manifest_files = list(expected_dir.glob("*.json"))
        assert len(manifest_files) == 1
        assert manifest["manifest_id"] in manifest_files[0].name

    @pytest.mark.asyncio
    async def test_idempotent_store_same_manifest(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Storing same manifest_id twice returns created=False second time.

        Validates idempotent behavior - storing a manifest with the same ID
        should not create a duplicate and should indicate it was not newly created.
        """
        manifest = create_test_manifest()

        # First store
        result1 = await handler.execute(create_store_envelope(manifest))
        assert result1.result["status"] == "success"
        assert result1.result["payload"]["created"] is True

        # Second store with same manifest_id
        result2 = await handler.execute(create_store_envelope(manifest))
        assert result2.result["status"] == "success"
        assert result2.result["payload"]["created"] is False
        # Compare as strings since manifest_id might be serialized
        assert str(result2.result["payload"]["manifest_id"]) == manifest["manifest_id"]

    @pytest.mark.asyncio
    async def test_atomic_write_creates_valid_json(
        self,
        handler: HandlerManifestPersistence,
        temp_storage_path: Path,
    ) -> None:
        """Stored files are valid JSON.

        Validates that the atomic write process produces valid JSON files
        that can be parsed without errors.
        """
        manifest = create_test_manifest(node_id="json-test-node")
        await handler.execute(create_store_envelope(manifest))

        # Find all JSON files in storage
        json_files = list(temp_storage_path.rglob("*.json"))
        assert len(json_files) == 1

        # Verify the file contains valid JSON
        file_content = json_files[0].read_text(encoding="utf-8")
        parsed = json.loads(file_content)

        assert parsed["manifest_id"] == manifest["manifest_id"]
        assert parsed["node_identity"]["node_id"] == "json-test-node"

    @pytest.mark.asyncio
    async def test_multiple_manifests_separate_files(
        self,
        handler: HandlerManifestPersistence,
        temp_storage_path: Path,
    ) -> None:
        """Each manifest is stored in a separate file.

        Validates that multiple manifests are stored as individual files,
        not appended to a single file.
        """
        manifest1 = create_test_manifest(node_id="node-1")
        manifest2 = create_test_manifest(node_id="node-2")
        manifest3 = create_test_manifest(node_id="node-3")

        await handler.execute(create_store_envelope(manifest1))
        await handler.execute(create_store_envelope(manifest2))
        await handler.execute(create_store_envelope(manifest3))

        # Verify three separate files
        json_files = list(temp_storage_path.rglob("*.json"))
        assert len(json_files) == 3

        # Verify each file contains a different manifest
        manifest_ids = set()
        for f in json_files:
            content = json.loads(f.read_text(encoding="utf-8"))
            manifest_ids.add(content["manifest_id"])

        assert manifest1["manifest_id"] in manifest_ids
        assert manifest2["manifest_id"] in manifest_ids
        assert manifest3["manifest_id"] in manifest_ids


# =============================================================================
# TestErrorHandling
# =============================================================================


class TestErrorHandling:
    """Test error handling for invalid inputs and edge cases."""

    @pytest.mark.asyncio
    async def test_store_invalid_manifest_raises_error(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Store with invalid payload raises ProtocolConfigurationError.

        Validates that storing an invalid manifest (missing required fields)
        raises an appropriate error.
        """
        invalid_manifest = {"invalid": "manifest"}

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.execute(create_store_envelope(invalid_manifest))

        assert (
            "manifest_id" in str(exc_info.value).lower()
            or "invalid" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_store_missing_manifest_in_payload_raises_error(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Store with missing manifest in payload raises ProtocolConfigurationError.

        Validates that the store operation requires a manifest in the payload.
        """
        envelope = {
            "id": str(uuid4()),
            "operation": "manifest.store",
            "payload": {},  # Missing 'manifest' key
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert "manifest" in error_msg or "payload" in error_msg

    @pytest.mark.asyncio
    async def test_handler_not_initialized_raises_error(
        self, temp_storage_path: Path
    ) -> None:
        """Operations before initialize() raise RuntimeHostError.

        Validates that attempting operations on an uninitialized handler
        raises an appropriate error.
        """
        handler = HandlerManifestPersistence()
        # Do NOT call initialize()

        manifest = create_test_manifest()

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(create_store_envelope(manifest))

        error_msg = str(exc_info.value).lower()
        assert "initialized" in error_msg or "initialize" in error_msg

    @pytest.mark.asyncio
    async def test_unsupported_operation_raises_error(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Unknown operation raises ProtocolConfigurationError.

        Validates that attempting an unsupported operation
        raises an appropriate error with helpful message.
        """
        envelope = {
            "id": str(uuid4()),
            "operation": "manifest.unsupported_operation",
            "payload": {},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert (
            "unsupported" in error_msg
            or "not supported" in error_msg
            or "unknown" in error_msg
        )

    @pytest.mark.asyncio
    async def test_retrieve_invalid_manifest_id_format_raises_error(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Retrieve with invalid manifest_id format raises error.

        Validates that retrieving with a malformed manifest_id
        raises an appropriate error.
        """
        envelope = {
            "id": str(uuid4()),
            "operation": "manifest.retrieve",
            "payload": {"manifest_id": "not-a-valid-uuid"},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert (
            "uuid" in error_msg or "manifest_id" in error_msg or "invalid" in error_msg
        )

    @pytest.mark.asyncio
    async def test_query_invalid_date_format_ignored(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Query with invalid created_after format ignores the filter.

        The handler silently ignores invalid filter values rather than
        raising errors, treating them as if the filter wasn't provided.
        """
        # Store a manifest first
        manifest = create_test_manifest(node_id="date-test")
        await handler.execute(create_store_envelope(manifest))

        envelope = {
            "id": str(uuid4()),
            "operation": "manifest.query",
            "payload": {"created_after": "not-a-date"},
            "correlation_id": str(uuid4()),
        }

        # Should succeed - invalid date is silently ignored
        result = await handler.execute(envelope)
        assert result.result["status"] == "success"
        # The manifest should be returned since filter was ignored
        assert result.result["payload"]["total_count"] == 1


# =============================================================================
# TestHandlerLifecycle
# =============================================================================


class TestHandlerLifecycle:
    """Test handler initialization, describe, and shutdown behaviors."""

    @pytest.mark.asyncio
    async def test_describe_returns_capabilities(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """describe() returns handler metadata.

        Validates that the describe method returns comprehensive
        handler metadata including supported operations and configuration.
        """
        description = handler.describe()

        assert "handler_type" in description
        assert "supported_operations" in description
        assert "manifest.store" in description["supported_operations"]
        assert "manifest.retrieve" in description["supported_operations"]
        assert "manifest.query" in description["supported_operations"]
        assert "initialized" in description
        assert description["initialized"] is True

    @pytest.mark.asyncio
    async def test_describe_includes_circuit_breaker_state_when_initialized(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """describe() includes circuit breaker state when handler is initialized.

        Validates that the describe method returns circuit breaker
        state information for observability purposes.
        """
        description = handler.describe()

        # Circuit breaker info should be present
        assert "circuit_breaker" in description
        cb_state = description["circuit_breaker"]
        assert isinstance(cb_state, dict)

        # Validate circuit breaker state fields
        assert "initialized" in cb_state
        assert cb_state["initialized"] is True  # Should be initialized

        assert "state" in cb_state
        assert cb_state["state"] == "closed"  # Should be closed initially

        assert "failures" in cb_state
        assert cb_state["failures"] == 0  # No failures yet

        assert "threshold" in cb_state
        assert cb_state["threshold"] == 5  # Default threshold

    @pytest.mark.asyncio
    async def test_describe_circuit_breaker_not_initialized_before_initialize(
        self, temp_storage_path: Path
    ) -> None:
        """describe() shows circuit breaker not initialized before initialize().

        Circuit breaker is only set up during initialize(), so describe()
        should show initialized=False when called before initialization.
        """
        handler = HandlerManifestPersistence()
        description = handler.describe()

        # Circuit breaker info should be present but show not initialized
        assert "circuit_breaker" in description
        cb_state = description["circuit_breaker"]
        assert cb_state["initialized"] is False
        assert cb_state["state"] == "closed"  # Default state
        assert cb_state["failures"] == 0  # Default failures
        assert cb_state["threshold"] == 5  # Default threshold
        assert description["initialized"] is False

    @pytest.mark.asyncio
    async def test_initialize_creates_storage_directory(
        self, temp_storage_path: Path
    ) -> None:
        """initialize() creates storage_path if it doesn't exist.

        Validates that the handler creates the storage directory
        during initialization if it doesn't already exist.
        """
        assert not temp_storage_path.exists()

        handler = HandlerManifestPersistence()
        await handler.initialize({"storage_path": str(temp_storage_path)})

        assert temp_storage_path.exists()
        assert temp_storage_path.is_dir()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_with_existing_directory(
        self, temp_storage_path: Path
    ) -> None:
        """initialize() succeeds when storage_path already exists.

        Validates that initialization does not fail if the storage
        directory already exists.
        """
        temp_storage_path.mkdir(parents=True, exist_ok=True)
        assert temp_storage_path.exists()

        handler = HandlerManifestPersistence()
        await handler.initialize({"storage_path": str(temp_storage_path)})

        assert handler.describe()["initialized"] is True

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_missing_storage_path_raises_error(self) -> None:
        """initialize() without storage_path raises ProtocolConfigurationError.

        Validates that initialization requires the storage_path configuration.
        """
        handler = HandlerManifestPersistence()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.initialize({})

        error_msg = str(exc_info.value).lower()
        assert "storage_path" in error_msg or "required" in error_msg

    @pytest.mark.asyncio
    async def test_shutdown_clears_initialized_state(
        self, temp_storage_path: Path
    ) -> None:
        """shutdown() clears the initialized state.

        Validates that shutdown properly resets the handler state.
        """
        handler = HandlerManifestPersistence()
        await handler.initialize({"storage_path": str(temp_storage_path)})
        assert handler.describe()["initialized"] is True

        await handler.shutdown()

        assert handler.describe()["initialized"] is False

    @pytest.mark.asyncio
    async def test_operations_after_shutdown_raise_error(
        self, temp_storage_path: Path
    ) -> None:
        """Operations after shutdown() raise RuntimeHostError.

        Validates that attempting operations after shutdown
        raises an appropriate error.
        """
        handler = HandlerManifestPersistence()
        await handler.initialize({"storage_path": str(temp_storage_path)})
        await handler.shutdown()

        manifest = create_test_manifest()

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(create_store_envelope(manifest))

        error_msg = str(exc_info.value).lower()
        assert "initialized" in error_msg or "shutdown" in error_msg


# =============================================================================
# TestQueryCombinations
# =============================================================================


class TestQueryCombinations:
    """Test complex query scenarios with multiple filters."""

    @pytest.mark.asyncio
    async def test_query_combined_filters(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Query with multiple filters combines them with AND logic.

        Validates that combining correlation_id and node_id filters
        returns only manifests matching both criteria.
        """
        target_correlation = uuid4()
        target_node = "target-combined"

        # Create manifests with various combinations
        match_both = create_test_manifest(
            correlation_id=target_correlation, node_id=target_node
        )
        match_correlation = create_test_manifest(
            correlation_id=target_correlation, node_id="other-node"
        )
        match_node = create_test_manifest(correlation_id=uuid4(), node_id=target_node)
        match_neither = create_test_manifest(
            correlation_id=uuid4(), node_id="other-node"
        )

        await handler.execute(create_store_envelope(match_both))
        await handler.execute(create_store_envelope(match_correlation))
        await handler.execute(create_store_envelope(match_node))
        await handler.execute(create_store_envelope(match_neither))

        # Query with both filters
        result = await handler.execute(
            create_query_envelope(
                correlation_id=target_correlation,
                node_id=target_node,
            )
        )

        assert result.result["status"] == "success"
        # When metadata_only=False (default), manifests are in manifest_data
        manifests = result.result["payload"]["manifest_data"]
        assert len(manifests) == 1
        assert manifests[0]["manifest_id"] == match_both["manifest_id"]

    @pytest.mark.asyncio
    async def test_query_empty_result(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Query with no matches returns empty list.

        Validates that queries with no matching manifests return
        an empty list gracefully without errors.
        """
        # Store a manifest
        manifest = create_test_manifest(node_id="existing-node")
        await handler.execute(create_store_envelope(manifest))

        # Query for non-existent node_id
        result = await handler.execute(
            create_query_envelope(node_id="nonexistent-node")
        )

        assert result.result["status"] == "success"
        # Both manifests (metadata) and manifest_data (full) should be empty
        assert result.result["payload"]["manifest_data"] == []
        assert result.result["payload"]["total_count"] == 0

    @pytest.mark.asyncio
    async def test_query_returns_count(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Query result includes count field.

        Validates that query results include a count field
        indicating the number of matching manifests.
        """
        for i in range(5):
            manifest = create_test_manifest(node_id=f"node-{i}")
            await handler.execute(create_store_envelope(manifest))

        result = await handler.execute(create_query_envelope())

        assert result.result["status"] == "success"
        assert result.result["payload"]["total_count"] == 5
        # When metadata_only=False (default), manifests are in manifest_data
        assert len(result.result["payload"]["manifest_data"]) == 5


# =============================================================================
# TestCircuitBreakerBehavior
# =============================================================================


class TestCircuitBreakerBehavior:
    """Test circuit breaker behavior for resilient I/O operations.

    The handler uses MixinAsyncCircuitBreaker for resilient I/O operations with:
    - Threshold: 5 failures before circuit opens
    - Reset timeout: 60 seconds before half-open transition
    - Transport type: FILESYSTEM

    These tests verify:
    - Circuit opens after threshold failures
    - Operations blocked when circuit is open (raises InfraUnavailableError)
    - Circuit resets on successful operation
    - Half-open state behavior (after timeout)
    """

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(
        self, temp_storage_path: Path
    ) -> None:
        """Circuit opens after 5 consecutive failures (default threshold).

        Validates that the circuit breaker opens after the configured threshold
        of failures is reached, preventing further operations.
        """
        from omnibase_infra.errors import InfraUnavailableError

        handler = HandlerManifestPersistence()
        await handler.initialize({"storage_path": str(temp_storage_path)})

        # Verify circuit breaker is configured with expected threshold
        assert handler.circuit_breaker_threshold == 5

        # Manually record failures to trip the circuit
        # We simulate 5 failures to reach the threshold
        for i in range(5):
            async with handler._circuit_breaker_lock:
                await handler._record_circuit_failure(
                    operation="test_operation",
                    correlation_id=uuid4(),
                )

        # Verify circuit is now open
        assert handler._circuit_breaker_open is True

        # Verify operations are blocked
        manifest = create_test_manifest()
        with pytest.raises(InfraUnavailableError) as exc_info:
            await handler.execute(create_store_envelope(manifest))

        # Verify error context
        error_msg = str(exc_info.value).lower()
        assert "circuit breaker" in error_msg or "unavailable" in error_msg

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_operations_blocked_when_circuit_open(
        self, temp_storage_path: Path
    ) -> None:
        """Operations raise InfraUnavailableError when circuit is open.

        Validates that all operations (store, retrieve, query) are blocked
        when the circuit breaker is in the open state.
        """
        from omnibase_infra.errors import InfraUnavailableError

        handler = HandlerManifestPersistence()
        await handler.initialize({"storage_path": str(temp_storage_path)})

        # Force circuit to open state
        handler._circuit_breaker_open = True
        handler._circuit_breaker_open_until = (
            time.time() + 60.0
        )  # Open for 60 more seconds

        # Test that store operation is blocked
        manifest = create_test_manifest()
        with pytest.raises(InfraUnavailableError):
            await handler.execute(create_store_envelope(manifest))

        # Test that retrieve operation is blocked
        with pytest.raises(InfraUnavailableError):
            await handler.execute(create_retrieve_envelope(uuid4()))

        # Test that query operation is blocked
        with pytest.raises(InfraUnavailableError):
            await handler.execute(create_query_envelope())

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_resets_on_successful_operation(
        self, temp_storage_path: Path
    ) -> None:
        """Circuit resets to closed state on successful operation.

        Validates that after a successful operation, the circuit breaker
        resets its failure count and returns to closed state.
        """
        handler = HandlerManifestPersistence()
        await handler.initialize({"storage_path": str(temp_storage_path)})

        # Record some failures (but not enough to trip the circuit)
        for _ in range(3):
            async with handler._circuit_breaker_lock:
                await handler._record_circuit_failure(
                    operation="test_operation",
                    correlation_id=uuid4(),
                )

        # Verify failures are recorded
        assert handler._circuit_breaker_failures == 3

        # Perform a successful operation
        manifest = create_test_manifest()
        result = await handler.execute(create_store_envelope(manifest))
        assert result.result["status"] == "success"

        # Verify circuit has been reset
        assert handler._circuit_breaker_failures == 0
        assert handler._circuit_breaker_open is False

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_half_open_state_after_timeout(self, temp_storage_path: Path) -> None:
        """Circuit transitions to half-open state after reset timeout.

        Validates that when the circuit is open and the reset timeout has
        elapsed, the circuit transitions to half-open state allowing a
        test request through.
        """
        handler = HandlerManifestPersistence()
        await handler.initialize({"storage_path": str(temp_storage_path)})

        # Force circuit to open state with timeout in the past
        handler._circuit_breaker_open = True
        handler._circuit_breaker_failures = 5
        # Set open_until to a time in the past (simulating timeout elapsed)
        handler._circuit_breaker_open_until = time.time() - 1.0

        # Perform an operation - should transition to half-open and succeed
        manifest = create_test_manifest()
        result = await handler.execute(create_store_envelope(manifest))

        # Operation should succeed (half-open allows test request)
        assert result.result["status"] == "success"

        # Circuit should be reset after successful operation
        assert handler._circuit_breaker_open is False
        assert handler._circuit_breaker_failures == 0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_failure_count_increments(
        self, temp_storage_path: Path
    ) -> None:
        """Each failure increments the circuit breaker failure count.

        Validates that failures are properly tracked by the circuit breaker
        and the count increases with each recorded failure.
        """
        handler = HandlerManifestPersistence()
        await handler.initialize({"storage_path": str(temp_storage_path)})

        # Verify initial state
        assert handler._circuit_breaker_failures == 0

        # Record failures incrementally
        for expected_count in range(1, 4):
            async with handler._circuit_breaker_lock:
                await handler._record_circuit_failure(
                    operation="test_operation",
                    correlation_id=uuid4(),
                )
            assert handler._circuit_breaker_failures == expected_count

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_configuration(self, temp_storage_path: Path) -> None:
        """Circuit breaker is initialized with correct configuration.

        Validates that the circuit breaker is configured with the expected
        threshold, timeout, and service name during handler initialization.
        """
        handler = HandlerManifestPersistence()
        await handler.initialize({"storage_path": str(temp_storage_path)})

        # Verify configuration values
        assert handler.circuit_breaker_threshold == 5
        assert handler.circuit_breaker_reset_timeout == 60.0
        assert handler.service_name == "manifest_persistence_handler"

        # Verify initial state
        assert handler._circuit_breaker_open is False
        assert handler._circuit_breaker_failures == 0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_infra_unavailable_error_contains_retry_after(
        self, temp_storage_path: Path
    ) -> None:
        """InfraUnavailableError includes retry_after_seconds when circuit is open.

        Validates that when the circuit is open, the raised error contains
        information about when to retry (seconds remaining until timeout).
        """
        from omnibase_infra.errors import InfraUnavailableError

        handler = HandlerManifestPersistence()
        await handler.initialize({"storage_path": str(temp_storage_path)})

        # Force circuit to open state with specific timeout
        handler._circuit_breaker_open = True
        handler._circuit_breaker_open_until = time.time() + 30.0  # 30 seconds remaining

        manifest = create_test_manifest()
        with pytest.raises(InfraUnavailableError) as exc_info:
            await handler.execute(create_store_envelope(manifest))

        # Verify error has retry_after information in model context
        error = exc_info.value
        assert "retry_after_seconds" in error.model.context
        # retry_after should be approximately 30 seconds (allow some tolerance)
        retry_after = error.model.context["retry_after_seconds"]
        assert 25 <= retry_after <= 31

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_opens_exactly_at_threshold(
        self, temp_storage_path: Path
    ) -> None:
        """Circuit opens exactly when failure count reaches threshold.

        Validates that the circuit stays closed at threshold-1 failures
        and opens exactly at threshold failures.
        """
        handler = HandlerManifestPersistence()
        await handler.initialize({"storage_path": str(temp_storage_path)})

        # Record threshold-1 failures (4 failures)
        for _ in range(4):
            async with handler._circuit_breaker_lock:
                await handler._record_circuit_failure(
                    operation="test_operation",
                    correlation_id=uuid4(),
                )

        # Circuit should still be closed
        assert handler._circuit_breaker_open is False
        assert handler._circuit_breaker_failures == 4

        # Record one more failure to reach threshold
        async with handler._circuit_breaker_lock:
            await handler._record_circuit_failure(
                operation="test_operation",
                correlation_id=uuid4(),
            )

        # Circuit should now be open
        assert handler._circuit_breaker_open is True
        assert handler._circuit_breaker_failures == 5

        await handler.shutdown()


# =============================================================================
# TestConcurrentWrites
# =============================================================================


class TestConcurrentWrites:
    """Test concurrent write safety for atomic operations.

    The handler uses atomic writes (temp file + rename) which should be
    thread-safe on POSIX systems. These tests verify that concurrent
    operations do not corrupt data or cause race conditions.

    Test Coverage:
        - Concurrent stores with different manifest IDs all succeed
        - Concurrent stores with same manifest ID are idempotent
    """

    @pytest.mark.asyncio
    async def test_concurrent_store_operations_are_safe(
        self, handler: HandlerManifestPersistence, temp_storage_path: Path
    ) -> None:
        """Multiple concurrent stores with different IDs should all succeed.

        Validates that the atomic write mechanism correctly handles
        concurrent store operations without data loss or corruption.
        Each manifest should be stored successfully in its own file.
        """
        import asyncio

        # Create 10 manifests with unique IDs
        num_manifests = 10
        manifests = [
            create_test_manifest(node_id=f"concurrent-{i}")
            for i in range(num_manifests)
        ]

        # Store all manifests concurrently
        tasks = [handler.execute(create_store_envelope(m)) for m in manifests]
        results = await asyncio.gather(*tasks)

        # Verify all operations succeeded
        for i, result in enumerate(results):
            assert result.result["status"] == "success", (
                f"Store operation {i} failed: {result.result}"
            )
            assert result.result["payload"]["created"] is True, (
                f"Manifest {i} should have been created"
            )

        # Verify all files were created
        json_files = list(temp_storage_path.rglob("*.json"))
        assert len(json_files) == num_manifests, (
            f"Expected {num_manifests} files, found {len(json_files)}"
        )

        # Verify each manifest can be retrieved correctly
        for manifest in manifests:
            result = await handler.execute(
                create_retrieve_envelope(manifest["manifest_id"])
            )
            assert result.result["status"] == "success"
            assert result.result["payload"]["found"] is True
            retrieved = result.result["payload"]["manifest"]
            assert retrieved["manifest_id"] == manifest["manifest_id"]
            assert (
                retrieved["node_identity"]["node_id"]
                == manifest["node_identity"]["node_id"]
            )

    @pytest.mark.asyncio
    async def test_concurrent_store_same_manifest_is_idempotent(
        self, handler: HandlerManifestPersistence, temp_storage_path: Path
    ) -> None:
        """Concurrent stores of same manifest_id should be idempotent.

        Validates that storing the same manifest_id multiple times concurrently
        results in exactly one file being created, with some operations
        returning created=True (first write) and others returning created=False
        (subsequent writes that find the file already exists).
        """
        import asyncio

        # Create a single manifest
        manifest = create_test_manifest(node_id="idempotent-test")
        manifest_id = manifest["manifest_id"]

        # Attempt to store the same manifest 5 times concurrently
        num_concurrent_stores = 5
        tasks = [
            handler.execute(create_store_envelope(manifest))
            for _ in range(num_concurrent_stores)
        ]
        results = await asyncio.gather(*tasks)

        # All operations should succeed
        for i, result in enumerate(results):
            assert result.result["status"] == "success", (
                f"Store operation {i} failed: {result.result}"
            )
            # All should return the same manifest_id
            assert str(result.result["payload"]["manifest_id"]) == manifest_id

        # Verify idempotent behavior: exactly one created=True, rest created=False
        created_true_count = sum(
            1 for r in results if r.result["payload"]["created"] is True
        )
        created_false_count = sum(
            1 for r in results if r.result["payload"]["created"] is False
        )

        # At least one should have created=True (the first to complete)
        assert created_true_count >= 1, "At least one operation should create the file"

        # Total should equal number of concurrent operations
        assert created_true_count + created_false_count == num_concurrent_stores

        # Verify only one file was created
        json_files = list(temp_storage_path.rglob("*.json"))
        assert len(json_files) == 1, (
            f"Expected 1 file for idempotent writes, found {len(json_files)}"
        )

        # Verify the file contains valid data
        result = await handler.execute(create_retrieve_envelope(manifest_id))
        assert result.result["status"] == "success"
        assert result.result["payload"]["found"] is True
        retrieved = result.result["payload"]["manifest"]
        assert retrieved["manifest_id"] == manifest_id
        assert retrieved["node_identity"]["node_id"] == "idempotent-test"

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations_are_safe(
        self, handler: HandlerManifestPersistence, temp_storage_path: Path
    ) -> None:
        """Mixed concurrent store/retrieve/query operations should be safe.

        Validates that the handler correctly handles concurrent operations
        of different types without deadlocks or data corruption.
        """
        import asyncio

        # First, store some manifests sequentially to ensure they exist
        manifests = [create_test_manifest(node_id=f"mixed-op-{i}") for i in range(5)]
        for manifest in manifests:
            await handler.execute(create_store_envelope(manifest))

        # Now execute mixed operations concurrently
        tasks: list[object] = []

        # Add store operations for new manifests
        new_manifests = [
            create_test_manifest(node_id=f"mixed-new-{i}") for i in range(3)
        ]
        for m in new_manifests:
            tasks.append(handler.execute(create_store_envelope(m)))

        # Add retrieve operations for existing manifests
        for manifest in manifests[:3]:
            tasks.append(
                handler.execute(create_retrieve_envelope(manifest["manifest_id"]))
            )

        # Add query operations
        tasks.append(handler.execute(create_query_envelope(limit=10)))
        tasks.append(handler.execute(create_query_envelope(node_id="mixed-op-0")))

        # Execute all concurrently
        results = await asyncio.gather(*tasks)

        # All operations should succeed
        for i, result in enumerate(results):
            assert result.result["status"] == "success", (
                f"Operation {i} failed: {result.result}"
            )

        # Verify total file count (5 original + 3 new = 8)
        json_files = list(temp_storage_path.rglob("*.json"))
        assert len(json_files) == 8, f"Expected 8 files, found {len(json_files)}"


__all__: list[str] = [
    "TestCoreOperations",
    "TestMetadataOnlyQuery",
    "TestFileBackendSpecifics",
    "TestErrorHandling",
    "TestHandlerLifecycle",
    "TestQueryCombinations",
    "TestCircuitBreakerBehavior",
    "TestConcurrentWrites",
]
