# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for ProjectionReaderContract.

This test suite validates:
- Reader instantiation with asyncpg connection pool
- Contract queries (get_contract_by_id, list_active_contracts, etc.)
- Topic queries (list_topics, get_topic, etc.)
- Cross-reference queries (get_topics_by_contract, get_contracts_by_topic)
- Error handling for database failures
- Circuit breaker integration

Test Organization:
    - TestProjectionReaderContractBasics: Instantiation and configuration
    - TestGetContractById: Contract lookup by ID
    - TestListActiveContracts: Paginated active contracts listing
    - TestListContractsByNodeName: Contracts by node name
    - TestSearchContracts: Search with ILIKE escaping
    - TestCountContractsByStatus: Count by active/inactive
    - TestListTopics: List topics with direction filter
    - TestGetTopic: Single topic lookup
    - TestGetTopicsByContract: Topics for a contract
    - TestGetContractsByTopic: Contracts for a topic
    - TestProjectionReaderContractErrorHandling: Error scenarios
    - TestProjectionReaderContractCircuitBreaker: Circuit breaker behavior

Coverage Goals:
    - >90% code coverage for reader
    - All query paths tested
    - Error handling validated
    - Circuit breaker integration tested

Related Tickets:
    - OMN-1845: Create ProjectionReaderContract for contract/topic queries
    - OMN-1653: Contract registry state materialization
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import asyncpg
import pytest

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    RuntimeHostError,
)
from omnibase_infra.models.projection import (
    ModelContractProjection,
    ModelTopicProjection,
)
from omnibase_infra.projectors.projection_reader_contract import (
    ProjectionReaderContract,
)


def create_mock_contract_row(
    contract_id: str = "test-node:1.0.0",
    node_name: str = "test-node",
    version_major: int = 1,
    version_minor: int = 0,
    version_patch: int = 0,
    contract_hash: str = "abc123def456",
    contract_yaml: str = "name: test-node\nversion: 1.0.0",
    is_active: bool = True,
    deregistered_at: datetime | None = None,
) -> dict:
    """Create a mock database row for contract projection with sensible defaults."""
    now = datetime.now(UTC)
    return {
        "contract_id": contract_id,
        "node_name": node_name,
        "version_major": version_major,
        "version_minor": version_minor,
        "version_patch": version_patch,
        "contract_hash": contract_hash,
        "contract_yaml": contract_yaml,
        "registered_at": now,
        "deregistered_at": deregistered_at,
        "last_seen_at": now,
        "is_active": is_active,
        "last_event_topic": "test.topic.v1",
        "last_event_partition": 0,
        "last_event_offset": 100,
        "created_at": now,
        "updated_at": now,
    }


def create_mock_topic_row(
    topic_suffix: str = "onex.evt.platform.contract-registered.v1",
    direction: str = "publish",
    contract_ids: list[str] | None = None,
    is_active: bool = True,
) -> dict:
    """Create a mock database row for topic projection with sensible defaults."""
    now = datetime.now(UTC)
    return {
        "topic_suffix": topic_suffix,
        "direction": direction,
        "contract_ids": contract_ids or ["test-node:1.0.0"],
        "first_seen_at": now,
        "last_seen_at": now,
        "is_active": is_active,
        "created_at": now,
        "updated_at": now,
    }


@pytest.fixture
def mock_pool() -> MagicMock:
    """Create a mock asyncpg connection pool."""
    pool = MagicMock(spec=asyncpg.Pool)
    return pool


@pytest.fixture
def mock_connection() -> AsyncMock:
    """Create a mock asyncpg connection."""
    conn = AsyncMock()
    return conn


@pytest.fixture
def reader(mock_pool: MagicMock) -> ProjectionReaderContract:
    """Create a ProjectionReaderContract instance with mocked pool."""
    return ProjectionReaderContract(pool=mock_pool)


@pytest.mark.unit
class TestProjectionReaderContractBasics:
    """Test basic reader instantiation and configuration."""

    def test_reader_instantiation(self, mock_pool: MagicMock) -> None:
        """Test that reader initializes correctly with connection pool."""
        reader = ProjectionReaderContract(pool=mock_pool)

        assert reader._pool is mock_pool
        # Verify circuit breaker is initialized
        assert hasattr(reader, "_circuit_breaker_lock")
        assert reader._circuit_breaker_failures == 0
        assert reader._circuit_breaker_open is False

    def test_reader_circuit_breaker_config(
        self, reader: ProjectionReaderContract
    ) -> None:
        """Test that circuit breaker is configured correctly."""
        # Default config: threshold=5, reset_timeout=60.0
        assert reader.circuit_breaker_threshold == 5
        assert reader.circuit_breaker_reset_timeout == 60.0
        assert reader.service_name == "projection_reader.contract"

    def test_row_to_contract_projection_conversion(
        self, reader: ProjectionReaderContract
    ) -> None:
        """Test internal row to contract projection conversion."""
        mock_row = create_mock_contract_row()

        projection = reader._row_to_contract_projection(mock_row)

        assert isinstance(projection, ModelContractProjection)
        assert projection.contract_id == mock_row["contract_id"]
        assert projection.node_name == mock_row["node_name"]
        assert projection.version_major == mock_row["version_major"]
        assert projection.version_minor == mock_row["version_minor"]
        assert projection.version_patch == mock_row["version_patch"]
        assert projection.contract_hash == mock_row["contract_hash"]
        assert projection.is_active is True

    def test_row_to_topic_projection_conversion(
        self, reader: ProjectionReaderContract
    ) -> None:
        """Test internal row to topic projection conversion."""
        mock_row = create_mock_topic_row()

        projection = reader._row_to_topic_projection(mock_row)

        assert isinstance(projection, ModelTopicProjection)
        assert projection.topic_suffix == mock_row["topic_suffix"]
        assert projection.direction == mock_row["direction"]
        assert projection.contract_ids == mock_row["contract_ids"]
        assert projection.is_active is True

    def test_row_to_topic_projection_with_string_json(
        self, reader: ProjectionReaderContract
    ) -> None:
        """Test row to topic conversion when contract_ids is JSON string."""
        mock_row = create_mock_topic_row()
        # Simulate asyncpg returning JSONB as string
        mock_row["contract_ids"] = '["node-a:1.0.0", "node-b:2.0.0"]'

        projection = reader._row_to_topic_projection(mock_row)

        assert isinstance(projection, ModelTopicProjection)
        assert projection.contract_ids == ["node-a:1.0.0", "node-b:2.0.0"]

    def test_row_to_topic_projection_with_invalid_json(
        self, reader: ProjectionReaderContract
    ) -> None:
        """Test row to topic conversion handles invalid JSON gracefully."""
        mock_row = create_mock_topic_row()
        # Simulate invalid JSON string
        mock_row["contract_ids"] = "not valid json"

        projection = reader._row_to_topic_projection(mock_row)

        # Should use empty list when JSON parsing fails
        assert isinstance(projection, ModelTopicProjection)
        assert projection.contract_ids == []


# =============================================================================
# Contract Query Method Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetContractById:
    """Test get_contract_by_id method."""

    async def test_get_contract_by_id_success(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_contract_by_id returns projection when found."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_row = create_mock_contract_row(contract_id="my-node:1.0.0")
        mock_connection.fetchrow.return_value = mock_row

        result = await reader.get_contract_by_id("my-node:1.0.0")

        assert result is not None
        assert isinstance(result, ModelContractProjection)
        assert result.contract_id == "my-node:1.0.0"
        mock_connection.fetchrow.assert_called_once()

    async def test_get_contract_by_id_not_found(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_contract_by_id returns None when not found."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = None

        result = await reader.get_contract_by_id("nonexistent:1.0.0")

        assert result is None

    async def test_get_contract_by_id_with_correlation_id(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_contract_by_id propagates correlation ID."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = create_mock_contract_row()

        correlation_id = uuid4()
        await reader.get_contract_by_id("test:1.0.0", correlation_id=correlation_id)

        # Should not raise - correlation ID used for tracing

    async def test_get_contract_by_id_connection_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
    ) -> None:
        """Test get_contract_by_id handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.get_contract_by_id("test:1.0.0")

        assert "Failed to connect" in str(exc_info.value)

    async def test_get_contract_by_id_timeout_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_contract_by_id handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.side_effect = asyncpg.QueryCanceledError("timeout")

        with pytest.raises(InfraTimeoutError) as exc_info:
            await reader.get_contract_by_id("test:1.0.0")

        assert "timed out" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
class TestListActiveContracts:
    """Test list_active_contracts method."""

    async def test_list_active_contracts_success(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_active_contracts returns list of projections."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_contract_row(contract_id="node-a:1.0.0"),
            create_mock_contract_row(contract_id="node-b:1.0.0"),
            create_mock_contract_row(contract_id="node-c:1.0.0"),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.list_active_contracts()

        assert isinstance(result, list)
        assert len(result) == 3
        for proj in result:
            assert isinstance(proj, ModelContractProjection)

    async def test_list_active_contracts_empty(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_active_contracts returns empty list when no contracts."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        result = await reader.list_active_contracts()

        assert result == []

    async def test_list_active_contracts_respects_pagination(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_active_contracts respects limit and offset parameters."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        await reader.list_active_contracts(limit=50, offset=10)

        # Verify pagination parameters were passed to query
        call_args = mock_connection.fetch.call_args
        assert call_args is not None
        args = call_args[0]
        assert 50 in args  # limit
        assert 10 in args  # offset

    async def test_list_active_contracts_connection_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
    ) -> None:
        """Test list_active_contracts handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.list_active_contracts()

        assert "Failed to connect" in str(exc_info.value)

    async def test_list_active_contracts_timeout_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_active_contracts handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = asyncpg.QueryCanceledError("timeout")

        with pytest.raises(InfraTimeoutError) as exc_info:
            await reader.list_active_contracts()

        assert "timed out" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
class TestListContractsByNodeName:
    """Test list_contracts_by_node_name method."""

    async def test_list_contracts_by_node_name_success(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_contracts_by_node_name returns versions of a node."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_contract_row(
                contract_id="my-node:2.0.0",
                node_name="my-node",
                version_major=2,
            ),
            create_mock_contract_row(
                contract_id="my-node:1.0.0",
                node_name="my-node",
                version_major=1,
            ),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.list_contracts_by_node_name("my-node")

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(proj.node_name == "my-node" for proj in result)

    async def test_list_contracts_by_node_name_not_found(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_contracts_by_node_name returns empty list when not found."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        result = await reader.list_contracts_by_node_name("nonexistent-node")

        assert result == []

    async def test_list_contracts_by_node_name_include_inactive(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_contracts_by_node_name with include_inactive flag."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_contract_row(contract_id="my-node:2.0.0", is_active=True),
            create_mock_contract_row(contract_id="my-node:1.0.0", is_active=False),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.list_contracts_by_node_name(
            "my-node", include_inactive=True
        )

        assert len(result) == 2
        # Verify SQL query does NOT filter by is_active when include_inactive=True
        call_args = mock_connection.fetch.call_args
        sql = call_args[0][0]
        assert "is_active = TRUE" not in sql or "include_inactive" in sql

    async def test_list_contracts_by_node_name_connection_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
    ) -> None:
        """Test list_contracts_by_node_name handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.list_contracts_by_node_name("my-node")

        assert "Failed to connect" in str(exc_info.value)

    async def test_list_contracts_by_node_name_timeout_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_contracts_by_node_name handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = asyncpg.QueryCanceledError("timeout")

        with pytest.raises(InfraTimeoutError) as exc_info:
            await reader.list_contracts_by_node_name("my-node")

        assert "timed out" in str(exc_info.value)

    async def test_list_contracts_by_node_name_generic_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_contracts_by_node_name handles generic errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = Exception("Unknown error")

        with pytest.raises(RuntimeHostError) as exc_info:
            await reader.list_contracts_by_node_name("my-node")

        assert "Failed to list contracts by node name" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
class TestSearchContracts:
    """Test search_contracts method."""

    async def test_search_contracts_success(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test search_contracts returns matching contracts."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_contract_row(contract_id="registry-effect:1.0.0"),
            create_mock_contract_row(contract_id="registry-reducer:1.0.0"),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.search_contracts("registry")

        assert isinstance(result, list)
        assert len(result) == 2

    async def test_search_contracts_no_results(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test search_contracts returns empty list when no matches."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        result = await reader.search_contracts("nonexistent")

        assert result == []

    async def test_search_contracts_escapes_ilike_metacharacters(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test search_contracts escapes % and _ for ILIKE safety."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        # Search with metacharacters that should be escaped
        await reader.search_contracts("test%node_name")

        # Verify escaped query was passed
        call_args = mock_connection.fetch.call_args
        args = call_args[0]
        # The escaped query should be in the args
        escaped_query = args[1]  # Second arg after SQL
        assert r"\%" in escaped_query
        assert r"\_" in escaped_query

    async def test_search_contracts_respects_limit(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test search_contracts respects limit parameter."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        await reader.search_contracts("test", limit=25)

        call_args = mock_connection.fetch.call_args
        args = call_args[0]
        assert 25 in args

    async def test_search_contracts_connection_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
    ) -> None:
        """Test search_contracts handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.search_contracts("test")

        assert "Failed to connect" in str(exc_info.value)

    async def test_search_contracts_timeout_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test search_contracts handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = asyncpg.QueryCanceledError("timeout")

        with pytest.raises(InfraTimeoutError) as exc_info:
            await reader.search_contracts("test")

        assert "timed out" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
class TestCountContractsByStatus:
    """Test count_contracts_by_status method."""

    async def test_count_contracts_by_status_success(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test count_contracts_by_status returns correct counts."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            {"is_active": True, "count": 10},
            {"is_active": False, "count": 5},
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.count_contracts_by_status()

        assert isinstance(result, dict)
        assert result["active"] == 10
        assert result["inactive"] == 5

    async def test_count_contracts_by_status_empty(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test count_contracts_by_status returns zeros when no contracts."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        result = await reader.count_contracts_by_status()

        assert result == {"active": 0, "inactive": 0}

    async def test_count_contracts_by_status_only_active(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test count_contracts_by_status with only active contracts."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            {"is_active": True, "count": 15},
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.count_contracts_by_status()

        assert result["active"] == 15
        assert result["inactive"] == 0

    async def test_count_contracts_by_status_connection_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
    ) -> None:
        """Test count_contracts_by_status handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.count_contracts_by_status()

        assert "Failed to connect" in str(exc_info.value)

    async def test_count_contracts_by_status_timeout_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test count_contracts_by_status handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = asyncpg.QueryCanceledError("timeout")

        with pytest.raises(InfraTimeoutError) as exc_info:
            await reader.count_contracts_by_status()

        assert "timed out" in str(exc_info.value)


# =============================================================================
# Topic Query Method Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestListTopics:
    """Test list_topics method."""

    async def test_list_topics_success(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_topics returns list of topic projections."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_topic_row(topic_suffix="topic.a.v1", direction="publish"),
            create_mock_topic_row(topic_suffix="topic.b.v1", direction="subscribe"),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.list_topics()

        assert isinstance(result, list)
        assert len(result) == 2
        for proj in result:
            assert isinstance(proj, ModelTopicProjection)

    async def test_list_topics_empty(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_topics returns empty list when no topics."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        result = await reader.list_topics()

        assert result == []

    async def test_list_topics_with_direction_filter(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_topics filters by direction."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_topic_row(direction="publish"),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.list_topics(direction="publish")

        assert len(result) == 1
        # Verify direction filter was passed to query
        call_args = mock_connection.fetch.call_args
        args = call_args[0]
        assert "publish" in args

    async def test_list_topics_respects_pagination(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_topics respects limit and offset."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        await reader.list_topics(limit=50, offset=20)

        call_args = mock_connection.fetch.call_args
        args = call_args[0]
        assert 50 in args
        assert 20 in args

    async def test_list_topics_connection_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
    ) -> None:
        """Test list_topics handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.list_topics()

        assert "Failed to connect" in str(exc_info.value)

    async def test_list_topics_timeout_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_topics handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = asyncpg.QueryCanceledError("timeout")

        with pytest.raises(InfraTimeoutError) as exc_info:
            await reader.list_topics()

        assert "timed out" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetTopic:
    """Test get_topic method."""

    async def test_get_topic_success(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_topic returns topic projection when found."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_row = create_mock_topic_row(
            topic_suffix="onex.evt.platform.contract-registered.v1",
            direction="publish",
        )
        mock_connection.fetchrow.return_value = mock_row

        result = await reader.get_topic(
            "onex.evt.platform.contract-registered.v1",
            "publish",
        )

        assert result is not None
        assert isinstance(result, ModelTopicProjection)
        assert result.topic_suffix == "onex.evt.platform.contract-registered.v1"
        assert result.direction == "publish"

    async def test_get_topic_not_found(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_topic returns None when not found."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = None

        result = await reader.get_topic("nonexistent.topic.v1", "publish")

        assert result is None

    async def test_get_topic_connection_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
    ) -> None:
        """Test get_topic handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.get_topic("topic.v1", "publish")

        assert "Failed to connect" in str(exc_info.value)

    async def test_get_topic_timeout_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_topic handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.side_effect = asyncpg.QueryCanceledError("timeout")

        with pytest.raises(InfraTimeoutError) as exc_info:
            await reader.get_topic("topic.v1", "publish")

        assert "timed out" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetTopicsByContract:
    """Test get_topics_by_contract method."""

    async def test_get_topics_by_contract_success(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_topics_by_contract returns topics for a contract."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_topic_row(
                topic_suffix="topic.output.v1",
                direction="publish",
                contract_ids=["my-node:1.0.0"],
            ),
            create_mock_topic_row(
                topic_suffix="topic.input.v1",
                direction="subscribe",
                contract_ids=["my-node:1.0.0"],
            ),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_topics_by_contract("my-node:1.0.0")

        assert isinstance(result, list)
        assert len(result) == 2
        for proj in result:
            assert isinstance(proj, ModelTopicProjection)

    async def test_get_topics_by_contract_not_found(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_topics_by_contract returns empty list when no topics."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        result = await reader.get_topics_by_contract("nonexistent:1.0.0")

        assert result == []

    async def test_get_topics_by_contract_connection_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
    ) -> None:
        """Test get_topics_by_contract handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.get_topics_by_contract("my-node:1.0.0")

        assert "Failed to connect" in str(exc_info.value)

    async def test_get_topics_by_contract_timeout_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_topics_by_contract handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = asyncpg.QueryCanceledError("timeout")

        with pytest.raises(InfraTimeoutError) as exc_info:
            await reader.get_topics_by_contract("my-node:1.0.0")

        assert "timed out" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetContractsByTopic:
    """Test get_contracts_by_topic method."""

    async def test_get_contracts_by_topic_success(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_contracts_by_topic returns contracts for a topic."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None

        mock_rows = [
            create_mock_contract_row(contract_id="node-a:1.0.0"),
            create_mock_contract_row(contract_id="node-b:1.0.0"),
        ]
        mock_connection.fetch.return_value = mock_rows

        result = await reader.get_contracts_by_topic(
            "onex.evt.platform.contract-registered.v1"
        )

        assert isinstance(result, list)
        assert len(result) == 2
        for proj in result:
            assert isinstance(proj, ModelContractProjection)

    async def test_get_contracts_by_topic_not_found(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_contracts_by_topic returns empty list when no contracts."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.return_value = []

        result = await reader.get_contracts_by_topic("nonexistent.topic.v1")

        assert result == []

    async def test_get_contracts_by_topic_connection_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
    ) -> None:
        """Test get_contracts_by_topic handles connection errors."""
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            await reader.get_contracts_by_topic("topic.v1")

        assert "Failed to connect" in str(exc_info.value)

    async def test_get_contracts_by_topic_timeout_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_contracts_by_topic handles timeout errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = asyncpg.QueryCanceledError("timeout")

        with pytest.raises(InfraTimeoutError) as exc_info:
            await reader.get_contracts_by_topic("topic.v1")

        assert "timed out" in str(exc_info.value)


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectionReaderContractErrorHandling:
    """Test error handling for database failures."""

    async def test_generic_exception_raises_runtime_host_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test generic exceptions are wrapped in RuntimeHostError."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.side_effect = Exception("Unknown database error")

        with pytest.raises(RuntimeHostError) as exc_info:
            await reader.get_contract_by_id("test:1.0.0")

        assert "Failed to get contract by ID" in str(exc_info.value)

    async def test_list_active_contracts_generic_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_active_contracts wraps generic errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = Exception("Unknown error")

        with pytest.raises(RuntimeHostError) as exc_info:
            await reader.list_active_contracts()

        assert "Failed to list active contracts" in str(exc_info.value)

    async def test_search_contracts_generic_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test search_contracts wraps generic errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = Exception("Unknown error")

        with pytest.raises(RuntimeHostError) as exc_info:
            await reader.search_contracts("test")

        assert "Failed to search contracts" in str(exc_info.value)

    async def test_count_contracts_by_status_generic_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test count_contracts_by_status wraps generic errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = Exception("Unknown error")

        with pytest.raises(RuntimeHostError) as exc_info:
            await reader.count_contracts_by_status()

        assert "Failed to count contracts" in str(exc_info.value)

    async def test_list_topics_generic_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test list_topics wraps generic errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = Exception("Unknown error")

        with pytest.raises(RuntimeHostError) as exc_info:
            await reader.list_topics()

        assert "Failed to list topics" in str(exc_info.value)

    async def test_get_topic_generic_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_topic wraps generic errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.side_effect = Exception("Unknown error")

        with pytest.raises(RuntimeHostError) as exc_info:
            await reader.get_topic("topic.v1", "publish")

        assert "Failed to get topic" in str(exc_info.value)

    async def test_get_topics_by_contract_generic_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_topics_by_contract wraps generic errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = Exception("Unknown error")

        with pytest.raises(RuntimeHostError) as exc_info:
            await reader.get_topics_by_contract("my-node:1.0.0")

        assert "Failed to get topics by contract" in str(exc_info.value)

    async def test_get_contracts_by_topic_generic_error(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test get_contracts_by_topic wraps generic errors."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetch.side_effect = Exception("Unknown error")

        with pytest.raises(RuntimeHostError) as exc_info:
            await reader.get_contracts_by_topic("topic.v1")

        assert "Failed to get contracts by topic" in str(exc_info.value)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestProjectionReaderContractCircuitBreaker:
    """Test circuit breaker behavior."""

    async def test_circuit_breaker_opens_after_threshold_failures(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test circuit breaker opens after threshold failures."""
        reader = ProjectionReaderContract(pool=mock_pool)

        # Simulate connection failures to reach threshold
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        # Make 5 failed calls (default threshold)
        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await reader.get_contract_by_id("test:1.0.0")

        # Circuit should now be open
        assert reader._circuit_breaker_open is True
        assert reader._circuit_breaker_failures >= 5

    async def test_circuit_breaker_blocks_when_open(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test circuit breaker blocks operations when open."""
        reader = ProjectionReaderContract(pool=mock_pool)

        # Simulate connection failures to open circuit
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        # Exhaust threshold
        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await reader.get_contract_by_id("test:1.0.0")

        # Next call should be blocked by circuit breaker
        with pytest.raises(InfraUnavailableError) as exc_info:
            await reader.get_contract_by_id("test:1.0.0")

        assert "Circuit breaker is open" in str(exc_info.value)

    async def test_circuit_breaker_resets_on_success(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test circuit breaker resets after successful operation."""
        # First, simulate a failure
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError):
            await reader.get_contract_by_id("test:1.0.0")

        assert reader._circuit_breaker_failures == 1

        # Now simulate success
        mock_pool.acquire.return_value.__aenter__.side_effect = None
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.return_value = create_mock_contract_row()

        await reader.get_contract_by_id("test:1.0.0")

        # Circuit breaker should be reset
        assert reader._circuit_breaker_failures == 0
        assert reader._circuit_breaker_open is False

    async def test_circuit_breaker_blocks_all_operations(
        self,
        mock_pool: MagicMock,
    ) -> None:
        """Test circuit breaker blocks all reader operations when open."""
        reader = ProjectionReaderContract(pool=mock_pool)

        # Open circuit breaker
        mock_pool.acquire.return_value.__aenter__.side_effect = (
            asyncpg.PostgresConnectionError("Connection refused")
        )

        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await reader.get_contract_by_id("test:1.0.0")

        # All operations should be blocked
        with pytest.raises(InfraUnavailableError):
            await reader.get_contract_by_id("test:1.0.0")

        with pytest.raises(InfraUnavailableError):
            await reader.list_active_contracts()

        with pytest.raises(InfraUnavailableError):
            await reader.list_contracts_by_node_name("my-node")

        with pytest.raises(InfraUnavailableError):
            await reader.search_contracts("test")

        with pytest.raises(InfraUnavailableError):
            await reader.count_contracts_by_status()

        with pytest.raises(InfraUnavailableError):
            await reader.list_topics()

        with pytest.raises(InfraUnavailableError):
            await reader.get_topic("topic.v1", "publish")

        with pytest.raises(InfraUnavailableError):
            await reader.get_topics_by_contract("my-node:1.0.0")

        with pytest.raises(InfraUnavailableError):
            await reader.get_contracts_by_topic("topic.v1")

    async def test_timeout_error_increments_circuit_breaker(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that timeout errors also increment circuit breaker failures."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.side_effect = asyncpg.QueryCanceledError("timeout")

        with pytest.raises(InfraTimeoutError):
            await reader.get_contract_by_id("test:1.0.0")

        assert reader._circuit_breaker_failures == 1

    async def test_generic_error_increments_circuit_breaker(
        self,
        reader: ProjectionReaderContract,
        mock_pool: MagicMock,
        mock_connection: AsyncMock,
    ) -> None:
        """Test that generic errors also increment circuit breaker failures."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        mock_connection.fetchrow.side_effect = Exception("Unknown error")

        with pytest.raises(RuntimeHostError):
            await reader.get_contract_by_id("test:1.0.0")

        assert reader._circuit_breaker_failures == 1
