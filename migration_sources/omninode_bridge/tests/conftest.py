"""
Simplified pytest configuration for OmniNode Bridge tests.

This configuration provides essential test functionality without the
complex auto-use fixtures that were causing timeout issues.
"""

import asyncio
import contextlib
import logging
import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

# Enhanced logging for test debugging
logging.basicConfig(level=logging.WARNING)
test_logger = logging.getLogger("test_simplified")

# Import centralized test environment configuration (if available)
try:
    from .test_env_config import (
        MockedTestEnvironment,
        ensure_test_mode,
        get_test_config,
    )

    TEST_ENV_CONFIG_AVAILABLE = True
    test_logger.info("Test environment configuration available")
except ImportError as e:
    TEST_ENV_CONFIG_AVAILABLE = False
    test_logger.warning(f"Test environment configuration not available: {e}")

# Import hook models for test data factories
try:
    from omninode_bridge.models.hooks import HookEvent, HookMetadata, HookPayload

    HOOK_MODELS_AVAILABLE = True
except ImportError as e:
    HOOK_MODELS_AVAILABLE = False
    test_logger.warning(f"Hook models not available: {e}")

# ============================================================================
# Test Configuration
# ============================================================================


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration with basic settings."""
    return {
        "test_timeout": 30,
        "container_startup_timeout": 60,
        "kafka_topics": [
            "test.hook.events.v1",
            "test.workflow.events.v1",
            "test.service.events.v1",
            "test.tool.events.v1",
        ],
        "postgres_db": "test_omninode_bridge",
        "test_data_size": {
            "small": 100,
            "medium": 1000,
            "large": 10000,
        },
    }


@pytest.fixture(scope="session")
def remote_test_config():
    """
    Remote test configuration for integration tests.

    Supports both local (Docker Compose) and remote (distributed system) testing.
    Loads configuration from .env (local) or remote.env (remote) based on TEST_MODE.

    Returns:
        TestConfiguration instance with all service endpoints configured

    Usage:
        def test_kafka_integration(remote_test_config):
            kafka_servers = remote_test_config.kafka_bootstrap_servers
            # Use kafka_servers for consumer/producer
    """
    try:
        from tests.integration.remote_config import get_test_config

        config = get_test_config()
        test_logger.info(
            f"Loaded remote test config: mode={config.test_mode}, "
            f"kafka={config.kafka_bootstrap_servers}, "
            f"postgres={config.postgres_host}:{config.postgres_port}"
        )
        return config
    except ImportError as e:
        test_logger.warning(f"Remote test config not available: {e}")
        pytest.skip("Remote test configuration not available")


# ============================================================================
# Basic Service Fixtures (Non-Auto)
# ============================================================================


@pytest.fixture()
def mock_postgres_client():
    """Lightweight mock PostgreSQL client."""
    mock_client = AsyncMock()

    # Minimal configuration for performance
    mock_client.is_connected = True
    mock_client.connect.return_value = None
    mock_client.disconnect.return_value = None
    mock_client.health_check.return_value = {
        "status": "healthy",
        "connected": True,
    }

    return mock_client


@pytest.fixture()
def mock_kafka_client():
    """Lightweight mock Kafka client."""
    mock_client = AsyncMock()

    # Minimal configuration for performance
    mock_client.is_connected = True
    mock_client.connect.return_value = None
    mock_client.disconnect.return_value = None
    mock_client.health_check.return_value = {
        "status": "healthy",
        "connected": True,
    }

    return mock_client


@pytest.fixture()
def mocked_test_environment(request):
    """Comprehensive mocked test environment (only when explicitly requested)."""
    if not TEST_ENV_CONFIG_AVAILABLE:
        pytest.skip("Test environment configuration not available")

    test_name = request.node.name

    with MockedTestEnvironment(test_name) as mocked_env:
        test_logger.debug(
            f"Mocked environment setup for {test_name} with test_id: {mocked_env.test_id}"
        )
        yield {
            "test_id": mocked_env.test_id,
            "config": mocked_env.config,
            "database_url": mocked_env.config.get_database_url(),
            "kafka_bootstrap_servers": mocked_env.config.get_kafka_bootstrap_servers(),
        }


@pytest.fixture()
def isolated_test_environment(request):
    """Isolated test environment (only when explicitly requested)."""
    if not TEST_ENV_CONFIG_AVAILABLE:
        pytest.skip("Test environment configuration not available")

    test_name = request.node.name
    test_config = get_test_config(test_name, isolated=True)

    with test_config:
        test_logger.debug(
            f"Isolated environment setup for {test_name} with test_id: {test_config.test_id}"
        )

        # Ensure we're in proper test mode for integration tests
        try:
            ensure_test_mode()
        except RuntimeError as e:
            pytest.fail(f"Test mode validation failed: {e}")

        yield {
            "test_id": test_config.test_id,
            "config": test_config,
            "database_url": test_config.get_database_url(),
            "kafka_bootstrap_servers": test_config.get_kafka_bootstrap_servers(),
            "test_env_vars": test_config.get_env_vars(),
        }


# ============================================================================
# Test Data Factories
# ============================================================================


@pytest.fixture()
def hook_event_factory():
    """Factory for creating test hook events."""
    if not HOOK_MODELS_AVAILABLE:
        pytest.skip("Hook models not available")

    def create_hook_event(
        source: str = None,
        action: str = None,
        resource: str = None,
        resource_id: str = None,
        data: dict[str, Any] = None,
        correlation_id: uuid.UUID = None,
    ) -> HookEvent:
        # Generate defaults only when needed
        source = source or f"test-service-{uuid.uuid4().hex[:8]}"
        action = action or "startup"
        resource = resource or "service"
        resource_id = resource_id or f"instance-{uuid.uuid4().hex[:8]}"
        data = data or {"status": "ready", "version": "1.0.0"}
        correlation_id = correlation_id or uuid.uuid4()

        metadata = HookMetadata(
            source=source,
            version="1.0.0",
            environment="dev",
            correlation_id=correlation_id,
        )
        payload = HookPayload(
            action=action,
            resource=resource,
            resource_id=resource_id,
            data=data.copy() if data else {},
        )
        return HookEvent(metadata=metadata, payload=payload)

    return create_hook_event


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture()
def async_test_utils():
    """Async test utilities."""

    class AsyncTestUtils:
        @staticmethod
        async def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
            start_time = asyncio.get_running_loop().time()
            while asyncio.get_running_loop().time() - start_time < timeout:
                if (
                    await condition_func()
                    if asyncio.iscoroutinefunction(condition_func)
                    else condition_func()
                ):
                    return True
                await asyncio.sleep(interval)
            return False

    return AsyncTestUtils()


# Security test data
@pytest.fixture()
def security_test_data():
    """Security test data for validation testing."""
    return {
        "sql_injection": ["'; DROP TABLE users; --", "1' OR '1'='1"],
        "xss_payloads": ["<script>alert('xss')</script>", "javascript:alert('xss')"],
    }


# ============================================================================
# SQL Injection Test Fixtures
# ============================================================================


@pytest.fixture(scope="function")
async def sql_injection_test_db():
    """
    PostgreSQL testcontainer for SQL injection tests.

    Reuses the postgres_container fixture from e2e tests to avoid duplication.
    Provides a clean PostgreSQL 16 database for security testing.

    Returns:
        dict: Connection details including:
            - connection_url: Full PostgreSQL connection URL
            - host: Container host IP
            - port: Exposed PostgreSQL port
            - database: Database name
            - username: PostgreSQL username
            - password: PostgreSQL password

    Note:
        - Function scope for test isolation (clean state per test)
        - Automatic cleanup in finally block
        - Falls back to local PostgreSQL if testcontainers unavailable
    """
    try:
        from testcontainers.postgres import PostgresContainer

        TESTCONTAINERS_AVAILABLE = True
    except ImportError:
        TESTCONTAINERS_AVAILABLE = False

    # Detect CI environment
    import os

    IS_CI = os.getenv("CI", "false").lower() == "true"
    USE_TESTCONTAINERS = os.getenv("USE_TESTCONTAINERS", str(IS_CI)).lower() == "true"

    if USE_TESTCONTAINERS and TESTCONTAINERS_AVAILABLE:
        # Use testcontainers for CI/E2E testing (PostgreSQL 16)
        container = PostgresContainer("postgres:16")
        container.start()

        try:
            connection_url = container.get_connection_url()
            yield {
                "connection_url": connection_url,
                "host": container.get_container_host_ip(),
                "port": container.get_exposed_port(5432),
                "database": container.dbname,
                "username": container.username,
                "password": container.password,
            }
        finally:
            container.stop()
    else:
        # Use local PostgreSQL for development
        yield {
            "connection_url": os.getenv(
                "DATABASE_URL",
                "postgresql://postgres:omninode-bridge-postgres-dev-2024@localhost:5436/omninode_bridge",
            ),
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5436")),
            "database": "omninode_bridge",
            "username": "postgres",
            "password": "omninode-bridge-postgres-dev-2024",
        }


@pytest.fixture
async def database_adapter_node(sql_injection_test_db):
    """
    Create database adapter node instance for SQL injection testing.

    This fixture:
    1. Uses sql_injection_test_db for database connection
    2. Creates a test node with GenericCRUDHandlers mixin
    3. Sets up necessary database tables
    4. Provides a ready-to-test node instance with CRUD methods
    5. Cleans up after tests complete

    Args:
        sql_injection_test_db: PostgreSQL test database fixture

    Returns:
        Test node with CRUD handler methods for SQL injection testing

    Note:
        Function scope ensures clean state for each test
    """
    import asyncpg

    from omninode_bridge.nodes.database_adapter_effect.v1_0_0._generic_crud_handlers import (
        GenericCRUDHandlers,
    )
    from omninode_bridge.nodes.database_adapter_effect.v1_0_0.circuit_breaker import (
        DatabaseCircuitBreaker,
    )

    # Get database connection details
    connection_url = sql_injection_test_db["connection_url"]

    # Create connection pool
    pool = await asyncpg.create_pool(
        connection_url,
        min_size=2,
        max_size=5,
        command_timeout=10,
    )

    try:
        # Create necessary database tables for testing
        async with pool.acquire() as conn:
            # Create workflow_executions table
            # Enable uuid-ossp extension for UUID generation
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    workflow_type TEXT NOT NULL,
                    correlation_id UUID NOT NULL,
                    current_state TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    execution_time_ms INTEGER,
                    error_message TEXT,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """
            )

            # Create metadata_stamps table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata_stamps (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    file_hash TEXT NOT NULL UNIQUE,
                    file_path TEXT,
                    namespace TEXT NOT NULL,
                    stamp_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """
            )

        # Create simple query executor for testing
        class TestQueryExecutor:
            """Simple query executor for SQL injection tests."""

            def __init__(self, db_pool):
                self._pool = db_pool

            async def execute_query(self, query: str, *params):
                """Execute a SELECT query and return rows."""
                async with self._pool.acquire() as conn:
                    return await conn.fetch(query, *params)

            async def execute_insert(self, query: str, *params):
                """Execute an INSERT query and return generated ID."""
                async with self._pool.acquire() as conn:
                    row = await conn.fetchrow(query, *params)
                    return row["id"] if row and "id" in row else None

            async def execute_update(self, query: str, *params):
                """Execute an UPDATE query and return rows affected."""
                async with self._pool.acquire() as conn:
                    result = await conn.execute(query, *params)
                    # Parse "UPDATE N" to get number of rows
                    return int(result.split()[-1]) if result else 0

            async def execute_delete(self, query: str, *params):
                """Execute a DELETE query and return rows affected."""
                async with self._pool.acquire() as conn:
                    result = await conn.execute(query, *params)
                    # Parse "DELETE N" to get number of rows
                    return int(result.split()[-1]) if result else 0

        # Create mock logger with required methods
        class MockLogger:
            """Mock logger for SQL injection testing."""

            def log_operation_complete(self, *args, **kwargs):
                """Mock log_operation_complete method."""
                pass

            def log_operation_warning(self, *args, **kwargs):
                """Mock log_operation_warning method."""
                pass

        # Create mock connection manager for batch operations
        class MockConnectionManager:
            """Mock connection manager for testing batch operations."""

            def __init__(self, db_pool):
                self.pool = db_pool

            @contextlib.asynccontextmanager
            async def transaction(
                self,
                isolation: str = "read_committed",
                readonly: bool = False,
                deferrable: bool = False,
            ):
                """Mock transaction context manager."""
                async with self.pool.acquire() as conn:
                    async with conn.transaction():
                        yield conn

        # Create test node class with CRUD handlers
        class TestDatabaseAdapterNode(GenericCRUDHandlers):
            """Test node with CRUD handler methods for SQL injection testing."""

            def __init__(self, db_pool, query_executor):
                self._connection_pool = db_pool
                self._query_executor = query_executor
                self._connection_manager = MockConnectionManager(db_pool)
                self._logger = MockLogger()
                self._circuit_breaker = DatabaseCircuitBreaker(
                    failure_threshold=5,
                    timeout_seconds=30,
                    half_open_max_calls=3,
                    half_open_success_threshold=2,
                )

        # Initialize query executor and test node
        query_executor = TestQueryExecutor(pool)
        node = TestDatabaseAdapterNode(pool, query_executor)

        yield node

        # Cleanup: Drop test tables
        async with pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS workflow_executions CASCADE;")
            await conn.execute("DROP TABLE IF EXISTS metadata_stamps CASCADE;")

    finally:
        await pool.close()


# ============================================================================
# Test Collection Optimization (Simplified)
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """Simplified test collection optimization."""
    # Categorize tests by expected performance
    fast_tests = []
    slow_tests = []
    integration_tests = []

    for item in items:
        # Auto-mark tests based on patterns
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            integration_tests.append(item)
        elif any(keyword in item.name.lower() for keyword in ["mock", "unit"]):
            item.add_marker(pytest.mark.fast)
            fast_tests.append(item)
        else:
            slow_tests.append(item)

    # Simple reordering: fast -> slow -> integration
    items[:] = fast_tests + slow_tests + integration_tests

    test_logger.info(
        f"Test execution order: {len(fast_tests)} fast, "
        f"{len(slow_tests)} slow, {len(integration_tests)} integration"
    )


# ============================================================================
# Entity Model Fixtures (Pydantic v2)
# ============================================================================


@pytest.fixture
def sample_correlation_id() -> uuid.UUID:
    """Generate a sample correlation ID for testing."""
    return uuid.uuid4()


@pytest.fixture
def sample_workflow_execution_entity(sample_correlation_id):
    """
    Create a sample ModelWorkflowExecution entity.

    Returns:
        ModelWorkflowExecution: Workflow execution entity
    """
    from datetime import UTC, datetime

    from omninode_bridge.infrastructure.entities.model_workflow_execution import (
        ModelWorkflowExecution,
    )

    return ModelWorkflowExecution(
        correlation_id=sample_correlation_id,
        workflow_type="metadata_stamping",
        current_state="PROCESSING",
        namespace="test_namespace",
        started_at=datetime.now(UTC),
        metadata={
            "test": True,
            "version": "1.0",
            "source": "pytest_fixture",
        },
    )


@pytest.fixture
def sample_bridge_state_entity():
    """
    Create a sample ModelBridgeState entity.

    Returns:
        ModelBridgeState: Bridge state entity
    """
    from datetime import UTC, datetime

    from omninode_bridge.infrastructure.entities.model_bridge_state import (
        ModelBridgeState,
    )

    return ModelBridgeState(
        bridge_id=uuid.uuid4(),
        namespace="test_namespace",
        total_workflows_processed=150,
        total_items_aggregated=750,
        aggregation_metadata={
            "file_type_distribution": {"jpeg": 500, "pdf": 250},
            "avg_file_size_bytes": 102400,
        },
        current_fsm_state="aggregating",
        last_aggregation_timestamp=datetime.now(UTC),
    )


@pytest.fixture
def sample_metadata_stamp_entity(sample_correlation_id):
    """
    Create a sample ModelMetadataStamp entity.

    Returns:
        ModelMetadataStamp: Metadata stamp entity
    """
    from omninode_bridge.infrastructure.entities.model_metadata_stamp import (
        ModelMetadataStamp,
    )

    return ModelMetadataStamp(
        workflow_id=sample_correlation_id,
        file_hash="abc123def456789abcdef0123456789abcdef0123456789abcdef01234567890",  # 64 chars
        stamp_data={
            "stamp_type": "inline",
            "stamp_position": "header",
            "file_size_bytes": 1024,
        },
        namespace="test_namespace",
    )


@pytest.fixture
def sample_fsm_transition_entity(sample_correlation_id):
    """
    Create a sample ModelFSMTransition entity.

    Returns:
        ModelFSMTransition: FSM transition entity
    """
    from datetime import UTC, datetime

    from omninode_bridge.infrastructure.entities.model_fsm_transition import (
        ModelFSMTransition,
    )

    return ModelFSMTransition(
        entity_id=sample_correlation_id,
        entity_type="workflow",
        from_state="PENDING",
        to_state="PROCESSING",
        transition_event="start_processing",
        transition_data={
            "triggered_by": "pytest_fixture",
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


@pytest.fixture
def sample_node_heartbeat_entity():
    """
    Create a sample ModelNodeHeartbeat entity.

    Returns:
        ModelNodeHeartbeat: Node heartbeat entity
    """
    from omninode_bridge.infrastructure.entities.model_node_heartbeat import (
        ModelNodeHeartbeat,
    )

    return ModelNodeHeartbeat(
        node_id="database_adapter_node_01",
        node_type="effect",
        node_version="1.0.0",
        health_status="HEALTHY",
        metadata={
            "version": "1.0.0",
            "uptime_seconds": 3600,
            "memory_usage_mb": 256,
            "cpu_usage_percent": 15.5,
        },
    )


@pytest.fixture
def sample_workflow_step_entity(sample_correlation_id):
    """
    Create a sample ModelWorkflowStep entity.

    Returns:
        ModelWorkflowStep: Workflow step entity
    """
    from omninode_bridge.infrastructure.entities.model_workflow_step import (
        ModelWorkflowStep,
    )

    return ModelWorkflowStep(
        workflow_id=sample_correlation_id,
        step_name="generate_blake3_hash",
        step_order=1,
        status="COMPLETED",
        execution_time_ms=2,
        step_data={
            "file_hash": "abc123",
            "file_size_bytes": 1024,
            "performance_grade": "A",
        },
    )


@pytest.fixture
def mock_container():
    """
    Create a mock container with mock service dependencies.

    Returns:
        MockContainer: Container with mock services
    """
    from tests.test_database_adapter_node_manual import MockContainer

    return MockContainer()


# ============================================================================
# Test Configuration
# ============================================================================


# Mock isolation fixtures removed - using simplified test infrastructure
MOCK_ISOLATION_AVAILABLE = False


def pytest_configure(config):
    """Basic pytest configuration."""
    # Check if omnibase_core is available, if not use test stubs
    import importlib.util
    import sys
    from pathlib import Path

    if importlib.util.find_spec("omnibase_core") is not None:
        test_logger.info("Using installed omnibase_core package")

        # Patch omnibase_core import bug BEFORE any imports
        # omnibase_core tries to import protocol_workflow_reducer from wrong module
        try:
            from omnibase_core.protocols import workflow_orchestration

            # Inject protocol_workflow_reducer into omnibase_core.protocols.core
            if "omnibase_core.protocols.core" not in sys.modules:
                pass

            # Add the missing import to fix omnibase_core bug
            sys.modules["omnibase_core.protocols.core"].protocol_workflow_reducer = (
                workflow_orchestration.protocol_workflow_reducer
            )
            test_logger.info(
                "Applied omnibase_core import patch: protocol_workflow_reducer"
            )
        except Exception as e:
            test_logger.warning(f"Failed to apply omnibase_core import patch: {e}")
    else:
        # omnibase_core not installed, use test stubs
        test_logger.info("omnibase_core not installed, using test stubs")
        stub_path = Path(__file__).parent / "stubs"
        if str(stub_path) not in sys.path:
            sys.path.insert(0, str(stub_path))
            test_logger.info(f"Added test stubs to Python path: {stub_path}")

    # Add custom markers (integration and e2e markers are in pyproject.toml)
    config.addinivalue_line("markers", "fast: mark test as fast unit test")
    config.addinivalue_line(
        "markers", "performance_test: mark test for performance monitoring"
    )
    config.addinivalue_line(
        "markers", "pydantic_validation: tests for Pydantic type safety"
    )
