"""
Test fixtures for CLI E2E tests.

Provides mock implementations and fixtures for comprehensive testing.
"""

import asyncio
from typing import Any
from uuid import UUID, uuid4

import pytest

from omninode_bridge.events.codegen import ModelEventNodeGenerationRequested


class MockKafkaClient:
    """
    Mock Kafka client for testing.

    Simulates Kafka operations without requiring actual Kafka infrastructure.
    """

    def __init__(self, bootstrap_servers: str = "localhost:29092"):
        self.bootstrap_servers = bootstrap_servers
        self._connected = False
        self.published_events: list[ModelEventNodeGenerationRequested] = []
        self.consumed_events: list[dict[str, Any]] = []
        self.consume_callback = None
        self.simulate_failure = False
        self.simulate_timeout = False

    async def connect(self) -> None:
        """Simulate connection."""
        self._connected = True

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False

    async def publish_request(self, event: ModelEventNodeGenerationRequested) -> None:
        """Simulate request publishing."""
        if not self._connected:
            raise RuntimeError("Not connected")
        self.published_events.append(event)

    async def consume_progress_events(
        self,
        correlation_id: UUID,
        callback,
    ) -> None:
        """Simulate progress event consumption."""
        if not self._connected:
            raise RuntimeError("Not connected")

        self.consume_callback = callback

        # Simulate event sequence
        if self.simulate_timeout:
            # Never send completion event
            await asyncio.sleep(10)
            return

        # Simulate started event
        workflow_id = uuid4()
        callback(
            "NODE_GENERATION_STARTED",
            {
                "event_type": "NODE_GENERATION_STARTED",
                "correlation_id": str(correlation_id),
                "workflow_id": str(workflow_id),
                "orchestrator_node_id": str(uuid4()),
                "prompt": "Test node generation",
                "output_directory": "/tmp/test",
                "node_type_hint": None,
            },
        )

        # Simulate stage completion events
        for stage_num in range(1, 9):
            await asyncio.sleep(0.01)  # Small delay for realism
            callback(
                "NODE_GENERATION_STAGE_COMPLETED",
                {
                    "event_type": "NODE_GENERATION_STAGE_COMPLETED",
                    "correlation_id": str(correlation_id),
                    "workflow_id": str(workflow_id),
                    "stage_number": stage_num,
                    "stage_name": f"stage_{stage_num}",
                    "duration_seconds": 0.1,
                    "success": True,
                },
            )

        # Simulate completion or failure
        if self.simulate_failure:
            callback(
                "NODE_GENERATION_FAILED",
                {
                    "event_type": "NODE_GENERATION_FAILED",
                    "correlation_id": str(correlation_id),
                    "workflow_id": str(workflow_id),
                    "error_message": "Simulated failure",
                    "error_code": "TEST_ERROR",
                    "failed_stage": "stage_5",
                    "partial_duration_seconds": 5.0,
                },
            )
        else:
            callback(
                "NODE_GENERATION_COMPLETED",
                {
                    "event_type": "NODE_GENERATION_COMPLETED",
                    "correlation_id": str(correlation_id),
                    "workflow_id": str(workflow_id),
                    "total_duration_seconds": 10.5,
                    "generated_files": [
                        "/tmp/test/node_test_effect.py",
                        "/tmp/test/test_node_test_effect.py",
                    ],
                    "node_type": "effect",
                    "service_name": "test_service",
                    "quality_score": 0.92,
                    "test_coverage": 95.0,
                    "primary_model": "gpt-4",
                    "total_tokens": 5000,
                    "total_cost_usd": 0.15,
                    "contract_yaml": "contract_test_effect.yaml",
                    "node_module": "node_test_effect.py",
                    "models": ["model_test.py"],
                    "enums": ["enum_test.py"],
                    "tests": ["test_node_test_effect.py"],
                },
            )

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected


class MockProgressDisplay:
    """
    Mock progress display for testing.

    Tracks all events and provides completion simulation.
    """

    def __init__(self, correlation_id: UUID):
        self.correlation_id = correlation_id
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self._completion_event = asyncio.Event()

    def on_event(self, event_type: str, event_data: dict[str, Any]) -> None:
        """Record event."""
        self.events.append((event_type, event_data))

        if event_type == "NODE_GENERATION_COMPLETED":
            self.result = event_data
            self._completion_event.set()
        elif event_type == "NODE_GENERATION_FAILED":
            self.error = event_data.get("error_message")
            self._completion_event.set()

    async def wait_for_completion(self, timeout_seconds: int = 300) -> dict[str, Any]:
        """Wait for completion."""
        try:
            await asyncio.wait_for(
                self._completion_event.wait(), timeout=timeout_seconds
            )
        except TimeoutError:
            raise TimeoutError(f"Timed out after {timeout_seconds}s")

        if self.error:
            raise RuntimeError(self.error)

        if not self.result:
            raise RuntimeError("No result")

        return self.result

    def get_elapsed_time(self) -> float:
        """Get elapsed time."""
        return 10.5

    def get_progress_percentage(self) -> float:
        """Get progress percentage."""
        return 100.0


@pytest.fixture
def mock_kafka_client():
    """Provide mock Kafka client."""
    return MockKafkaClient()


@pytest.fixture
def mock_kafka_client_with_failure():
    """Provide mock Kafka client that simulates failure."""
    client = MockKafkaClient()
    client.simulate_failure = True
    return client


@pytest.fixture
def mock_kafka_client_with_timeout():
    """Provide mock Kafka client that simulates timeout."""
    client = MockKafkaClient()
    client.simulate_timeout = True
    return client


@pytest.fixture
def mock_progress_display():
    """Provide mock progress display."""
    return MockProgressDisplay(correlation_id=uuid4())


@pytest.fixture
def sample_correlation_id():
    """Provide sample correlation ID."""
    return uuid4()


@pytest.fixture
def sample_prompt():
    """Provide sample prompt."""
    return "Create a PostgreSQL CRUD Effect node"


@pytest.fixture
def sample_output_dir(tmp_path):
    """Provide sample output directory."""
    output_dir = tmp_path / "generated_nodes"
    output_dir.mkdir()
    return str(output_dir)
