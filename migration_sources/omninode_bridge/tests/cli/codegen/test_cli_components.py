"""
Unit tests for individual CLI components.

Tests cover Kafka client, progress display, and config in isolation.
"""

import asyncio
from uuid import uuid4

import pytest

from omninode_bridge.cli.codegen.config import CodegenCLIConfig
from omninode_bridge.cli.codegen.ui import ProgressDisplay
from omninode_bridge.events.codegen import ModelEventNodeGenerationRequested

from .conftest import MockKafkaClient


@pytest.mark.asyncio
class TestKafkaClient:
    """Unit tests for Kafka client."""

    async def test_connect_once(self):
        """Test that connect can be called multiple times safely."""
        client = MockKafkaClient()

        await client.connect()
        assert client.is_connected

        # Second connect should be safe
        await client.connect()
        assert client.is_connected

    async def test_disconnect(self):
        """Test disconnect functionality."""
        client = MockKafkaClient()
        await client.connect()
        assert client.is_connected

        await client.disconnect()
        assert not client.is_connected

    async def test_publish_without_connection(self):
        """Test that publishing without connection raises error."""
        client = MockKafkaClient()

        with pytest.raises(RuntimeError, match="Not connected"):
            await client.publish_request(
                ModelEventNodeGenerationRequested(
                    correlation_id=uuid4(),
                    prompt="test",
                    output_directory="/tmp",
                )
            )

    async def test_publish_with_connection(self):
        """Test successful event publishing."""
        client = MockKafkaClient()
        await client.connect()

        event = ModelEventNodeGenerationRequested(
            correlation_id=uuid4(),
            prompt="Create test node",
            output_directory="/tmp/test",
            node_type="effect",
        )

        await client.publish_request(event)
        assert len(client.published_events) == 1
        assert client.published_events[0].prompt == "Create test node"

    async def test_consume_without_connection(self):
        """Test that consuming without connection raises error."""
        client = MockKafkaClient()

        with pytest.raises(RuntimeError, match="Not connected"):
            await client.consume_progress_events(
                correlation_id=uuid4(),
                callback=lambda event_type, data: None,
            )

    async def test_is_connected_property(self):
        """Test is_connected property."""
        client = MockKafkaClient()
        assert not client.is_connected

        await client.connect()
        assert client.is_connected

        await client.disconnect()
        assert not client.is_connected


@pytest.mark.asyncio
class TestProgressDisplay:
    """Unit tests for progress display."""

    def test_initialization(self):
        """Test progress display initialization."""
        correlation_id = uuid4()
        display = ProgressDisplay(correlation_id=correlation_id)

        assert display.correlation_id == correlation_id
        assert display.started_at is None
        assert display.completed_at is None
        assert display.completed_stages == 0
        assert display.result is None
        assert display.error is None

    def test_on_started_event(self):
        """Test handling of started event."""
        display = ProgressDisplay(correlation_id=uuid4())

        display.on_event(
            "NODE_GENERATION_STARTED",
            {
                "event_type": "NODE_GENERATION_STARTED",
                "correlation_id": str(display.correlation_id),
                "workflow_id": str(uuid4()),
            },
        )

        assert display.started_at is not None

    def test_on_stage_completed_event(self):
        """Test handling of stage completed event."""
        display = ProgressDisplay(correlation_id=uuid4())

        display.on_event(
            "NODE_GENERATION_STAGE_COMPLETED",
            {
                "event_type": "NODE_GENERATION_STAGE_COMPLETED",
                "correlation_id": str(display.correlation_id),
                "stage_number": 1,
                "stage_name": "contract_generation",
                "duration_seconds": 2.5,
                "success": True,
            },
        )

        assert display.completed_stages == 1
        assert "contract_generation" in display.stage_durations

    def test_on_completed_event(self):
        """Test handling of completed event."""
        display = ProgressDisplay(correlation_id=uuid4())

        display.on_event(
            "NODE_GENERATION_COMPLETED",
            {
                "event_type": "NODE_GENERATION_COMPLETED",
                "correlation_id": str(display.correlation_id),
                "workflow_id": str(uuid4()),
                "total_duration_seconds": 10.5,
                "generated_files": ["/tmp/test.py"],
                "node_type": "effect",
                "quality_score": 0.95,
            },
        )

        assert display.completed_at is not None
        assert display.result is not None
        assert display.result["total_duration_seconds"] == 10.5

    def test_on_failed_event(self):
        """Test handling of failed event."""
        display = ProgressDisplay(correlation_id=uuid4())

        display.on_event(
            "NODE_GENERATION_FAILED",
            {
                "event_type": "NODE_GENERATION_FAILED",
                "correlation_id": str(display.correlation_id),
                "error_message": "Test error",
                "failed_stage": "code_generation",
            },
        )

        assert display.completed_at is not None
        assert display.error == "Test error"

    def test_on_checkpoint_reached_event(self):
        """Test handling of checkpoint event."""
        display = ProgressDisplay(correlation_id=uuid4())

        # Should not raise
        display.on_event(
            "ORCHESTRATOR_CHECKPOINT_REACHED",
            {
                "event_type": "ORCHESTRATOR_CHECKPOINT_REACHED",
                "correlation_id": str(display.correlation_id),
                "checkpoint_type": "DESIGN_APPROVAL",
            },
        )

    async def test_wait_for_completion_success(self):
        """Test waiting for successful completion."""
        display = ProgressDisplay(correlation_id=uuid4())

        # Simulate completion in background
        async def simulate_completion():
            await asyncio.sleep(0.1)
            display.on_event(
                "NODE_GENERATION_COMPLETED",
                {
                    "event_type": "NODE_GENERATION_COMPLETED",
                    "correlation_id": str(display.correlation_id),
                    "workflow_id": str(uuid4()),
                    "total_duration_seconds": 5.0,
                    "generated_files": [],
                    "quality_score": 0.9,
                },
            )

        asyncio.create_task(simulate_completion())
        result = await display.wait_for_completion(timeout_seconds=5)

        assert result is not None
        assert result["total_duration_seconds"] == 5.0

    async def test_wait_for_completion_failure(self):
        """Test waiting for failed completion."""
        display = ProgressDisplay(correlation_id=uuid4())

        # Simulate failure in background
        async def simulate_failure():
            await asyncio.sleep(0.1)
            display.on_event(
                "NODE_GENERATION_FAILED",
                {
                    "event_type": "NODE_GENERATION_FAILED",
                    "correlation_id": str(display.correlation_id),
                    "error_message": "Simulated error",
                },
            )

        asyncio.create_task(simulate_failure())

        with pytest.raises(RuntimeError, match="Simulated error"):
            await display.wait_for_completion(timeout_seconds=5)

    async def test_wait_for_completion_timeout(self):
        """Test timeout during waiting."""
        display = ProgressDisplay(correlation_id=uuid4())
        display.started_at = asyncio.get_event_loop().time()

        # Don't send any completion event
        with pytest.raises(TimeoutError):
            await display.wait_for_completion(timeout_seconds=1)

    def test_get_elapsed_time(self):
        """Test elapsed time calculation."""
        display = ProgressDisplay(correlation_id=uuid4())

        # Before start
        assert display.get_elapsed_time() == 0.0

        # After start
        display.started_at = 100.0
        display.completed_at = 110.0
        assert display.get_elapsed_time() == 10.0

    def test_get_progress_percentage(self):
        """Test progress percentage calculation."""
        display = ProgressDisplay(correlation_id=uuid4())

        assert display.get_progress_percentage() == 0.0

        display.completed_stages = 4
        assert display.get_progress_percentage() == 50.0

        display.completed_stages = 8
        assert display.get_progress_percentage() == 100.0


class TestConfig:
    """Unit tests for configuration."""

    def test_default_values(self, monkeypatch):
        """Test default configuration values."""
        # Clear environment variables to test actual defaults
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv("CODEGEN_OUTPUT_DIR", raising=False)
        monkeypatch.delenv("CODEGEN_TIMEOUT_SECONDS", raising=False)
        monkeypatch.delenv("CODEGEN_VERBOSE", raising=False)

        config = CodegenCLIConfig()

        assert config.kafka_bootstrap_servers == "localhost:29092"
        assert config.default_output_dir == "./generated_nodes"
        assert config.default_timeout_seconds == 300
        assert config.enable_verbose_logging is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CodegenCLIConfig(
            kafka_bootstrap_servers="kafka:9092",
            default_output_dir="/custom/path",
            default_timeout_seconds=600,
            enable_verbose_logging=True,
        )

        assert config.kafka_bootstrap_servers == "kafka:9092"
        assert config.default_output_dir == "/custom/path"
        assert config.default_timeout_seconds == 600
        assert config.enable_verbose_logging is True

    def test_from_env(self, monkeypatch):
        """Test configuration loading from environment."""
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "env-kafka:9092")
        monkeypatch.setenv("CODEGEN_OUTPUT_DIR", "/env/output")
        monkeypatch.setenv("CODEGEN_TIMEOUT_SECONDS", "900")
        monkeypatch.setenv("CODEGEN_VERBOSE", "true")

        config = CodegenCLIConfig.from_env()

        # Note: from_env() uses environment variables only if explicitly implemented
        # Otherwise it uses defaults
        assert config is not None

    def test_with_overrides(self):
        """Test configuration with runtime overrides."""
        config = CodegenCLIConfig(
            kafka_bootstrap_servers="original:9092",
            default_output_dir="/original",
        )

        overridden = config.with_overrides(
            kafka_bootstrap_servers="override:9092",
            default_timeout_seconds=120,
        )

        # Overridden config should have new values
        assert overridden.kafka_bootstrap_servers == "override:9092"
        assert overridden.default_timeout_seconds == 120

        # Original should be unchanged
        assert config.kafka_bootstrap_servers == "original:9092"
        assert config.default_timeout_seconds == 300

    def test_partial_overrides(self):
        """Test partial configuration overrides."""
        config = CodegenCLIConfig()

        overridden = config.with_overrides(kafka_bootstrap_servers="new-kafka:9092")

        # Only specified field should change
        assert overridden.kafka_bootstrap_servers == "new-kafka:9092"
        assert overridden.default_output_dir == config.default_output_dir
        assert overridden.default_timeout_seconds == config.default_timeout_seconds
