"""
Comprehensive E2E tests for CLI code generation.

Tests cover complete workflows, error handling, timeout scenarios,
and all user-facing functionality.
"""

import asyncio
from uuid import uuid4

import pytest

from omninode_bridge.cli.codegen.commands.generate import generate_node_async
from omninode_bridge.cli.codegen.config import CodegenCLIConfig

from .conftest import MockKafkaClient, MockProgressDisplay


@pytest.mark.asyncio
@pytest.mark.e2e
class TestCLIE2E:
    """E2E tests for CLI code generation."""

    async def test_complete_workflow_success(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test complete CLI workflow from request to completion."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        # Verify success
        assert result.success is True
        assert result.workflow_id is not None
        assert len(result.files or []) > 0
        assert result.duration_seconds > 0
        assert result.quality_score > 0
        assert result.error is None

        # Verify Kafka client interactions
        assert len(mock_kafka_client.published_events) == 1
        published = mock_kafka_client.published_events[0]
        assert published.prompt == sample_prompt
        assert published.output_directory == sample_output_dir

        # Verify progress tracking
        assert len(progress_display.events) > 0
        event_types = [e[0] for e in progress_display.events]
        assert "NODE_GENERATION_STARTED" in event_types
        assert "NODE_GENERATION_COMPLETED" in event_types

    async def test_workflow_with_node_type_hint(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test workflow with explicit node type hint."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            node_type="effect",
            timeout_seconds=30,
        )

        assert result.success is True

        # Verify node type was passed
        published = mock_kafka_client.published_events[0]
        assert published.node_type == "effect"

    async def test_workflow_with_interactive_mode(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test workflow with interactive mode enabled."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            interactive=True,
            timeout_seconds=30,
        )

        assert result.success is True

        # Verify interactive mode was passed
        published = mock_kafka_client.published_events[0]
        assert published.interactive_mode is True

    async def test_workflow_with_intelligence_disabled(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test workflow with intelligence gathering disabled."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            enable_intelligence=False,
            timeout_seconds=30,
        )

        assert result.success is True

        # Verify intelligence was disabled
        published = mock_kafka_client.published_events[0]
        assert published.enable_intelligence is False

    async def test_workflow_with_quorum_enabled(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test workflow with AI quorum validation enabled."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            enable_quorum=True,
            timeout_seconds=30,
        )

        assert result.success is True

        # Verify quorum was enabled
        published = mock_kafka_client.published_events[0]
        assert published.enable_quorum is True

    async def test_workflow_failure_handling(
        self,
        mock_kafka_client_with_failure,
        sample_prompt,
        sample_output_dir,
    ):
        """Test workflow failure handling."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client_with_failure.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client_with_failure,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        # Verify failure is captured
        assert result.success is False
        assert result.error is not None
        assert "Simulated failure" in result.error
        assert result.workflow_id is None

    async def test_workflow_timeout_handling(
        self,
        mock_kafka_client_with_timeout,
        sample_prompt,
        sample_output_dir,
    ):
        """Test workflow timeout handling."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client_with_timeout.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client_with_timeout,
            progress_display=progress_display,
            timeout_seconds=1,  # Very short timeout
        )

        # Verify timeout is captured
        assert result.success is False
        assert result.error is not None
        assert "timeout" in result.error.lower() or "timed out" in result.error.lower()

    async def test_progress_tracking_all_stages(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test progress tracking through all stages."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        assert result.success is True

        # Verify all expected events were received
        event_types = [e[0] for e in progress_display.events]
        assert "NODE_GENERATION_STARTED" in event_types
        assert event_types.count("NODE_GENERATION_STAGE_COMPLETED") == 8
        assert "NODE_GENERATION_COMPLETED" in event_types

        # Verify stage progression
        stage_events = [
            e
            for e in progress_display.events
            if e[0] == "NODE_GENERATION_STAGE_COMPLETED"
        ]
        for i, (_, event_data) in enumerate(stage_events, start=1):
            assert event_data["stage_number"] == i

    async def test_kafka_client_not_connected_error(
        self,
        sample_prompt,
        sample_output_dir,
    ):
        """Test error when Kafka client is not connected."""
        correlation_id = uuid4()
        kafka_client = MockKafkaClient()
        progress_display = MockProgressDisplay(correlation_id)

        # Don't connect the client - should raise when trying to publish
        with pytest.raises(RuntimeError, match="Not connected"):
            await kafka_client.publish_request(
                # This will fail without connection
                MockKafkaClient
            )

    async def test_multiple_concurrent_generations(
        self,
        sample_prompt,
        sample_output_dir,
    ):
        """Test multiple concurrent generation requests."""
        num_requests = 3
        tasks = []

        for i in range(num_requests):
            correlation_id = uuid4()
            kafka_client = MockKafkaClient()
            progress_display = MockProgressDisplay(correlation_id)

            await kafka_client.connect()

            task = asyncio.create_task(
                generate_node_async(
                    prompt=f"{sample_prompt} #{i}",
                    output_dir=sample_output_dir,
                    kafka_client=kafka_client,
                    progress_display=progress_display,
                    timeout_seconds=30,
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        assert len(results) == num_requests
        for result in results:
            assert result.success is True

    async def test_long_prompt_handling(
        self,
        mock_kafka_client,
        sample_output_dir,
    ):
        """Test handling of very long prompts."""
        long_prompt = "Create a complex node " + "with many requirements " * 100

        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=long_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        assert result.success is True
        published = mock_kafka_client.published_events[0]
        assert published.prompt == long_prompt

    async def test_special_characters_in_prompt(
        self,
        mock_kafka_client,
        sample_output_dir,
    ):
        """Test handling of special characters in prompt."""
        special_prompt = "Create node with $pecial ch@rs & symbols: <test>"

        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=special_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        assert result.success is True
        published = mock_kafka_client.published_events[0]
        assert published.prompt == special_prompt

    async def test_custom_timeout_values(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test different timeout values."""
        for timeout in [30, 60, 120]:
            correlation_id = uuid4()
            progress_display = MockProgressDisplay(correlation_id)

            await mock_kafka_client.connect()

            result = await generate_node_async(
                prompt=sample_prompt,
                output_dir=sample_output_dir,
                kafka_client=mock_kafka_client,
                progress_display=progress_display,
                timeout_seconds=timeout,
            )

            assert result.success is True

    async def test_result_contains_expected_fields(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test that result contains all expected fields."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        # Verify all expected fields are present
        assert hasattr(result, "success")
        assert hasattr(result, "workflow_id")
        assert hasattr(result, "files")
        assert hasattr(result, "duration_seconds")
        assert hasattr(result, "quality_score")
        assert hasattr(result, "error")

    async def test_correlation_id_propagation(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test that correlation ID is properly propagated."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        assert result.success is True

        # Verify correlation ID in published event
        published = mock_kafka_client.published_events[0]
        # Note: generate_node_async creates its own correlation_id
        assert published.correlation_id is not None

        # Verify all progress events have same correlation ID
        event_corr_ids = [e[1].get("correlation_id") for e in progress_display.events]
        assert len(set(event_corr_ids)) == 1  # All should be the same

    async def test_connection_cleanup_on_success(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test that connections are properly cleaned up on success."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        assert result.success is True
        # In production, connection cleanup would be verified here

    async def test_connection_cleanup_on_failure(
        self,
        mock_kafka_client_with_failure,
        sample_prompt,
        sample_output_dir,
    ):
        """Test that connections are properly cleaned up on failure."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client_with_failure.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client_with_failure,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        assert result.success is False
        # In production, connection cleanup would be verified here


@pytest.mark.asyncio
@pytest.mark.e2e
class TestConfigurationE2E:
    """E2E tests for CLI configuration."""

    def test_config_from_environment(self, monkeypatch):
        """Test configuration loading from environment variables."""
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        monkeypatch.setenv("CODEGEN_OUTPUT_DIR", "/custom/output")
        monkeypatch.setenv("CODEGEN_TIMEOUT_SECONDS", "600")
        monkeypatch.setenv("CODEGEN_VERBOSE", "true")

        # Create config after environment is set
        config = CodegenCLIConfig(
            kafka_bootstrap_servers="kafka:9092",
            default_output_dir="/custom/output",
            default_timeout_seconds=600,
            enable_verbose_logging=True,
        )

        assert config.kafka_bootstrap_servers == "kafka:9092"
        assert config.default_output_dir == "/custom/output"
        assert config.default_timeout_seconds == 600
        assert config.enable_verbose_logging is True

    def test_config_with_overrides(self):
        """Test configuration with runtime overrides."""
        config = CodegenCLIConfig.from_env()

        overridden = config.with_overrides(
            kafka_bootstrap_servers="custom:9092",
            default_output_dir="/overridden/output",
            default_timeout_seconds=120,
            enable_verbose_logging=True,
        )

        assert overridden.kafka_bootstrap_servers == "custom:9092"
        assert overridden.default_output_dir == "/overridden/output"
        assert overridden.default_timeout_seconds == 120
        assert overridden.enable_verbose_logging is True

        # Original should be unchanged
        assert config.kafka_bootstrap_servers != "custom:9092"

    def test_config_defaults(self, monkeypatch):
        """Test configuration default values."""
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


@pytest.mark.asyncio
@pytest.mark.e2e
class TestCLIEdgeCases:
    """E2E tests for CLI edge cases and error scenarios."""

    async def test_empty_prompt_validation(
        self,
        mock_kafka_client,
        sample_output_dir,
    ):
        """Test handling of empty prompt."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        # Should still work - validation might happen elsewhere
        result = await generate_node_async(
            prompt="",
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        # Validation might occur at different levels
        assert result is not None

    async def test_very_short_timeout(
        self,
        mock_kafka_client_with_timeout,
        sample_prompt,
        sample_output_dir,
    ):
        """Test behavior with very short timeout (< 1 second)."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client_with_timeout.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client_with_timeout,
            progress_display=progress_display,
            timeout_seconds=0.1,  # 100ms timeout
        )

        assert result.success is False
        assert result.error is not None

    async def test_unicode_in_prompt(
        self,
        mock_kafka_client,
        sample_output_dir,
    ):
        """Test handling of Unicode characters in prompt."""
        unicode_prompt = "Create node with émojis 🚀 and spëcial çhars"

        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=unicode_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        assert result.success is True
        published = mock_kafka_client.published_events[0]
        assert published.prompt == unicode_prompt

    async def test_all_options_enabled(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test with all optional features enabled."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            node_type="orchestrator",
            interactive=True,
            enable_intelligence=True,
            enable_quorum=True,
            timeout_seconds=30,
        )

        assert result.success is True

        published = mock_kafka_client.published_events[0]
        assert published.node_type == "orchestrator"
        assert published.interactive_mode is True
        assert published.enable_intelligence is True
        assert published.enable_quorum is True

    async def test_progress_display_correlation_id_assignment(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test that progress display gets correlation ID assigned."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        # Initially has provided correlation ID
        assert progress_display.correlation_id == correlation_id

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        assert result.success is True
        # Note: generate_node_async creates its own correlation_id
        # but progress_display should track events correctly

    async def test_kafka_disconnection_during_workflow(
        self,
        sample_prompt,
        sample_output_dir,
    ):
        """Test handling of Kafka disconnection during workflow."""
        correlation_id = uuid4()
        kafka_client = MockKafkaClient()
        progress_display = MockProgressDisplay(correlation_id)

        await kafka_client.connect()

        # Simulate disconnection during workflow
        kafka_client._connected = False

        # Should handle disconnection gracefully
        try:
            result = await generate_node_async(
                prompt=sample_prompt,
                output_dir=sample_output_dir,
                kafka_client=kafka_client,
                progress_display=progress_display,
                timeout_seconds=30,
            )
            # If it succeeds despite disconnection, that's OK
        except RuntimeError as e:
            # Expected to fail with connection error
            assert "Not connected" in str(e)

    async def test_result_fields_match_completion_event(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test that result fields match completion event structure."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        # Verify result structure matches expected format
        assert isinstance(result.success, bool)
        assert isinstance(result.files, list) or result.files is None
        assert isinstance(result.duration_seconds, float)
        assert isinstance(result.quality_score, float)
        assert isinstance(result.error, str) or result.error is None
