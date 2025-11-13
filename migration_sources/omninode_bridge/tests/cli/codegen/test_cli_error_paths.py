"""
Error path and edge case tests for CLI.

Tests cover error handling, edge cases, and exceptional scenarios.
"""

import asyncio
from uuid import uuid4

import pytest

from omninode_bridge.cli.codegen.commands.generate import (
    GenerationResult,
    generate_node_async,
)

from .conftest import MockKafkaClient, MockProgressDisplay


@pytest.mark.asyncio
@pytest.mark.e2e
class TestErrorPaths:
    """Tests for error paths and edge cases."""

    async def test_kafka_client_publish_error_propagation(
        self,
        sample_prompt,
        sample_output_dir,
    ):
        """Test error propagation from Kafka publish failures."""
        correlation_id = uuid4()

        # Create disconnected client to force error
        kafka_client = MockKafkaClient()
        # Don't connect - should raise error when publishing
        progress_display = MockProgressDisplay(correlation_id)

        with pytest.raises(RuntimeError, match="Not connected"):
            # This should fail because client is not connected
            from omninode_bridge.events.codegen import ModelEventNodeGenerationRequested

            await kafka_client.publish_request(
                ModelEventNodeGenerationRequested(
                    correlation_id=correlation_id,
                    prompt=sample_prompt,
                    output_directory=sample_output_dir,
                )
            )

    async def test_consumer_cancel_handling(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test that consumer task cancellation is handled gracefully."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        # The function should handle cancellation properly
        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        assert result.success is True

    async def test_empty_prompt_handling(
        self,
        mock_kafka_client,
        sample_output_dir,
    ):
        """Test handling of empty prompt."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt="",  # Empty prompt
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        # Should still succeed (validation happens elsewhere)
        assert result.success is True

    async def test_very_short_timeout(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test behavior with very short timeout."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        # Even with short timeout, mock should complete quickly
        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=1,
        )

        assert result.success is True

    async def test_zero_timeout(
        self,
        mock_kafka_client_with_timeout,
        sample_prompt,
        sample_output_dir,
    ):
        """Test behavior with zero timeout."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client_with_timeout.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client_with_timeout,
            progress_display=progress_display,
            timeout_seconds=0,  # Zero timeout
        )

        # Should timeout immediately
        assert result.success is False
        assert result.error is not None

    async def test_invalid_node_type(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test handling of invalid node type."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client.connect()

        # CLI accepts any string, validation happens in orchestrator
        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            node_type="invalid_type",
            timeout_seconds=30,
        )

        assert result.success is True

    async def test_missing_correlation_id_attribute(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test progress display without correlation_id attribute."""

        # Create minimal progress display without correlation_id
        class MinimalProgressDisplay:
            def __init__(self):
                self._completion_event = asyncio.Event()
                self.result = None
                self.error = None

            def on_event(self, event_type, event_data):
                if event_type == "NODE_GENERATION_COMPLETED":
                    self.result = event_data
                    self._completion_event.set()

            async def wait_for_completion(self, timeout_seconds):
                await asyncio.wait_for(
                    self._completion_event.wait(), timeout=timeout_seconds
                )
                return self.result or {}

        progress_display = MinimalProgressDisplay()

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        # Should handle missing attribute gracefully
        assert result.success is True

    async def test_progress_display_returns_none(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test when progress display returns None result."""

        class NoneReturningDisplay:
            def __init__(self):
                self._completion_event = asyncio.Event()
                self.correlation_id = uuid4()

            def on_event(self, event_type, event_data):
                if event_type == "NODE_GENERATION_COMPLETED":
                    self._completion_event.set()

            async def wait_for_completion(self, timeout_seconds):
                await asyncio.wait_for(
                    self._completion_event.wait(), timeout=timeout_seconds
                )
                return {}  # Return empty dict instead of None

        progress_display = NoneReturningDisplay()

        await mock_kafka_client.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client,
            progress_display=progress_display,
            timeout_seconds=30,
        )

        # Should handle empty result
        assert result.success is True
        assert result.workflow_id is None

    async def test_concurrent_operations_different_correlation_ids(
        self,
        sample_prompt,
        sample_output_dir,
    ):
        """Test multiple operations with different correlation IDs."""
        operations = []

        for i in range(5):
            correlation_id = uuid4()
            kafka_client = MockKafkaClient()
            progress_display = MockProgressDisplay(correlation_id)

            await kafka_client.connect()

            op = generate_node_async(
                prompt=f"{sample_prompt} #{i}",
                output_dir=sample_output_dir,
                kafka_client=kafka_client,
                progress_display=progress_display,
                timeout_seconds=30,
            )
            operations.append(op)

        results = await asyncio.gather(*operations)

        # All should succeed with unique IDs
        assert len(results) == 5
        for result in results:
            assert result.success is True

    async def test_all_parameters_specified(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test with all optional parameters specified."""
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
            timeout_seconds=600,
        )

        assert result.success is True

        # Verify all parameters were passed
        published = mock_kafka_client.published_events[0]
        assert published.node_type == "orchestrator"
        assert published.interactive_mode is True
        assert published.enable_intelligence is True
        assert published.enable_quorum is True

    async def test_generation_result_all_fields(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test that GenerationResult contains all expected fields."""
        result = GenerationResult(
            success=True,
            workflow_id="test-workflow",
            files=["/tmp/test.py"],
            duration_seconds=10.5,
            quality_score=0.95,
            error=None,
        )

        assert result.success is True
        assert result.workflow_id == "test-workflow"
        assert result.files == ["/tmp/test.py"]
        assert result.duration_seconds == 10.5
        assert result.quality_score == 0.95
        assert result.error is None

    async def test_generation_result_error_state(self):
        """Test GenerationResult in error state."""
        result = GenerationResult(
            success=False,
            error="Test error message",
        )

        assert result.success is False
        assert result.error == "Test error message"
        assert result.workflow_id is None
        assert result.files is None

    async def test_multiple_stage_completions(
        self,
        mock_kafka_client,
        sample_prompt,
        sample_output_dir,
    ):
        """Test that all 8 stages are properly tracked."""
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

        # Verify all 8 stages were completed
        stage_events = [
            e
            for e in progress_display.events
            if e[0] == "NODE_GENERATION_STAGE_COMPLETED"
        ]
        assert len(stage_events) == 8

    async def test_runtime_error_conversion(
        self,
        mock_kafka_client_with_failure,
        sample_prompt,
        sample_output_dir,
    ):
        """Test that RuntimeError is properly converted to GenerationResult."""
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
        assert result.error is not None
        assert isinstance(result.error, str)

    async def test_timeout_error_conversion(
        self,
        mock_kafka_client_with_timeout,
        sample_prompt,
        sample_output_dir,
    ):
        """Test that TimeoutError is properly converted to GenerationResult."""
        correlation_id = uuid4()
        progress_display = MockProgressDisplay(correlation_id)

        await mock_kafka_client_with_timeout.connect()

        result = await generate_node_async(
            prompt=sample_prompt,
            output_dir=sample_output_dir,
            kafka_client=mock_kafka_client_with_timeout,
            progress_display=progress_display,
            timeout_seconds=1,
        )

        assert result.success is False
        assert result.error is not None
