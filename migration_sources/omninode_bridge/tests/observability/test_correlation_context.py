"""Tests for correlation context management."""

from uuid import uuid4

import pytest

from omninode_bridge.observability.logging_config import (
    correlation_context,
    correlation_context_sync,
    get_correlation_context,
)


@pytest.mark.asyncio
class TestCorrelationContextAsync:
    """Tests for async correlation context."""

    async def test_nested_correlation_contexts(self):
        """Test nested correlation contexts."""
        outer_id = uuid4()
        inner_id = uuid4()

        async with correlation_context(correlation_id=outer_id):
            context = get_correlation_context()
            assert context["correlation_id"] == outer_id

            async with correlation_context(correlation_id=inner_id):
                context = get_correlation_context()
                assert context["correlation_id"] == inner_id

            # Should restore outer context
            context = get_correlation_context()
            assert context["correlation_id"] == outer_id

    async def test_partial_context_updates(self):
        """Test updating only some context fields."""
        correlation_id = uuid4()
        workflow_id = uuid4()

        async with correlation_context(correlation_id=correlation_id):
            # Update only workflow_id
            async with correlation_context(workflow_id=workflow_id):
                context = get_correlation_context()
                assert context["correlation_id"] == correlation_id
                assert context["workflow_id"] == workflow_id

    async def test_context_isolation_between_coroutines(self):
        """Test that context is isolated between coroutines."""
        correlation_id_1 = uuid4()
        correlation_id_2 = uuid4()

        async def task1():
            async with correlation_context(correlation_id=correlation_id_1):
                context = get_correlation_context()
                assert context["correlation_id"] == correlation_id_1

        async def task2():
            async with correlation_context(correlation_id=correlation_id_2):
                context = get_correlation_context()
                assert context["correlation_id"] == correlation_id_2

        await task1()
        await task2()

    async def test_exception_handling_in_context(self):
        """Test that context is reset even when exception occurs."""
        correlation_id = uuid4()

        with pytest.raises(ValueError):
            async with correlation_context(correlation_id=correlation_id):
                raise ValueError("Test error")

        # Context should be reset
        context = get_correlation_context()
        assert context["correlation_id"] is None

    async def test_multiple_field_updates(self):
        """Test updating multiple fields at once."""
        correlation_id = uuid4()
        workflow_id = uuid4()
        request_id = uuid4()

        async with correlation_context(
            correlation_id=correlation_id,
            workflow_id=workflow_id,
            request_id=request_id,
            session_id="session123",
            user_id="user456",
            stage_name="processing",
        ):
            context = get_correlation_context()
            assert context["correlation_id"] == correlation_id
            assert context["workflow_id"] == workflow_id
            assert context["request_id"] == request_id
            assert context["session_id"] == "session123"
            assert context["user_id"] == "user456"
            assert context["stage_name"] == "processing"

    async def test_context_reset_to_none(self):
        """Test that all context values reset to None."""
        correlation_id = uuid4()
        workflow_id = uuid4()

        async with correlation_context(
            correlation_id=correlation_id, workflow_id=workflow_id
        ):
            pass

        context = get_correlation_context()
        assert context["correlation_id"] is None
        assert context["workflow_id"] is None
        assert context["request_id"] is None


class TestCorrelationContextSync:
    """Tests for sync correlation context."""

    def test_sync_context_basic(self):
        """Test basic sync context functionality."""
        correlation_id = uuid4()

        with correlation_context_sync(correlation_id=correlation_id):
            context = get_correlation_context()
            assert context["correlation_id"] == correlation_id

    def test_sync_context_nested(self):
        """Test nested sync contexts."""
        outer_id = uuid4()
        inner_id = uuid4()

        with correlation_context_sync(correlation_id=outer_id):
            with correlation_context_sync(correlation_id=inner_id):
                context = get_correlation_context()
                assert context["correlation_id"] == inner_id

            context = get_correlation_context()
            assert context["correlation_id"] == outer_id
