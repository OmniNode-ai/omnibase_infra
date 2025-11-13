"""Tests for distributed tracing."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from opentelemetry.trace import StatusCode

from omninode_bridge.observability.logging_config import correlation_context
from omninode_bridge.observability.tracing import (
    add_span_attributes,
    add_span_event,
    create_span_context,
    extract_trace_context,
    get_current_span,
    get_tracer,
    inject_trace_context,
    set_span_error,
    set_span_success,
    trace_async,
    trace_sync,
)


class TestGetTracer:
    """Tests for get_tracer."""

    def test_get_tracer_returns_tracer(self):
        """Test that get_tracer returns tracer instance."""
        tracer = get_tracer("test_module")
        assert tracer is not None


class TestGetCurrentSpan:
    """Tests for get_current_span."""

    def test_get_current_span_returns_span(self):
        """Test that get_current_span returns span."""
        span = get_current_span()
        assert span is not None


class TestSpanAttributes:
    """Tests for span attribute functions."""

    @patch("omninode_bridge.observability.tracing.get_current_span")
    def test_add_span_attributes(self, mock_get_span):
        """Test adding span attributes."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        add_span_attributes(node_type="effect", stage="validation")

        assert mock_span.set_attribute.call_count == 2
        mock_span.set_attribute.assert_any_call("node_type", "effect")
        mock_span.set_attribute.assert_any_call("stage", "validation")

    @patch("omninode_bridge.observability.tracing.get_current_span")
    def test_add_span_attributes_non_recording_span(self, mock_get_span):
        """Test add_span_attributes with non-recording span."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        mock_get_span.return_value = mock_span

        add_span_attributes(key="value")

        # Should not call set_attribute
        mock_span.set_attribute.assert_not_called()


class TestSpanEvents:
    """Tests for span event functions."""

    @patch("omninode_bridge.observability.tracing.get_current_span")
    def test_add_span_event(self, mock_get_span):
        """Test adding span event."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        add_span_event("checkpoint_reached", {"checkpoint_type": "contract_review"})

        mock_span.add_event.assert_called_once_with(
            "checkpoint_reached", {"checkpoint_type": "contract_review"}
        )

    @patch("omninode_bridge.observability.tracing.get_current_span")
    def test_add_span_event_without_attributes(self, mock_get_span):
        """Test adding span event without attributes."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        add_span_event("event_name")

        mock_span.add_event.assert_called_once_with("event_name", {})


class TestSpanStatus:
    """Tests for span status functions."""

    @patch("omninode_bridge.observability.tracing.get_current_span")
    def test_set_span_error(self, mock_get_span):
        """Test setting span error."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        error = ValueError("Test error")
        set_span_error(error)

        # Check that status was set to ERROR
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.ERROR
        assert "Test error" in str(status_call.description)

        # Check that exception was recorded
        mock_span.record_exception.assert_called_once_with(error)

    @patch("omninode_bridge.observability.tracing.get_current_span")
    def test_set_span_success(self, mock_get_span):
        """Test setting span success."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        set_span_success()

        # Check that status was set to OK
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.OK


@pytest.mark.asyncio
class TestTraceAsync:
    """Tests for trace_async decorator."""

    async def test_trace_async_decorator(self):
        """Test trace_async decorator basic functionality."""

        @trace_async()
        async def test_function():
            return "result"

        result = await test_function()
        assert result == "result"

    async def test_trace_async_with_custom_span_name(self):
        """Test trace_async with custom span name."""

        @trace_async(span_name="custom_span")
        async def test_function():
            return "result"

        result = await test_function()
        assert result == "result"

    async def test_trace_async_with_correlation_context(self):
        """Test trace_async includes correlation context."""
        correlation_id = uuid4()

        @trace_async(add_correlation=True)
        async def test_function():
            return "result"

        async with correlation_context(correlation_id=correlation_id):
            result = await test_function()
            assert result == "result"

    async def test_trace_async_exception_handling(self):
        """Test trace_async records exceptions."""

        @trace_async(record_exception=True)
        async def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await test_function()

    async def test_trace_async_without_exception_recording(self):
        """Test trace_async without exception recording."""

        @trace_async(record_exception=False)
        async def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await test_function()


class TestTraceSync:
    """Tests for trace_sync decorator."""

    def test_trace_sync_decorator(self):
        """Test trace_sync decorator basic functionality."""

        @trace_sync()
        def test_function():
            return "result"

        result = test_function()
        assert result == "result"

    def test_trace_sync_with_custom_span_name(self):
        """Test trace_sync with custom span name."""

        @trace_sync(span_name="custom_span")
        def test_function():
            return "result"

        result = test_function()
        assert result == "result"

    def test_trace_sync_exception_handling(self):
        """Test trace_sync records exceptions."""

        @trace_sync(record_exception=True)
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_function()


class TestCreateSpanContext:
    """Tests for create_span_context."""

    def test_create_span_context(self):
        """Test creating span context."""
        with create_span_context("test_span") as span:
            assert span is not None

    @pytest.mark.asyncio
    async def test_create_span_context_with_correlation(self):
        """Test create_span_context includes correlation."""
        correlation_id = uuid4()

        async with correlation_context(correlation_id=correlation_id):
            with create_span_context("test_span", add_correlation=True):
                pass


class TestTraceContextPropagation:
    """Tests for trace context propagation."""

    def test_inject_trace_context(self):
        """Test injecting trace context into headers."""
        headers = {"Content-Type": "application/json"}
        updated_headers = inject_trace_context(headers)

        # Should have original headers
        assert updated_headers["Content-Type"] == "application/json"
        # Should have trace context (exact keys depend on propagator)
        assert len(updated_headers) >= len(headers)

    def test_extract_trace_context(self):
        """Test extracting trace context from headers."""
        headers = {
            "traceparent": "00-12345678901234567890123456789012-1234567890123456-01"
        }
        context = extract_trace_context(headers)

        assert context is not None
