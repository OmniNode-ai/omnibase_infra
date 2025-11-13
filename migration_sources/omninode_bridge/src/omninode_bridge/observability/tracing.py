"""
Distributed tracing wrapper for omninode_bridge.

Provides simplified API around OpenTelemetry for distributed tracing,
span management, and trace context propagation.
"""

import functools
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from omninode_bridge.observability.logging_config import get_correlation_context

T = TypeVar("T")


def get_tracer(name: str = "omninode_bridge") -> Tracer:
    """Get OpenTelemetry tracer instance.

    Args:
        name: Tracer name (typically module name)

    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


def get_current_span() -> Span:
    """Get the current active span.

    Returns:
        Current span instance
    """
    return trace.get_current_span()


def add_span_attributes(**attributes) -> None:
    """Add attributes to the current span.

    Args:
        **attributes: Key-value pairs to add as span attributes

    Example:
        add_span_attributes(
            node_type="effect",
            stage_name="code_generation",
            quality_score=0.95
        )
    """
    current_span = get_current_span()
    if current_span and current_span.is_recording():
        for key, value in attributes.items():
            current_span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[dict[str, Any]] = None) -> None:
    """Add an event to the current span.

    Args:
        name: Event name
        attributes: Optional event attributes

    Example:
        add_span_event(
            "checkpoint_reached",
            {"checkpoint_type": "contract_review", "action": "approved"}
        )
    """
    current_span = get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(name, attributes or {})


def set_span_error(error: Exception) -> None:
    """Mark the current span as having an error.

    Args:
        error: The exception that occurred

    Example:
        try:
            process_data()
        except Exception as e:
            set_span_error(e)
            raise
    """
    current_span = get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_status(Status(StatusCode.ERROR, str(error)))
        current_span.record_exception(error)


def set_span_success() -> None:
    """Mark the current span as successful."""
    current_span = get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_status(Status(StatusCode.OK))


def trace_async(
    span_name: Optional[str] = None,
    add_correlation: bool = True,
    record_exception: bool = True,
):
    """Decorator to automatically trace async functions.

    Args:
        span_name: Custom span name (default: function name)
        add_correlation: Add correlation context to span
        record_exception: Record exceptions in span

    Example:
        @trace_async(span_name="generate_node_code")
        async def generate_code(node_type: str):
            return await process_generation(node_type)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)
            name = span_name or func.__name__

            with tracer.start_as_current_span(name) as span:
                # Add correlation context
                if add_correlation:
                    correlation_ctx = get_correlation_context()
                    for key, value in correlation_ctx.items():
                        if value is not None:
                            span.set_attribute(f"omninode.{key}", str(value))

                # Add function metadata
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                try:
                    result = await func(*args, **kwargs)
                    set_span_success()
                    return result
                except Exception as e:
                    if record_exception:
                        set_span_error(e)
                    raise

        return wrapper

    return decorator


def trace_sync(
    span_name: Optional[str] = None,
    add_correlation: bool = True,
    record_exception: bool = True,
):
    """Decorator to automatically trace sync functions.

    Args:
        span_name: Custom span name (default: function name)
        add_correlation: Add correlation context to span
        record_exception: Record exceptions in span

    Example:
        @trace_sync(span_name="validate_contract")
        def validate(contract: dict):
            return perform_validation(contract)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)
            name = span_name or func.__name__

            with tracer.start_as_current_span(name) as span:
                # Add correlation context
                if add_correlation:
                    correlation_ctx = get_correlation_context()
                    for key, value in correlation_ctx.items():
                        if value is not None:
                            span.set_attribute(f"omninode.{key}", str(value))

                # Add function metadata
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    set_span_success()
                    return result
                except Exception as e:
                    if record_exception:
                        set_span_error(e)
                    raise

        return wrapper

    return decorator


def create_span_context(
    span_name: str,
    tracer_name: str = "omninode_bridge",
    add_correlation: bool = True,
):
    """Create a span context manager for manual tracing.

    Args:
        span_name: Name of the span
        tracer_name: Tracer name
        add_correlation: Add correlation context to span

    Returns:
        Span context manager

    Example:
        with create_span_context("process_stage") as span:
            span.set_attribute("stage.name", "validation")
            result = perform_processing()
            span.add_event("processing_completed")
    """
    tracer = get_tracer(tracer_name)
    span = tracer.start_span(span_name)

    # Add correlation context
    if add_correlation:
        correlation_ctx = get_correlation_context()
        for key, value in correlation_ctx.items():
            if value is not None:
                span.set_attribute(f"omninode.{key}", str(value))

    return trace.use_span(span, end_on_exit=True)


def extract_trace_context(headers: dict[str, str]) -> dict[str, Any]:
    """Extract trace context from HTTP headers.

    Args:
        headers: HTTP headers dictionary

    Returns:
        Dictionary with trace context
    """
    from opentelemetry.propagate import extract

    return extract(headers)


def inject_trace_context(headers: dict[str, str]) -> dict[str, str]:
    """Inject trace context into HTTP headers.

    Args:
        headers: HTTP headers dictionary

    Returns:
        Updated headers with trace context
    """
    from opentelemetry.propagate import inject

    carrier = headers.copy()
    inject(carrier)
    return carrier
