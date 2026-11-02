# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OpenTelemetry span around ``MessageDispatchEngine.dispatch`` (OMN-22097).

``traced_dispatch`` opens one span per dispatch call carrying the topic, the
message category, the success flag and the duration in milliseconds. The tracer
is acquired inside the call, after ``configure_tracing`` has had its chance to
install a provider; with no ``TracerProvider`` set the wrapper adds no span.

This leaf helper imports OpenTelemetry and the dispatch result model only. It
does not import runtime modules, so the runtime can use it without an import cycle.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable

from opentelemetry import trace
from opentelemetry.trace import NoOpTracerProvider, ProxyTracerProvider, StatusCode

from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult

_TRACER_NAME = "omnibase_infra.dispatch"
_SPAN_NAME = "message_dispatch_engine.dispatch"


def traced_dispatch[**P](
    dispatch: Callable[P, Awaitable[ModelDispatchResult]],
) -> Callable[P, Awaitable[ModelDispatchResult]]:
    """Wrap an async ``dispatch(self, topic, envelope, ...)`` in one span."""

    @functools.wraps(dispatch)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> ModelDispatchResult:
        provider = trace.get_tracer_provider()
        if isinstance(provider, ProxyTracerProvider | NoOpTracerProvider):
            return await dispatch(*args, **kwargs)
        topic = str(kwargs["topic"] if "topic" in kwargs else args[1])
        started = time.perf_counter()
        with provider.get_tracer(_TRACER_NAME).start_as_current_span(
            _SPAN_NAME, attributes={"omninode.dispatch.topic": topic}
        ) as span:
            try:
                result = await dispatch(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, type(exc).__name__)
                span.set_attribute("omninode.dispatch.success", False)
                span.set_attribute(
                    "omninode.dispatch.duration_ms",
                    (time.perf_counter() - started) * 1000,
                )
                raise
            category = result.message_category
            span.set_attribute(
                "omninode.dispatch.category",
                category.value if category is not None else "",
            )
            span.set_attribute("omninode.dispatch.success", result.is_successful())
            span.set_attribute(
                "omninode.dispatch.duration_ms",
                result.duration_ms
                if result.duration_ms is not None
                else (time.perf_counter() - started) * 1000,
            )
            return result

    return wrapper
