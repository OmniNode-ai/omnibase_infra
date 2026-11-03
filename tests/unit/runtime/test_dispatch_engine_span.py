# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One OpenTelemetry span per ``MessageDispatchEngine.dispatch`` call (OMN-22097)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.unit

_TOPIC = "dev.user.events.v1"


class _UserCreated:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id


def _engine(*, fail: bool = False) -> MessageDispatchEngine:
    engine = MessageDispatchEngine()

    async def dispatcher(envelope: ModelEventEnvelope[object]) -> str:
        if fail:
            raise RuntimeError("dispatcher boom")
        return "output.topic.v1"

    engine.register_dispatcher(
        dispatcher_id="event-handler",
        dispatcher=dispatcher,
        category=EnumMessageCategory.EVENT,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="event-route",
            topic_pattern="*.user.events.*",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id="event-handler",
        )
    )
    engine.freeze()
    return engine


def _envelope() -> ModelEventEnvelope[object]:
    return ModelEventEnvelope(payload=_UserCreated("u-1"))


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """Install a provider for the duration of one test, without touching the global once-only slot."""
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    yield memory
    provider.shutdown()


def _spans(exporter: InMemorySpanExporter) -> tuple[ReadableSpan, ...]:
    return tuple(exporter.get_finished_spans())


class TestDispatchSpan:
    @pytest.mark.asyncio
    async def test_one_span_with_topic_category_success_and_duration(
        self, exporter: InMemorySpanExporter
    ) -> None:
        result = await _engine().dispatch(_TOPIC, _envelope())

        (span,) = _spans(exporter)
        assert span.attributes is not None
        assert span.attributes["omninode.dispatch.topic"] == _TOPIC
        assert span.attributes["omninode.dispatch.category"] == "event"
        assert span.attributes["omninode.dispatch.success"] is True
        assert span.attributes["omninode.dispatch.duration_ms"] == result.duration_ms

    @pytest.mark.asyncio
    async def test_failed_dispatch_reports_success_false(
        self, exporter: InMemorySpanExporter
    ) -> None:
        result = await _engine(fail=True).dispatch(_TOPIC, _envelope())

        assert not result.is_successful()
        (span,) = _spans(exporter)
        assert span.attributes is not None
        assert span.attributes["omninode.dispatch.success"] is False

    @pytest.mark.asyncio
    async def test_raising_dispatch_records_error_and_reraises(
        self, exporter: InMemorySpanExporter
    ) -> None:
        with pytest.raises(ModelOnexError):
            await _engine().dispatch("  ", _envelope())

        (span,) = _spans(exporter)
        assert span.attributes is not None
        assert span.attributes["omninode.dispatch.success"] is False
        assert span.status.status_code is trace.StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_topic_keyword_argument_is_recorded(
        self, exporter: InMemorySpanExporter
    ) -> None:
        await _engine().dispatch(topic=_TOPIC, envelope=_envelope())

        (span,) = _spans(exporter)
        assert span.attributes is not None
        assert span.attributes["omninode.dispatch.topic"] == _TOPIC

    @pytest.mark.asyncio
    async def test_call_without_topic_raises_the_engines_own_error(
        self, exporter: InMemorySpanExporter
    ) -> None:
        """Argument binding failures are the engine's to report, not the wrapper's."""
        dispatch: Callable[..., Awaitable[object]] = _engine().dispatch
        with pytest.raises(TypeError, match="topic"):
            await dispatch(envelope=_envelope())
        with pytest.raises(TypeError):
            await dispatch()

        assert _spans(exporter) == ()

    @pytest.mark.asyncio
    async def test_no_span_without_a_tracer_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unset = trace.ProxyTracerProvider()
        monkeypatch.setattr(trace, "get_tracer_provider", lambda: unset)

        result = await _engine().dispatch(_TOPIC, _envelope())

        assert result.is_successful()
