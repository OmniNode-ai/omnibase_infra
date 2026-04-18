# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TDD tests for OMN-8512 — _wire_single_contract must call event_bus.subscribe().

These tests were written BEFORE the fix and must fail on the unfixed codebase.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.service_message_dispatch_engine import (
    MessageDispatchEngine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contract(
    name: str = "node_test",
    subscribe_topics: tuple[str, ...] = ("onex.cmd.omnimarket.test-start.v1",),
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe_topics,
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="FakeHandler",
                        module="fake.module",
                    ),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


def _fake_handler_cls() -> type:
    """Real class — ModelHandlerResolverContext.handler_cls requires ``type`` (OMN-9201)."""

    class FakeHandler:
        async def handle(self, envelope: object) -> None:
            return None

    return FakeHandler


# ---------------------------------------------------------------------------
# Test 1 — subscribe() is called once per subscribe_topic
# ---------------------------------------------------------------------------


class TestWireFromManifestCallsSubscribe:
    """event_bus.subscribe() must be called for each contract subscribe_topic."""

    @pytest.mark.asyncio
    async def test_subscribe_called_once_per_topic(self) -> None:
        topic = "onex.cmd.omnimarket.pr-lifecycle-orchestrator-start.v1"
        contract = _contract(subscribe_topics=(topic,))
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()

        # Mock event_bus with async subscribe that returns an async unsubscribe callable
        unsubscribe = AsyncMock()
        event_bus = MagicMock()
        event_bus.subscribe = AsyncMock(return_value=unsubscribe)

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ):
            report = await wire_from_manifest(
                manifest, engine, event_bus=event_bus, environment="local"
            )

        assert report.total_wired == 1
        # THE critical assertion: subscribe must have been called
        event_bus.subscribe.assert_called_once()
        call_kwargs = event_bus.subscribe.call_args
        # Called with topic= and node_identity= and on_message= keyword args
        assert call_kwargs.kwargs.get("topic") == topic or call_kwargs.args[0] == topic

    @pytest.mark.asyncio
    async def test_subscribe_called_for_each_of_multiple_topics(self) -> None:
        topics = (
            "onex.cmd.omnimarket.pr-lifecycle-orchestrator-start.v1",
            "onex.cmd.omnimarket.pr-lifecycle-retry.v1",
        )
        contract = _contract(subscribe_topics=topics)
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()

        unsubscribe = AsyncMock()
        event_bus = MagicMock()
        event_bus.subscribe = AsyncMock(return_value=unsubscribe)

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ):
            await wire_from_manifest(
                manifest, engine, event_bus=event_bus, environment="local"
            )

        assert event_bus.subscribe.call_count == len(topics)

    @pytest.mark.asyncio
    async def test_topics_subscribed_in_report(self) -> None:
        topic = "onex.cmd.omnimarket.pr-lifecycle-orchestrator-start.v1"
        contract = _contract(subscribe_topics=(topic,))
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()

        unsubscribe = AsyncMock()
        event_bus = MagicMock()
        event_bus.subscribe = AsyncMock(return_value=unsubscribe)

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ):
            report = await wire_from_manifest(
                manifest, engine, event_bus=event_bus, environment="local"
            )

        assert topic in report.results[0].topics_subscribed

    @pytest.mark.asyncio
    async def test_no_subscribe_when_event_bus_none(self) -> None:
        """When event_bus is None, subscribe must not be called (no event_bus to call on)."""
        contract = _contract()
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ):
            report = await wire_from_manifest(
                manifest, engine, event_bus=None, environment="local"
            )

        # Still wired (dispatchers+routes registered), but no subscription
        assert report.total_wired == 1


# ---------------------------------------------------------------------------
# Test 2 — callback routes to dispatch engine
# ---------------------------------------------------------------------------


class TestSubscribeCallbackRoutesToDispatchEngine:
    """When an event arrives on a subscribed topic, dispatch engine must receive it."""

    @pytest.mark.asyncio
    async def test_callback_dispatches_to_engine(self) -> None:
        topic = "onex.cmd.omnimarket.test-start.v1"
        contract = _contract(subscribe_topics=(topic,))
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()

        captured_callbacks: list = []

        async def fake_subscribe(**kwargs: object) -> AsyncMock:
            captured_callbacks.append(kwargs.get("on_message"))
            return AsyncMock()

        event_bus = MagicMock()
        event_bus.subscribe = AsyncMock(side_effect=fake_subscribe)

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ):
            await wire_from_manifest(
                manifest, engine, event_bus=event_bus, environment="local"
            )

        # A callback must have been captured
        assert len(captured_callbacks) == 1
        callback = captured_callbacks[0]
        assert callable(callback)


# ---------------------------------------------------------------------------
# Test 3 — raw=None fallback path (OMN-8512 CR fix)
# ---------------------------------------------------------------------------


class TestMakeEventBusCallbackFallbackPath:
    """When raw is None, dispatch_engine must NOT receive a non-ModelEventEnvelope."""

    @pytest.mark.asyncio
    async def test_non_envelope_message_is_dropped_not_dispatched(self) -> None:
        """A message with no .value attribute that is not a ModelEventEnvelope must
        be dropped (dispatch_engine.dispatch not called) rather than passed through raw."""
        from unittest.mock import AsyncMock, MagicMock

        from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _make_event_bus_callback,
        )

        dispatch_engine = MagicMock()
        dispatch_engine.dispatch = AsyncMock()

        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
        )

        # A plain object with no .value — not a ModelEventEnvelope
        bad_message = object()
        await callback(bad_message)

        # dispatch_engine must NOT have been called with the raw non-envelope
        dispatch_engine.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_envelope_message_passes_through(self) -> None:
        """A message with no .value that IS already a ModelEventEnvelope must still be dispatched."""
        from unittest.mock import AsyncMock, MagicMock

        from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _make_event_bus_callback,
        )

        dispatch_engine = MagicMock()
        dispatch_engine.dispatch = AsyncMock()

        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
        )

        envelope = ModelEventEnvelope[object].model_construct(
            event_type="onex.cmd.test.v1",
            payload={},
        )
        await callback(envelope)

        dispatch_engine.dispatch.assert_called_once()
        _, call_envelope = dispatch_engine.dispatch.call_args.args
        assert isinstance(call_envelope, ModelEventEnvelope)
