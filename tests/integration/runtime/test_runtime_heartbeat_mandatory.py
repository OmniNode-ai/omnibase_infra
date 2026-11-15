# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests: the runtime heartbeat is mandatory (OMN-24496).

A real ``RuntimeHostProcess`` runs on an ``EventBusInmemory`` with a real
``ServiceNodeIntrospection`` built from the registration orchestrator
contract — the same construction the kernel uses — with the heartbeat period
injected small (``HEARTBEAT_S``) so beats arrive while the test waits.

Test Coverage:
- Golden: startup publishes the runtime's own introspection exactly once,
  then the heartbeat task publishes node-heartbeat events for the same
  ``node_id``, and no heartbeat lands before the introspection that
  announced the node.
- Error chain: a runtime whose heartbeat task cannot start refuses to start
  with ``RuntimeHeartbeatRequiredError``, and announces no node.
- Failing control: the old behaviour — no introspection service — starts and
  publishes nothing, which is why the kernel must refuse to build no service.

Related:
    - OMN-24496: Runtime heartbeat is mandatory
    - src/omnibase_infra/handlers/handler_runtime_introspection_startup.py
    - src/omnibase_infra/runtime/runtime_host_process.py
    - src/omnibase_infra/services/service_node_introspection.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from omnibase_core.enums.enum_announced_node_kind import EnumAnnouncedNodeKind
from omnibase_core.enums.enum_announcement_operation import (
    EnumAnnouncementOperation,
)
from omnibase_core.models.events.contract_registration.model_announced_package import (
    ModelAnnouncedPackage,
)
from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_infra.enums import (
    EnumIntrospectionReason,
    EnumRuntimeHeartbeatRefusalReason,
)
from omnibase_infra.event_bus import ModelEventMessage
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers.handler_runtime_introspection_startup import (
    HandlerRuntimeIntrospectionStartup,
    RuntimeHeartbeatRequiredError,
)
from omnibase_infra.models.runtime import ModelRuntimeIntrospectionConfig
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.services.service_node_introspection import ServiceNodeIntrospection
from omnibase_infra.topics import SUFFIX_NODE_HEARTBEAT, SUFFIX_NODE_INTROSPECTION
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

pytestmark = pytest.mark.integration

# The injected heartbeat period H (seconds): small so that beats arrive while
# the test waits.
HEARTBEAT_S = 0.2

# Wait tolerance T (seconds) added on top of the wait window.
TOLERANCE_S = 0.5

# The contract directory the kernel builds its introspection service from.
CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_registration_orchestrator"
)

ANNOUNCER_ID = UUID("00000000-0000-4000-8000-000000000001")
ANNOUNCED_AT = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
ANNOUNCED_NODE_NAME = "node_example_effect"


class ServiceWithInjectedHeartbeatPeriod(ServiceNodeIntrospection):
    """ServiceNodeIntrospection whose heartbeat period is injected small.

    Built through ``from_contract_dir`` exactly as the kernel builds its
    service (the factory constructs ``cls()``, so the subclass is what gets
    built); only ``start_heartbeat_task`` is narrowed to start the heartbeat
    loop at the injected :data:`HEARTBEAT_S` period instead of the runtime
    default, with the registry and ACK listeners left off.
    """

    async def start_heartbeat_task(self) -> None:
        await self.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=HEARTBEAT_S,
            enable_registry_listener=False,
            enable_ack_listener=False,
        )


# =============================================================================
# Helpers
# =============================================================================


def make_fast_heartbeat_service(
    event_bus: EventBusInmemory,
) -> ServiceNodeIntrospection:
    """Build the introspection service the kernel builds, with H injected."""
    return ServiceWithInjectedHeartbeatPeriod.from_contract_dir(
        contracts_dir=CONTRACTS_DIR,
        event_bus=event_bus,
        node_name="heartbeat-test-runtime",
        environment="local",
    )


def make_announcement(entity_id: str = ANNOUNCED_NODE_NAME) -> ModelNodeAnnouncement:
    """Build one node announcement to inject into the runtime."""
    return ModelNodeAnnouncement(
        entity_id=entity_id,
        operation=EnumAnnouncementOperation.ANNOUNCE,
        content_hash="a" * 64,
        package=ModelAnnouncedPackage(distribution="omnibase-example", version="1.2.3"),
        node_kind=EnumAnnouncedNodeKind.NODE,
        declared_topics=("example.topic.v1",),
        announcer_id=ANNOUNCER_ID,
        generation=4,
        announced_at=ANNOUNCED_AT,
    )


def make_introspection_config() -> ModelRuntimeIntrospectionConfig:
    """Deterministic introspection config: no jitter, long throttle."""
    return ModelRuntimeIntrospectionConfig(
        enabled=True,
        jitter_max_ms=0,
        throttle_min_interval_s=10,
    )


def load_payload(message: ModelEventMessage) -> dict[str, object]:
    """Decode one bus history message into its event payload dict.

    Publishers wrap events in a ``ModelEventEnvelope`` (a ``payload`` key
    inside the serialized JSON); the raw-bytes fallback path publishes the
    bare event. Handle both.
    """
    decoded = json.loads(message.value)
    if isinstance(decoded, dict) and isinstance(decoded.get("payload"), dict):
        return decoded["payload"]
    assert isinstance(decoded, dict), (
        f"unexpected history value type: {type(decoded).__name__}"
    )
    return decoded


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def event_bus() -> EventBusInmemory:
    """Create an in-memory event bus for testing.

    Returns:
        EventBusInmemory instance.
    """
    return EventBusInmemory()


# =============================================================================
# Golden: introspection once, then heartbeats
# =============================================================================


class TestRuntimePublishesOwnIntrospectionThenHeartbeats:
    """Golden path: the runtime announces itself, then stays alive."""

    @pytest.mark.asyncio
    async def test_runtime_publishes_own_introspection_then_two_heartbeats(
        self, event_bus: EventBusInmemory
    ) -> None:
        """A started runtime publishes its own introspection, then heartbeats.

        Verifies:
        - exactly one node-introspection event for the runtime's own node_id,
          with reason ``startup``
        - at least two node-heartbeat events with the same node_id after
          ``2 * HEARTBEAT_S + TOLERANCE_S``
        - the first heartbeat is not earlier in the history than the
          introspection event (compare via the publish-time header timestamp,
          the only cross-topic ordering the two separate topic histories give)
        """
        HandlerRuntimeIntrospectionStartup.reset_refusal_counts()
        service = make_fast_heartbeat_service(event_bus)
        # The service's own identity is fixed at construction; every event it
        # publishes (introspection and heartbeat) must carry it.
        runtime_node_id = service._introspection_node_id
        assert runtime_node_id is not None

        process = RuntimeHostProcess(
            event_bus=event_bus,
            config=make_runtime_config(),
            introspection_service=service,
            introspection_config=make_introspection_config(),
        )
        seed_mock_handlers(process)

        try:
            await process.start()
            assert process.is_running

            await asyncio.sleep(2 * HEARTBEAT_S + TOLERANCE_S)

            introspections = await event_bus.get_event_history(
                topic=SUFFIX_NODE_INTROSPECTION
            )
            introspection_payloads = [load_payload(m) for m in introspections]
            assert len(introspection_payloads) == 1
            own = introspection_payloads[0]
            assert own["node_id"] == str(runtime_node_id)
            assert own["reason"] == EnumIntrospectionReason.STARTUP.value

            heartbeats = await event_bus.get_event_history(topic=SUFFIX_NODE_HEARTBEAT)
            heartbeat_payloads = [load_payload(m) for m in heartbeats]
            assert len(heartbeat_payloads) >= 2
            assert all(
                payload["node_id"] == str(runtime_node_id)
                for payload in heartbeat_payloads
            )

            # The node was announced before it was kept alive: the first
            # heartbeat must not precede the introspection event.
            assert (
                heartbeats[0].headers.timestamp >= introspections[0].headers.timestamp
            )

        finally:
            await process.stop()


# =============================================================================
# Error chain: a heartbeat that cannot start refuses the runtime
# =============================================================================


class TestRuntimeRefusesWhenHeartbeatCannotStart:
    """The mandatory rule: no heartbeat task, no runtime, no announcement."""

    @pytest.mark.asyncio
    async def test_runtime_whose_heartbeat_cannot_start_refuses_and_announces_no_node(
        self, event_bus: EventBusInmemory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A runtime whose heartbeat task fails to start refuses to start.

        The handler starts the heartbeat right AFTER the runtime's own
        startup introspection is published and BEFORE any announced node is
        published. So when ``start_heartbeat_task`` raises, the bus holds
        exactly one introspection event — the runtime's own — and nothing
        for the announced node: the refusal propagates out of
        ``process.start()`` as a typed, counted
        ``RuntimeHeartbeatRequiredError``.
        """
        HandlerRuntimeIntrospectionStartup.reset_refusal_counts()
        service = make_fast_heartbeat_service(event_bus)
        runtime_node_id = service._introspection_node_id
        assert runtime_node_id is not None
        monkeypatch.setattr(
            service,
            "start_heartbeat_task",
            AsyncMock(side_effect=RuntimeError("boom")),
        )

        process = RuntimeHostProcess(
            event_bus=event_bus,
            config=make_runtime_config(),
            introspection_service=service,
            introspection_config=make_introspection_config(),
            node_announcements=(make_announcement(),),
        )
        seed_mock_handlers(process)

        try:
            with pytest.raises(RuntimeHeartbeatRequiredError) as exc_info:
                await process.start()

            refusal = exc_info.value.refusal
            assert refusal.reason is (
                EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_START_FAILED
            )
            assert refusal.count == 1
            assert (
                HandlerRuntimeIntrospectionStartup.refusal_count(
                    EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_START_FAILED
                )
                == 1
            )

            # No heartbeat task was started, so no liveness was published.
            heartbeats = await event_bus.get_event_history(topic=SUFFIX_NODE_HEARTBEAT)
            assert heartbeats == []

            # Exactly one introspection event: the runtime's own, published
            # before the heartbeat was attempted. The announced node was
            # never built or published.
            introspections = await event_bus.get_event_history(
                topic=SUFFIX_NODE_INTROSPECTION
            )
            payloads = [load_payload(m) for m in introspections]
            assert len(payloads) == 1
            assert payloads[0]["node_id"] == str(runtime_node_id)
            assert all(
                payload["node_name"] != ANNOUNCED_NODE_NAME for payload in payloads
            )

        finally:
            # stop() must not raise even after a refused start.
            await process.stop()


# =============================================================================
# Failing control: the old behaviour runs on with no service at all
# =============================================================================


class TestControlOldBehaviourNoService:
    """Documents why the kernel must refuse to build no service."""

    @pytest.mark.asyncio
    async def test_control_old_behaviour_logs_and_runs_on_with_no_service(
        self, event_bus: EventBusInmemory
    ) -> None:
        """Control: with no introspection service the runtime starts anyway.

        This is the pre-OMN-24496 behaviour: ``introspection_service=None``
        (what the kernel used to fall back to when it could not build the
        service) makes ``RuntimeHostProcess.start()`` succeed and publish
        nothing — no introspection, no heartbeat — leaving the runtime
        silently absent and stale in the registry. The kernel now never
        passes ``None``: it builds the service through
        ``build_required_service``, which refuses the startup instead.
        """
        HandlerRuntimeIntrospectionStartup.reset_refusal_counts()
        process = RuntimeHostProcess(
            event_bus=event_bus,
            config=make_runtime_config(),
            introspection_service=None,
            introspection_config=make_introspection_config(),
        )
        seed_mock_handlers(process)

        try:
            await process.start()
            assert process.is_running

            assert (
                await event_bus.get_event_history(topic=SUFFIX_NODE_INTROSPECTION) == []
            )
            assert await event_bus.get_event_history(topic=SUFFIX_NODE_HEARTBEAT) == []

        finally:
            await process.stop()
