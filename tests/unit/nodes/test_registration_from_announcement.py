# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registration from a node announcement (OMN-21590, hot loading plan task A2).

Covers the adapter (announcement + runtime identity -> introspection event),
the real registration orchestrator handler consuming that event, and heartbeat
startup after boot with and without injected announcements.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.enums.enum_announced_node_kind import EnumAnnouncedNodeKind
from omnibase_core.enums.enum_announcement_operation import EnumAnnouncementOperation
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.contract_registration.model_announced_package import (
    ModelAnnouncedPackage,
)
from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumIntrospectionReason
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers.handler_registration_from_announcement import (
    HandlerRegistrationFromAnnouncement,
)
from omnibase_infra.handlers.handler_runtime_introspection_startup import (
    HandlerRuntimeIntrospectionStartup,
)
from omnibase_infra.models.model_registration_from_announcement_request import (
    ModelRegistrationFromAnnouncementRequest,
)
from omnibase_infra.models.model_runtime_identity import ModelRuntimeIdentity
from omnibase_infra.models.model_runtime_introspection_startup_request import (
    ModelRuntimeIntrospectionStartupRequest,
)
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.models.registration.events import ModelNodeRegistrationInitiated
from omnibase_infra.models.runtime import ModelRuntimeIntrospectionConfig
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.topics import SUFFIX_NODE_INTROSPECTION
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

TEST_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
ANNOUNCER = UUID("00000000-0000-4000-8000-000000000001")
RUNTIME_NODE_ID = UUID("00000000-0000-4000-8000-0000000000aa")


def make_announcement(
    entity_id: str = "node_example_effect",
    operation: EnumAnnouncementOperation = EnumAnnouncementOperation.ANNOUNCE,
    version: str = "1.2.3",
    generation: int = 4,
) -> ModelNodeAnnouncement:
    return ModelNodeAnnouncement(
        entity_id=entity_id,
        operation=operation,
        content_hash="a" * 64,
        package=ModelAnnouncedPackage(distribution="omnibase-example", version=version),
        node_kind=EnumAnnouncedNodeKind.NODE,
        declared_topics=("example.topic.v1",),
        announcer_id=ANNOUNCER,
        generation=generation,
        announced_at=TEST_NOW,
    )


def make_identity() -> ModelRuntimeIdentity:
    return ModelRuntimeIdentity(
        node_id=RUNTIME_NODE_ID,
        node_type=EnumNodeKind.EFFECT,
        endpoints={"health": "http://localhost:8085/health"},
        node_role="runtime",
        network_id="net-1",
        deployment_id="deploy-1",
    )


def adapt(
    announcement: ModelNodeAnnouncement, identity: ModelRuntimeIdentity
) -> ModelNodeIntrospectionEvent | None:
    return (
        HandlerRegistrationFromAnnouncement()
        .handle(
            ModelRegistrationFromAnnouncementRequest(
                announcement=announcement, identity=identity
            )
        )
        .event
    )


class TestAdapter:
    def test_same_inputs_give_same_event(self) -> None:
        first = adapt(make_announcement(), make_identity())
        second = adapt(make_announcement(), make_identity())
        assert first is not None
        assert first == second

    def test_field_mapping(self) -> None:
        announcement = make_announcement()
        event = adapt(announcement, make_identity())
        assert event is not None
        assert event.node_id == RUNTIME_NODE_ID
        assert event.node_name == "node_example_effect"
        assert event.node_type == EnumNodeKind.EFFECT
        assert event.node_version == ModelSemVer(major=1, minor=2, patch=3)
        assert event.endpoints == {"health": "http://localhost:8085/health"}
        assert event.reason == EnumIntrospectionReason.STARTUP
        assert event.timestamp == announcement.announced_at
        assert event.epoch == announcement.generation
        assert event.node_role == "runtime"
        assert event.network_id == "net-1"
        assert event.deployment_id == "deploy-1"
        assert event.metadata.version == "1.2.3"

    def test_correlation_id_depends_on_announcement(self) -> None:
        a = adapt(make_announcement(generation=1), make_identity())
        b = adapt(make_announcement(generation=2), make_identity())
        assert a is not None and b is not None
        assert a.correlation_id != b.correlation_id

    def test_non_semver_package_version_falls_back_to_identity_version(self) -> None:
        event = adapt(make_announcement(version="not-a-version"), make_identity())
        assert event is not None
        assert event.node_version == make_identity().node_version

    def test_withdraw_builds_no_event(self) -> None:
        result = HandlerRegistrationFromAnnouncement().handle(
            ModelRegistrationFromAnnouncementRequest(
                announcement=make_announcement(
                    operation=EnumAnnouncementOperation.WITHDRAW
                ),
                identity=make_identity(),
            )
        )
        assert result.event is None
        assert result.skipped_reason

    def test_identity_round_trips_from_introspection_event(self) -> None:
        event = adapt(make_announcement(version="1.0.0"), make_identity())
        assert event is not None
        assert ModelRuntimeIdentity.from_introspection_event(event) == make_identity()


class TestRegistrationOrchestratorConsumesAnnouncement:
    @pytest.mark.asyncio
    async def test_announced_node_registers(self) -> None:
        reader = AsyncMock(spec=ProjectionReaderRegistration)
        reader.get_entity_state = AsyncMock(return_value=None)  # new node
        handler = HandlerNodeIntrospected(reader, RegistrationReducerService())

        event = adapt(make_announcement(), make_identity())
        assert event is not None
        correlation_id = uuid4()
        envelope: ModelEventEnvelope[ModelNodeIntrospectionEvent] = ModelEventEnvelope(
            envelope_id=uuid4(),
            payload=event,
            envelope_timestamp=TEST_NOW,
            correlation_id=correlation_id,
            source_tool="test",
        )

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-node-introspected"
        assert len(output.events) == 2
        initiated = output.events[0]
        assert isinstance(initiated, ModelNodeRegistrationInitiated)
        assert initiated.node_id == RUNTIME_NODE_ID
        assert initiated.entity_id == RUNTIME_NODE_ID
        assert initiated.correlation_id == correlation_id
        assert initiated.causation_id == event.correlation_id
        assert initiated.registration_attempt_id is not None
        assert initiated.emitted_at == TEST_NOW


def make_service() -> MagicMock:
    service = MagicMock()
    service.publish_introspection = AsyncMock()
    service.start_heartbeat_task = AsyncMock()
    service.stop_heartbeat_task = AsyncMock()
    service.get_introspection_data = AsyncMock(
        return_value=ModelNodeIntrospectionEvent(
            node_id=RUNTIME_NODE_ID,
            node_type=EnumNodeKind.EFFECT,
            correlation_id=uuid4(),
            timestamp=TEST_NOW,
            endpoints={"health": "http://localhost:8085/health"},
        )
    )
    return service


def make_config() -> ModelRuntimeIntrospectionConfig:
    return ModelRuntimeIntrospectionConfig(
        enabled=True, jitter_max_ms=0, throttle_min_interval_s=10
    )


def make_process(
    service: MagicMock,
    announcements: tuple[ModelNodeAnnouncement, ...] = (),
    event_bus: EventBusInmemory | None = None,
) -> RuntimeHostProcess:
    process = RuntimeHostProcess(
        event_bus=event_bus or EventBusInmemory(),
        config=make_runtime_config(),
        introspection_service=service,
        introspection_config=make_config(),
        node_announcements=announcements,
    )
    seed_mock_handlers(process)
    return process


def make_request(
    announcements: tuple[ModelNodeAnnouncement, ...] = (),
    last_introspection_time: float | None = None,
) -> ModelRuntimeIntrospectionStartupRequest:
    return ModelRuntimeIntrospectionStartupRequest(
        correlation_id=uuid4(),
        announcements=announcements,
        last_introspection_time=last_introspection_time,
    )


def make_bus() -> MagicMock:
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.publish_envelope = AsyncMock()
    bus.publish = AsyncMock()
    return bus


class TestStartupHandlerWithoutAnnouncements:
    @pytest.mark.asyncio
    async def test_publishes_through_the_service(self) -> None:
        service = make_service()
        handler = HandlerRuntimeIntrospectionStartup(service, make_config())
        request = make_request()

        result = await handler.handle(request)

        assert result.published
        assert result.last_introspection_time is not None
        service.publish_introspection.assert_awaited_once_with(
            reason=EnumIntrospectionReason.STARTUP,
            correlation_id=request.correlation_id,
        )

    @pytest.mark.asyncio
    async def test_no_service_publishes_nothing(self) -> None:
        handler = HandlerRuntimeIntrospectionStartup(None, make_config())
        result = await handler.handle(make_request())
        assert not result.published
        assert result.last_introspection_time is None

    @pytest.mark.asyncio
    async def test_disabled_publishes_nothing(self) -> None:
        service = make_service()
        handler = HandlerRuntimeIntrospectionStartup(
            service, ModelRuntimeIntrospectionConfig(enabled=False)
        )
        result = await handler.handle(make_request())
        assert not result.published
        service.publish_introspection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_recent_publish_is_throttled(self) -> None:
        service = make_service()
        handler = HandlerRuntimeIntrospectionStartup(service, make_config())
        recent = time.monotonic()

        result = await handler.handle(make_request(last_introspection_time=recent))

        assert not result.published
        assert result.last_introspection_time == recent
        service.publish_introspection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failure_keeps_the_previous_throttle_time(self) -> None:
        service = make_service()
        service.publish_introspection.side_effect = RuntimeError("bus down")
        handler = HandlerRuntimeIntrospectionStartup(service, make_config())

        result = await handler.handle(make_request())

        assert not result.published
        assert result.last_introspection_time is None

    @pytest.mark.asyncio
    async def test_heartbeat_failure_is_logged_not_raised(self) -> None:
        service = make_service()
        service.start_heartbeat_task.side_effect = RuntimeError("no loop")
        handler = HandlerRuntimeIntrospectionStartup(service, make_config())
        await handler.start_heartbeat(uuid4())
        service.start_heartbeat_task.assert_awaited_once()


class TestStartupHandlerWithAnnouncements:
    @pytest.mark.asyncio
    async def test_publishes_one_envelope_per_announce_record(self) -> None:
        service = make_service()
        bus = make_bus()
        handler = HandlerRuntimeIntrospectionStartup(service, make_config(), bus)
        announcements = (
            make_announcement("node_a_effect"),
            make_announcement("node_b_effect"),
            make_announcement("node_gone_effect", EnumAnnouncementOperation.WITHDRAW),
        )

        result = await handler.handle(make_request(announcements))

        assert result.published
        service.publish_introspection.assert_not_awaited()
        calls = bus.publish_envelope.await_args_list
        assert [c.kwargs["topic"] for c in calls] == [SUFFIX_NODE_INTROSPECTION] * 2
        envelopes: list[ModelEventEnvelope[object]] = [
            c.kwargs["envelope"] for c in calls
        ]
        assert all(isinstance(e, ModelEventEnvelope) for e in envelopes)
        payloads = [
            e.payload
            for e in envelopes
            if isinstance(e.payload, ModelNodeIntrospectionEvent)
        ]
        assert len(payloads) == 2
        assert [p.node_name for p in payloads] == ["node_a_effect", "node_b_effect"]
        assert all(p.node_id == RUNTIME_NODE_ID for p in payloads)
        assert [e.correlation_id for e in envelopes] == [
            p.correlation_id for p in payloads
        ]

    @pytest.mark.asyncio
    async def test_falls_back_to_raw_bytes_without_publish_envelope(self) -> None:
        class RawBus:
            def __init__(self) -> None:
                self.published: list[tuple[str, bytes | None, bytes]] = []

            async def publish(
                self, topic: str, key: bytes | None, value: bytes
            ) -> None:
                self.published.append((topic, key, value))

        bus = RawBus()
        handler = HandlerRuntimeIntrospectionStartup(
            make_service(), make_config(), cast("ProtocolEventBusLike", bus)
        )

        result = await handler.handle(make_request((make_announcement(),)))

        assert result.published
        ((topic, key, value),) = bus.published
        assert topic == SUFFIX_NODE_INTROSPECTION
        assert key == str(RUNTIME_NODE_ID).encode("utf-8")
        assert json.loads(value)["node_name"] == "node_example_effect"

    @pytest.mark.asyncio
    async def test_bus_failure_is_not_a_publish(self) -> None:
        bus = make_bus()
        bus.publish_envelope.side_effect = RuntimeError("bus down")
        handler = HandlerRuntimeIntrospectionStartup(make_service(), make_config(), bus)

        result = await handler.handle(make_request((make_announcement(),)))

        assert not result.published

    @pytest.mark.asyncio
    async def test_service_that_cannot_report_identity_is_not_a_publish(self) -> None:
        class ServiceWithoutIdentity:
            async def publish_introspection(
                self,
                reason: EnumIntrospectionReason,
                correlation_id: UUID | None = None,
            ) -> None:
                return None

            async def start_heartbeat_task(self) -> None:
                return None

            async def stop_heartbeat_task(self) -> None:
                return None

        handler = HandlerRuntimeIntrospectionStartup(
            ServiceWithoutIdentity(), make_config(), make_bus()
        )

        result = await handler.handle(make_request((make_announcement(),)))

        assert not result.published


class TestHeartbeatAfterBoot:
    @pytest.mark.asyncio
    async def test_heartbeat_started_once_without_announcements(self) -> None:
        service = make_service()
        process = make_process(service)
        try:
            await process.start()
            service.publish_introspection.assert_awaited_once()
            service.start_heartbeat_task.assert_awaited_once()
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_heartbeat_started_once_with_announcements(self) -> None:
        service = make_service()
        bus = EventBusInmemory()
        announcements = (
            make_announcement("node_a_effect"),
            make_announcement("node_b_effect"),
        )
        process = make_process(service, announcements, bus)
        try:
            await process.start()
            service.publish_introspection.assert_not_awaited()
            history = await bus.get_event_history(topic=SUFFIX_NODE_INTROSPECTION)
            published = [json.loads(m.value)["payload"] for m in history]
            assert [p["node_name"] for p in published] == [
                "node_a_effect",
                "node_b_effect",
            ]
            service.start_heartbeat_task.assert_awaited_once()
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_heartbeat_not_started_when_publish_fails(self) -> None:
        service = make_service()
        service.publish_introspection.side_effect = RuntimeError("bus down")
        process = make_process(service)
        try:
            await process.start()
            assert process.is_running
            service.start_heartbeat_task.assert_not_awaited()
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_runtime_throttles_from_the_handler_result(self) -> None:
        service = make_service()
        process = make_process(service)
        try:
            await process.start()
            first = process._last_introspection_time
            assert first is not None
            await process._publish_introspection_with_jitter(correlation_id=uuid4())
            assert process._last_introspection_time == first
            service.publish_introspection.assert_awaited_once()
        finally:
            await process.stop()
