# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A node-selected kernel does not start the Pattern B broker (OMN-22311).

Recorded on the primary dev lane 2026-11-03T07:42Z. The claude hook-events writer
(OMN-19513) is the runtime kernel booted with the environment below, copied from
the running container ``omnimarket-projection-claude-hook-events-writer``:

    RUNTIME_PROFILE=main
    ONEX_RUNTIME_NODE_SELECTION=projection_claude_hook_events
    KAFKA_INSTANCE_ID=runtime-main

The broker gate read only the profile, so the writer started a second Pattern B
broker. ``rpk group describe`` on
``pattern-b-broker-onex-cmd-omnimarket-pattern-b-dispatch-v1.__i.runtime-main``
``.__t.onex.cmd.omnimarket.pattern-b-dispatch.v1`` read Stable, 2 members, one
partition held by the writer, and ``omninode-runtime`` with ``set()``; its
``consumer_sync`` dimension read CRITICAL
(``stalled=17301s reason=not_stalled_progressing``) and failed the container
healthcheck. These tests fail on the code that admitted the writer on profile
alone and pass once node selection keeps the broker and the local ingress off.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import UUID

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec
from tests.helpers.runtime_helpers import make_runtime_config

pytestmark = pytest.mark.unit

_COMMAND_TOPIC = "onex.cmd.omnimarket.pattern-b-dispatch.v1"
_BOOTSTRAP = "redpanda:9092"
_WRITER_SELECTION = "projection_claude_hook_events"


class _RecordingBroker:
    """Broker stand-in that records every provision and subscribe."""

    def __init__(self) -> None:
        self.provisioned: list[str] = []
        self.subscribed: list[tuple[str, str | None]] = []

    async def ensure_topic_exists(
        self,
        topic_name: str,
        config: object | None = None,
        correlation_id: UUID | None = None,
        *,
        spec: ModelTopicSpec | None = None,
    ) -> bool:
        self.provisioned.append(topic_name)
        return True

    async def subscribe(
        self,
        topic: str,
        node_identity: object | None = None,
        on_message: Callable[[object], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        **kwargs: object,
    ) -> Callable[[], Awaitable[None]]:
        self.subscribed.append((topic, group_id))

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe


def _build_process(
    monkeypatch: pytest.MonkeyPatch,
    broker: _RecordingBroker,
    *,
    node_selection: str | None,
) -> RuntimeHostProcess:
    """A main-profile host process with the dev lane's broker and ingress config."""
    monkeypatch.setenv("RUNTIME_PROFILE", "main")
    monkeypatch.setenv("KAFKA_INSTANCE_ID", "runtime-main")
    monkeypatch.delenv("ONEX_RUNTIME_NODE_EXCLUSION", raising=False)
    if node_selection is None:
        monkeypatch.delenv("ONEX_RUNTIME_NODE_SELECTION", raising=False)
    else:
        monkeypatch.setenv("ONEX_RUNTIME_NODE_SELECTION", node_selection)
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_host_process."
        "discover_runtime_local_ingress_routes",
        lambda _packages: {},
    )
    monkeypatch.setattr(
        "omnibase_infra.event_bus.service_topic_manager.TopicProvisioner",
        lambda **_kwargs: broker,
    )
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=_BOOTSTRAP,
            environment="local",
        )
    )
    monkeypatch.setattr(bus, "subscribe", broker.subscribe)
    # The values contracts/runtime/runtime_config.yaml declares for both services.
    return RuntimeHostProcess(
        event_bus=bus,
        config=make_runtime_config(
            pattern_b_broker={
                "enabled": True,
                "command_topic": _COMMAND_TOPIC,
                "package_names": ("omnibase_infra", "omnimarket"),
                "enabled_profiles": ("main",),
            },
            local_ingress={
                "enabled": True,
                "socket_path": "/run/onex-runtime/onex-runtime.sock",
                "package_names": ("omnibase_infra", "omnimarket"),
                "enabled_profiles": ("main",),
            },
        ),
    )


@pytest.mark.asyncio
async def test_node_selected_kernel_does_not_join_the_broker_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hook-events writer's environment starts no broker and subscribes nothing."""
    broker = _RecordingBroker()
    process = _build_process(monkeypatch, broker, node_selection=_WRITER_SELECTION)

    assert process._is_pattern_b_broker_effectively_enabled() is False
    assert process._is_local_ingress_effectively_enabled() is False

    await process._start_pattern_b_broker()

    assert process._pattern_b_broker is None
    assert broker.subscribed == [], (
        "a node-selected kernel must not join the pattern-b-broker group: it "
        "shares main's KAFKA_INSTANCE_ID, so it would be a second member of "
        "main's group on a one-partition topic"
    )
    assert broker.provisioned == []
    assert process._build_local_ingress_health().enabled is False


@pytest.mark.asyncio
async def test_unselected_main_runtime_still_starts_the_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The main runtime keeps its broker and its group name is unchanged."""
    broker = _RecordingBroker()
    process = _build_process(monkeypatch, broker, node_selection=None)

    assert process._is_pattern_b_broker_effectively_enabled() is True
    assert process._is_local_ingress_effectively_enabled() is True

    await process._start_pattern_b_broker()

    assert process._pattern_b_broker is not None
    assert process._pattern_b_broker.is_running is True
    assert broker.subscribed == [
        (_COMMAND_TOPIC, "pattern-b-broker-onex-cmd-omnimarket-pattern-b-dispatch-v1")
    ]


@pytest.mark.asyncio
async def test_blank_node_selection_is_no_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank selection is the main runtime, as ModelRuntimeNodeSelection reads it."""
    broker = _RecordingBroker()
    process = _build_process(monkeypatch, broker, node_selection=" , ")

    assert process._is_pattern_b_broker_effectively_enabled() is True
    assert process._is_local_ingress_effectively_enabled() is True


@pytest.mark.asyncio
async def test_node_exclusion_alone_keeps_the_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main runs with ONEX_RUNTIME_NODE_EXCLUSION set on the dev lane and keeps its broker."""
    broker = _RecordingBroker()
    process = _build_process(monkeypatch, broker, node_selection=None)
    monkeypatch.setenv("ONEX_RUNTIME_NODE_EXCLUSION", _WRITER_SELECTION)

    assert process._is_pattern_b_broker_effectively_enabled() is True
    assert process._is_local_ingress_effectively_enabled() is True
