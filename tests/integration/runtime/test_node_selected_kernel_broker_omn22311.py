# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The shipped runtime config under the hook-events writer's environment (OMN-22311).

Loads ``contracts/runtime/runtime_config.yaml`` through the kernel's own loader,
hands it to ``RuntimeHostProcess`` as the kernel does (``model_dump(mode="json")``
plus the selection-scoped service name), and starts the Pattern B broker against
a recording transport. The environment is the one recorded on the dev lane's
``omnimarket-projection-claude-hook-events-writer`` container on 2026-11-03:
``RUNTIME_PROFILE=main``, ``KAFKA_INSTANCE_ID=runtime-main``,
``ONEX_RUNTIME_NODE_SELECTION=projection_claude_hook_events``. Under it the
shipped config must start no broker and no local ingress; the unselected main
runtime, on the same config, must start the broker under the unchanged group.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.models.runtime import ModelRuntimeNodeSelection
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.service_kernel import load_runtime_config
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SELECTION = "projection_claude_hook_events"


class _RecordingTransport:
    """Records the broker's provision and subscribe calls instead of reaching Kafka."""

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


def _kernel_process(
    monkeypatch: pytest.MonkeyPatch,
    transport: _RecordingTransport,
    *,
    selection: str | None,
) -> RuntimeHostProcess:
    """Build the host process from the shipped config the way the kernel does."""
    monkeypatch.setenv("RUNTIME_PROFILE", "main")
    monkeypatch.setenv("KAFKA_INSTANCE_ID", "runtime-main")
    monkeypatch.delenv("ONEX_RUNTIME_NODE_EXCLUSION", raising=False)
    if selection is None:
        monkeypatch.delenv("ONEX_RUNTIME_NODE_SELECTION", raising=False)
    else:
        monkeypatch.setenv("ONEX_RUNTIME_NODE_SELECTION", selection)
    # Route discovery imports whole packages; the route table is not under test.
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_host_process."
        "discover_runtime_local_ingress_routes",
        lambda _packages: {},
    )
    monkeypatch.setattr(
        "omnibase_infra.event_bus.service_topic_manager.TopicProvisioner",
        lambda **_kwargs: transport,
    )
    config = load_runtime_config(_REPO_ROOT / "contracts")
    runtime_config = cast("dict[str, object]", config.model_dump(mode="json"))
    runtime_config["service_name"] = (
        ModelRuntimeNodeSelection.from_env().consumer_service(config.consumer_group)
    )
    runtime_config["node_name"] = config.name
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers="redpanda:9092", environment="local"
        )
    )
    monkeypatch.setattr(bus, "subscribe", transport.subscribe)
    return RuntimeHostProcess(event_bus=bus, config=runtime_config)


@pytest.mark.asyncio
async def test_shipped_config_starts_no_broker_in_the_hook_events_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _RecordingTransport()
    process = _kernel_process(monkeypatch, transport, selection=_SELECTION)

    await process._start_local_ingress()
    await process._start_pattern_b_broker()

    assert process._pattern_b_broker is None
    assert transport.subscribed == []
    assert transport.provisioned == []
    assert process._build_local_ingress_health().enabled is False


@pytest.mark.asyncio
async def test_shipped_config_starts_the_broker_in_the_main_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _RecordingTransport()
    process = _kernel_process(monkeypatch, transport, selection=None)

    await process._start_pattern_b_broker()

    assert process._pattern_b_broker is not None
    assert transport.subscribed == [
        (
            "onex.cmd.omnimarket.pattern-b-dispatch.v1",
            "pattern-b-broker-onex-cmd-omnimarket-pattern-b-dispatch-v1",
        )
    ]
