# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pattern B broker boot must not die on an unprovisioned command topic (OMN-17857).

Reproduces the staging delivery failure recorded on run 33843160175 of
``deliver-dev-candidate-to-staging.yml`` (head ``9ac332d4``): the boot gate
failed because ``omninode-runtime`` ran 1677s and then exited 1 from

    runtime_host_process.py:2066 _start_pattern_b_broker
      -> service_pattern_b_broker.py:494  (RuntimePatternBBroker.start)
        -> event_bus_kafka.py:2073
           InfraTimeoutError [ONEX_CORE_092_TIMEOUT_ERROR]: Timeout starting
           consumer for topic onex.cmd.omnimarket.pattern-b-dispatch.v1 after
           120s (topic metadata unavailable)

An ephemeral lane brings up an empty redpanda and has no producer for the
Pattern B command topic at boot, so the topic does not exist and the consumer
attach burns its whole metadata budget. The auto-wiring boot interleave already
provisions every contract-declared topic through ``TopicProvisioner``
(OMN-13237), but the Pattern B command topic comes from runtime *config* rather
than a contract's ``event_bus`` subcontract, so that path never sees it.

Related Tickets:
    - OMN-17857: Pattern B broker boot fails closed on a missing command topic.
    - OMN-13237: per-contract provision -> confirm-ready -> attach boot interleave.
    - OMN-11242: the existing in-process TopicProvisioner reuse this fix extends.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraTimeoutError, ModelTimeoutErrorContext
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec
from tests.helpers.runtime_helpers import make_runtime_config

pytestmark = pytest.mark.unit

_COMMAND_TOPIC = "onex.cmd.omnimarket.pattern-b-dispatch.v1"
_BOOTSTRAP = "redpanda.onex-staging.svc.cluster.local:9092"


class _EphemeralLaneBroker:
    """Broker whose topic set starts empty, like a fresh ephemeral-lane redpanda."""

    def __init__(self) -> None:
        self.existing_topics: set[str] = set()
        self.provisioned: list[str] = []
        self.subscribed: list[str] = []

    async def ensure_topic_exists(
        self,
        topic_name: str,
        config: object | None = None,
        correlation_id: UUID | None = None,
        *,
        spec: ModelTopicSpec | None = None,
    ) -> bool:
        self.provisioned.append(topic_name)
        self.existing_topics.add(topic_name)
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
        """Mirror EventBusKafka.subscribe: a missing topic burns the budget."""
        if topic not in self.existing_topics:
            raise InfraTimeoutError(
                f"Timeout starting consumer for topic {topic} after 120s "
                "(topic metadata unavailable)",
                context=ModelTimeoutErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="start_consumer",
                    target_name=f"kafka.{topic}",
                    correlation_id=uuid4(),
                    timeout_seconds=120.0,
                ),
                topic=topic,
                servers=_BOOTSTRAP,
            )
        self.subscribed.append(topic)

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe


def _build_process(
    monkeypatch: pytest.MonkeyPatch, lane: _EphemeralLaneBroker
) -> RuntimeHostProcess:
    """RuntimeHostProcess wired to a Kafka bus fronting the empty lane broker."""
    monkeypatch.setenv("RUNTIME_PROFILE", "main")
    # No contract scan: the Pattern B route table is irrelevant to the boot
    # failure under test, and discovery would import whole packages.
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_host_process."
        "discover_runtime_local_ingress_routes",
        lambda _packages: {},
    )
    monkeypatch.setattr(
        "omnibase_infra.event_bus.service_topic_manager.TopicProvisioner",
        lambda **_kwargs: lane,
    )

    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=_BOOTSTRAP,
            environment="test",
        )
    )
    monkeypatch.setattr(bus, "subscribe", lane.subscribe)

    return RuntimeHostProcess(
        event_bus=bus,
        config=make_runtime_config(
            pattern_b_broker={
                "enabled": True,
                "command_topic": _COMMAND_TOPIC,
                "package_names": ("omnibase_infra",),
                "enabled_profiles": ("main",),
            }
        ),
    )


@pytest.mark.asyncio
async def test_pattern_b_broker_boot_provisions_missing_command_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boot survives an ephemeral lane that has never seen the command topic.

    Before OMN-17857 this raised InfraTimeoutError out of ``start()`` and the
    runtime process exited 1 after burning the whole consumer metadata budget.
    """
    lane = _EphemeralLaneBroker()
    process = _build_process(monkeypatch, lane)

    await process._start_pattern_b_broker()

    assert lane.provisioned == [_COMMAND_TOPIC], (
        "the Pattern B command topic must be provisioned through the existing "
        "TopicProvisioner before the broker attaches its consumer"
    )
    assert lane.subscribed == [_COMMAND_TOPIC]
    broker = process._pattern_b_broker
    assert broker is not None
    assert broker.is_running is True


@pytest.mark.asyncio
async def test_pattern_b_boot_provisioning_is_idempotent_when_topic_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lane that already carries the topic still provisions idempotently."""
    lane = _EphemeralLaneBroker()
    lane.existing_topics.add(_COMMAND_TOPIC)
    process = _build_process(monkeypatch, lane)

    await process._start_pattern_b_broker()

    assert lane.provisioned == [_COMMAND_TOPIC]
    assert lane.subscribed == [_COMMAND_TOPIC]


@pytest.mark.asyncio
async def test_pattern_b_boot_reports_provisioning_failure_without_swallowing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provisioner that cannot create the topic degrades loudly, never silently.

    The attach still fails closed against a real broker problem -- this asserts
    the operator gets a named warning first, not an unexplained 120s timeout.
    """
    lane = _EphemeralLaneBroker()

    async def _refuse(
        topic_name: str,
        config: object | None = None,
        correlation_id: UUID | None = None,
        *,
        spec: ModelTopicSpec | None = None,
    ) -> bool:
        lane.provisioned.append(topic_name)
        return False

    monkeypatch.setattr(lane, "ensure_topic_exists", _refuse)
    process = _build_process(monkeypatch, lane)

    with caplog.at_level("WARNING"), pytest.raises(InfraTimeoutError):
        await process._start_pattern_b_broker()

    assert lane.provisioned == [_COMMAND_TOPIC]
    assert any(
        _COMMAND_TOPIC in record.getMessage() and "OMN-17857" in record.getMessage()
        for record in caplog.records
    ), "provisioning failure must be logged against the command topic, not swallowed"


@pytest.mark.asyncio
async def test_pattern_b_boot_skips_provisioning_for_non_kafka_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-memory bus needs no admin client; provisioning must not be attempted."""
    monkeypatch.setenv("RUNTIME_PROFILE", "main")
    monkeypatch.setattr(
        "omnibase_infra.runtime.runtime_host_process."
        "discover_runtime_local_ingress_routes",
        lambda _packages: {},
    )
    lane = _EphemeralLaneBroker()
    monkeypatch.setattr(
        "omnibase_infra.event_bus.service_topic_manager.TopicProvisioner",
        lambda **_kwargs: lane,
    )

    process = RuntimeHostProcess(
        config=make_runtime_config(
            pattern_b_broker={
                "enabled": True,
                "command_topic": _COMMAND_TOPIC,
                "package_names": ("omnibase_infra",),
                "enabled_profiles": ("main",),
            }
        ),
    )
    monkeypatch.setattr(cast("object", process._event_bus), "subscribe", lane.subscribe)
    lane.existing_topics.add(_COMMAND_TOPIC)

    await process._start_pattern_b_broker()

    assert lane.provisioned == []
    assert lane.subscribed == [_COMMAND_TOPIC]


@pytest.mark.asyncio
async def test_pattern_b_boot_survives_unbuildable_provisioner(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Provisioning must never introduce a boot failure mode of its own.

    ``TopicProvisioner.__init__`` validates its contracts root; a construction
    error has to degrade to the pre-OMN-17857 behaviour (attach, fail closed
    only if the topic really is absent), not kill the runtime earlier than
    before.
    """
    lane = _EphemeralLaneBroker()
    lane.existing_topics.add(_COMMAND_TOPIC)
    process = _build_process(monkeypatch, lane)

    def _explode(**_kwargs: object) -> object:
        raise ValueError("contracts_root does not exist")

    monkeypatch.setattr(
        "omnibase_infra.event_bus.service_topic_manager.TopicProvisioner",
        _explode,
    )

    with caplog.at_level("WARNING"):
        await process._start_pattern_b_broker()

    assert lane.provisioned == []
    assert lane.subscribed == [_COMMAND_TOPIC]
    assert any(
        _COMMAND_TOPIC in record.getMessage() and "OMN-17857" in record.getMessage()
        for record in caplog.records
    )
