# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18691: a MIRROR-ONLY forwarder moves CI-bus records onto the dev lane.

The unit tests beside this one pin that the mirror-only SHAPE validates. This
one pins that the shape actually WORKS: a resolved runtime config carrying a
lane mirror and no trust-boundary legs at all drives the real
``NodeLaneMirror`` over the real ``ModelGatewayLaneMirrorConfig``, and a CI-bus
command topic crosses from the dedicated broker's leg to the dev lane's leg
exactly once.

That distinction is the whole reason this file exists. A config that validates
and a process that delivers are different claims, and OMN-18691's failure class
-- a publish that is green while nothing is moved -- is precisely the gap
between them. Only the two broker transports are faked, because a broker is not
available in this tier; every layer above them is the shipped code path.

The last test is the honest half: the tenant-edge entrypoint REFUSES a
mirror-only config today. Wiring a mirror-only process is phase 3 step 3 and is
a change to the live gateway process, so the refusal is asserted rather than
left to be discovered by whoever deploys it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.idempotency.store_sqlite import StoreIdempotencySqlite
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayForwarderConfig,
    ModelGatewayLaneMirrorConfig,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_forwarder_runtime_config import (
    ModelGatewayForwarderRuntimeConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror import (
    NodeLaneMirror,
)

pytestmark = pytest.mark.integration

# The two legs of the phase-3 topology, by the container names they resolve on
# the lab host. The external addresses are the committed lane map's business
# (omnimarket `config/ci_bus_lanes.yaml`), never this file's.
CI_BUS_SOURCE = "omninode-ci-bus-redpanda:9092"
DEV_LANE_MIRROR = "omnibase-infra-redpanda:9092"

# The command topic whose loss reddens a Receipt Gate. It is the one worth
# driving end to end, rather than a synthetic probe topic.
OCC_AUTOBIND_TOPIC = "onex.cmd.omnimarket.occ-autobind.v1"
CI_BUS_TOPICS = (
    OCC_AUTOBIND_TOPIC,
    "onex.cmd.omnimarket.occ-companion-effect-requested.v1",
    "onex.evt.github.pr-merged.v1",
    "onex.cmd.omnimarket.redeploy-start.v1",
)


@dataclass(frozen=True)
class _Sent:
    topic: str
    value: bytes


class _FakeMirrorProducer:
    """The dev lane's publish surface."""

    def __init__(self) -> None:
        self.sent: list[_Sent] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.sent.append(_Sent(topic=topic, value=value))


class _FakeSourceConsumer:
    """The dedicated broker's pull surface, with explicit offset tracking."""

    def __init__(self) -> None:
        self._pending: list[ModelTransportMessage] = []
        self.committed: list[object] = []
        self.nacked: list[object] = []

    def offer(self, message: ModelTransportMessage) -> None:
        self._pending.append(message)

    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> Sequence[ModelTransportMessage]:
        batch = self._pending[:max_messages]
        self._pending = self._pending[max_messages:]
        return batch

    async def commit(self, message: object) -> None:
        self.committed.append(message)

    async def nack(self, message: object) -> None:
        self.nacked.append(message)


def _autobind_command(*, envelope_id: str, offset: int = 0) -> ModelTransportMessage:
    """One OCC autobind command, in the wire shape the broker produces."""
    stable_id = uuid5(NAMESPACE_URL, f"omn18691/{envelope_id}")
    envelope = {
        "envelope_id": str(stable_id),
        "envelope_timestamp": datetime.now(UTC).isoformat(),
        "correlation_id": str(stable_id),
        "event_type": OCC_AUTOBIND_TOPIC,
        "payload": {"probe": "omn18691"},
    }
    return ModelTransportMessage(
        topic=OCC_AUTOBIND_TOPIC,
        partition=0,
        offset=offset,
        key=None,
        value=json.dumps(envelope).encode("utf-8"),
        headers={
            "content_type": b"application/json",
            "correlation_id": str(stable_id).encode("utf-8"),
            "message_id": str(stable_id).encode("utf-8"),
            "event_type": OCC_AUTOBIND_TOPIC.encode("utf-8"),
            "source": b"publish_occ_autobind_command",
        },
        ack_token=f"{OCC_AUTOBIND_TOPIC}:0:{offset}",
    )


def _lane_mirror() -> ModelGatewayLaneMirrorConfig:
    return ModelGatewayLaneMirrorConfig(
        source_lane="ci-bus",
        mirror_lanes=("dev",),
        topics=CI_BUS_TOPICS,
    )


def _mirror_only_runtime_config(
    dedupe_store_path: Path = Path("/app/data/ci-bus-mirror.sqlite3"),
) -> ModelGatewayForwarderRuntimeConfig:
    """The resolved config a mirror-only deployment would boot from."""
    return ModelGatewayForwarderRuntimeConfig(
        forwarder=ModelGatewayForwarderConfig(
            tenant_identity=ModelGatewayTenantIdentity(
                tenant_id=UUID("79afa726-3852-464f-b7a4-d4b8b9c75ee7"),
                tenant_slug="beta-gateway-canary-79afa7263852",
                principal_id="t-79afa7263852464fb7a4d4b8b9c75ee7",
            ),
            local_transport_flavor="containerized",
            dedupe_store_path=dedupe_store_path,
            lane_mirror=_lane_mirror(),
        ),
        lane_mirror_source_bus=ModelKafkaEventBusConfig(
            bootstrap_servers=CI_BUS_SOURCE,
            environment="ci-bus-mirror-source",
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        ),
        lane_mirror_buses={
            "dev": ModelKafkaEventBusConfig(
                bootstrap_servers=DEV_LANE_MIRROR,
                environment="ci-bus-mirror-destination",
                enable_auto_commit=False,
                auto_offset_reset="earliest",
            )
        },
    )


@pytest.mark.asyncio
async def test_an_occ_autobind_command_crosses_from_the_ci_bus_to_the_dev_lane(
    tmp_path: Path,
) -> None:
    """The record whose loss reddens a Receipt Gate makes the crossing.

    This is the claim option B rests on: the four consumers stay in the dev
    lane and the command reaches them from a broker no lane rebuild touches.
    """
    runtime = _mirror_only_runtime_config(tmp_path / "dedupe.sqlite3")
    source = _FakeSourceConsumer()
    dev = _FakeMirrorProducer()
    lane_mirror = runtime.forwarder.lane_mirror
    assert lane_mirror is not None
    store = StoreIdempotencySqlite(runtime.forwarder.dedupe_store_path)
    await store.start()
    try:
        mirror = NodeLaneMirror(
            config=lane_mirror,
            source_consumer=source,
            mirror_producers={"dev": dev},
            idempotency_store=store,
        )

        source.offer(_autobind_command(envelope_id="pr-3798"))
        await mirror.drain_once()
    finally:
        await store.close()

    assert [sent.topic for sent in dev.sent] == [OCC_AUTOBIND_TOPIC]
    # The source offset is committed only after the mirror acknowledged, which
    # is what makes a dev-lane rebuild lose nothing: the offsets live on the
    # CI-bus broker and an unacknowledged record is simply redelivered.
    assert len(source.committed) == 1
    assert source.nacked == []


@pytest.mark.asyncio
async def test_redelivery_after_a_lane_rebuild_does_not_duplicate_the_command(
    tmp_path: Path,
) -> None:
    """A rebuild mid-flight means at-least-once redelivery, not a double mint.

    An OCC autobind command delivered twice mints two companion pull requests
    for one product pull request, which is a worse outcome than the red publish
    this ticket set out to remove. The idempotency store here is the real
    durable one, because that is the component the claim rests on.
    """
    runtime = _mirror_only_runtime_config(tmp_path / "dedupe.sqlite3")
    source = _FakeSourceConsumer()
    dev = _FakeMirrorProducer()
    lane_mirror = runtime.forwarder.lane_mirror
    assert lane_mirror is not None
    store = StoreIdempotencySqlite(runtime.forwarder.dedupe_store_path)
    await store.start()
    try:
        mirror = NodeLaneMirror(
            config=lane_mirror,
            source_consumer=source,
            mirror_producers={"dev": dev},
            idempotency_store=store,
        )

        source.offer(_autobind_command(envelope_id="pr-3798"))
        await mirror.drain_once()
        source.offer(_autobind_command(envelope_id="pr-3798", offset=1))
        await mirror.drain_once()
    finally:
        await store.close()

    assert len(dev.sent) == 1


@pytest.mark.asyncio
async def test_the_tenant_edge_entrypoint_refuses_a_mirror_only_config() -> None:
    """Refused loudly, because half a tenant edge reports ready and moves nothing.

    Making ``run_gateway_forwarder`` build a mirror-only edge is phase 3 step 3
    and is a change to the live gateway process. Until it lands, a mirror-only
    config handed to this entrypoint must fail at the door rather than start
    with its trust-boundary legs unwired.
    """
    import asyncio

    from omnibase_infra.runtime.gateway_forwarder import run_gateway_forwarder

    async def _never_resolves(_ref: str) -> str | None:  # pragma: no cover - unused
        raise AssertionError("the refusal must precede any secret resolution")

    with pytest.raises(ValueError, match="requires a declared cloud_bus"):
        await run_gateway_forwarder(
            _mirror_only_runtime_config(),
            shutdown_event=asyncio.Event(),
            resolve_secret=_never_resolves,
        )
