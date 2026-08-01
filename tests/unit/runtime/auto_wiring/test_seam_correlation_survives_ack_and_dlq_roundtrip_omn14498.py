# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14498 Lane C — ACK/DLQ correlation survival seam.

This is a cross-boundary seam test, not two unit suites: one poisoned command
is driven through the REAL production chain

    ``_make_event_bus_callback`` (runtime/auto_wiring/handler_wiring.py)
      -> ``EventBusKafka._publish_raw_to_dlq`` (event_bus/mixin_kafka_dlq.py)
      -> ``ModelDlqMessage.from_kafka_message``
      -> ``DLQProducer.replay_message`` (nodes/node_dlq_replay_effect/engine_dlq_replay.py)

with only the Kafka producer socket stubbed. Every correlation-id hop is real
code; nothing about the lineage is asserted against a surrogate.

RED (exists-but-wrong, NOT missing-file)
----------------------------------------
``callback()`` opened with ``correlation_id: UUID = uuid4()`` and only ever
replaced it from the *decoded body* (``envelope.correlation_id``, or
``data["correlation_id"]`` on the raw-command path). A POISONED message --
one whose value is truncated/undecodable JSON -- raises inside ``json.loads``
before either recovery runs, so the boundary fell through to the outer
``except`` still holding that freshly minted uuid4. The ingress lineage was
sitting on ``message.headers.correlation_id`` the whole time and was never
read.

The consequence is not a missing id: it is a VALID id with the WRONG lineage.
The DLQ record persists the minted id, ``ModelDlqMessage`` parses it
faithfully, ``replay_message`` stamps it faithfully onto the replayed
``correlation_id`` header -- so the replay is correctly-behaving machinery
propagating a fabricated ancestry, and the terminal it eventually produces
is an orphan that joins to nothing upstream. Silent, green, and unjoinable.

Second seam assertion — a NACK never ACKs the offset
----------------------------------------------------
``_route_swallowed_exception`` returned normally even when the DLQ write was
NOT durable (``_publish_raw_to_dlq`` returning ``False``, its documented
non-persistence contract). A callback that returns normally is a successful
callback: ``EventBusKafka._dispatch_to_subscriber`` returns ``True`` and the
offset advances. So the message existed nowhere durable AND was acknowledged
-- the exact loss the OMN-15232 rewind path exists to prevent, made
unreachable by the boundary swallowing its own failure. The fix raises
``BoundaryDlqNotPersistedError`` in that one case so offset advancement is
withheld and Kafka redelivers.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from omnibase_core.models.event_bus.model_event_headers import ModelEventHeaders
from omnibase_core.models.event_bus.model_event_message import ModelEventMessage
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQProducer,
    ModelDlqReplayEngineConfig,
    generate_replay_correlation_id,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _BOUNDARY_DLQ_ENV,
    BoundaryDlqNotPersistedError,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_event_bus_callback as _make_contract_scoped_event_bus_callback,
)


def _make_event_bus_callback(
    topic: str,
    dispatch_engine: object,
    **kwargs: object,
) -> Callable[..., Awaitable[None]]:
    """Build the boundary under its required synthetic contract scope.

    OMN-15474 made the ingress boundary contract-scoped: the callback dispatches
    only to the dispatcher ids its owning contract registered, and refuses to be
    built without them rather than falling back to a process-global fan-out.
    These OMN-14498 seam tests are about correlation lineage across the DLQ
    boundary, not about ownership, so they supply one synthetic dispatcher id —
    mirroring the identical shim in test_boundary_dlq_omn14507.py.
    """
    return _make_contract_scoped_event_bus_callback(
        topic,
        dispatch_engine,  # type: ignore[arg-type]
        allowed_dispatcher_ids={"test-dispatcher"},
        **kwargs,  # type: ignore[arg-type]
    )


_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "seams"
    / "ack_dlq"
    / "poisoned_command.json"
)


def _fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _ingress_message(fx: dict[str, object]) -> ModelEventMessage:
    """The poisoned command exactly as the consume boundary receives it.

    Lineage lives on the headers (where the Kafka transport puts it); the
    value is truncated JSON, so the body is not a recoverable lineage source.
    """
    return ModelEventMessage(
        topic=str(fx["topic"]),
        key=str(fx["ingress_key"]).encode("utf-8"),
        value=str(fx["poisoned_value"]).encode("utf-8"),
        headers=ModelEventHeaders(
            correlation_id=UUID(str(fx["ingress_correlation_id"])),
            timestamp=datetime.now(UTC),
            source="seam-test-ingress",
            event_type=str(fx["topic"]),
        ),
        offset=str(fx["ingress_offset"]),
        partition=int(str(fx["ingress_partition"])),
    )


def _bus_with_stubbed_producer() -> tuple[EventBusKafka, AsyncMock]:
    """A REAL ``EventBusKafka`` (real ``_publish_raw_to_dlq``, real DLQ
    envelope construction) with only the aiokafka producer socket stubbed."""
    bus = EventBusKafka()
    send_and_wait = AsyncMock(return_value=None)
    producer = MagicMock()
    producer.send_and_wait = send_and_wait
    bus._producer = producer
    return bus, send_and_wait


def _raising_dispatch_engine(exc: Exception) -> MagicMock:
    engine = MagicMock()
    engine.dispatch = AsyncMock(side_effect=exc)
    return engine


def _dlq_payload_from_send(send_and_wait: AsyncMock) -> dict[str, object]:
    """Decode the DLQ envelope the real mixin actually put on the wire."""
    assert send_and_wait.await_count >= 1, (
        "the poisoned message never reached a DLQ producer send at all"
    )
    kwargs = send_and_wait.await_args.kwargs
    raw_value = kwargs["value"] if "value" in kwargs else send_and_wait.await_args[0][1]
    decoded = json.loads(
        raw_value.decode("utf-8") if isinstance(raw_value, bytes) else raw_value
    )
    assert isinstance(decoded, dict)
    return decoded


@pytest.mark.asyncio
async def test_seam_correlation_survives_ack_and_dlq_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ingress correlation_id must be byte-identical at every hop of the
    poison -> DLQ -> replay roundtrip, and an unpersisted DLQ write must
    never be ACKed."""
    monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

    fx = _fixture()
    ingress_id = str(fx["ingress_correlation_id"])
    topic = str(fx["topic"])
    message = _ingress_message(fx)

    # Sanity: the fixture really is poisoned (exists-but-wrong, not
    # missing-input). If this ever decodes, the test below would be
    # exercising the already-working envelope path and proving nothing.
    with pytest.raises(json.JSONDecodeError):
        json.loads(str(fx["poisoned_value"]))

    # --- Hop 1: consume boundary -> DLQ envelope -----------------------
    bus, send_and_wait = _bus_with_stubbed_producer()
    callback = _make_event_bus_callback(
        topic,
        _raising_dispatch_engine(RuntimeError("unreachable — decode fails first")),  # type: ignore[arg-type]
        event_bus=bus,
    )

    await callback(message)  # never crashes the consumer loop

    dlq_payload = _dlq_payload_from_send(send_and_wait)
    assert dlq_payload["correlation_id"] == ingress_id, (
        "DLQ record carries a REGENERATED correlation_id: the boundary minted "
        f"{dlq_payload['correlation_id']!r} instead of preserving the ingress "
        f"lineage {ingress_id!r}. A valid id with the wrong ancestry is worse "
        "than a missing one — every downstream join silently resolves to "
        "nothing."
    )
    assert dlq_payload["original_topic"] == topic

    # --- Hop 2: DLQ envelope -> parsed DLQ message ---------------------
    dlq_message = ModelDlqMessage.from_kafka_message(
        payload=dlq_payload, dlq_offset=0, dlq_partition=0
    )
    assert str(dlq_message.correlation_id) == ingress_id

    # --- Hop 3: parsed DLQ message -> replayed message -----------------
    replay_producer = DLQProducer(
        ModelDlqReplayEngineConfig(
            bootstrap_servers="seam-test:9092",
            dlq_topic="onex.dlq.omnibase-infra.commands.v1",
        )
    )
    replay_send = AsyncMock(return_value=None)
    replay_producer._producer = MagicMock(send_and_wait=replay_send)
    replay_producer._started = True

    await replay_producer.replay_message(dlq_message, generate_replay_correlation_id())

    replay_headers = dict(replay_send.await_args.kwargs["headers"])
    assert replay_headers["correlation_id"] == ingress_id.encode("utf-8"), (
        "replayed message carries the wrong lineage — the replay machinery is "
        "faithful, so a fabricated id in the DLQ record produces an ORPHAN "
        "TERMINAL here: "
        f"{replay_headers['correlation_id']!r} != {ingress_id.encode('utf-8')!r}"
    )
    # The replay attempt gets its OWN id; it must not overwrite lineage.
    assert replay_headers["x-replay-correlation-id"] != ingress_id.encode("utf-8")
    assert replay_send.await_args.kwargs["headers"] is not None
    assert replay_send.await_args[0][0] == topic


@pytest.mark.asyncio
async def test_seam_unpersisted_dlq_write_is_never_acked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NACK never ACKs the offset.

    When the DLQ write is NOT durable, the boundary must not return normally
    — a normally-returning callback is an ACK
    (``_dispatch_to_subscriber`` -> ``True`` -> offset advances), which would
    lose a message that exists nowhere durable.
    """
    monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

    fx = _fixture()
    message = _ingress_message(fx)

    bus = MagicMock(spec=EventBusKafka)
    # The documented non-persistence contract: no exception, False return.
    bus._publish_raw_to_dlq = AsyncMock(return_value=False)

    callback = _make_event_bus_callback(
        str(fx["topic"]),
        _raising_dispatch_engine(RuntimeError("boom")),  # type: ignore[arg-type]
        event_bus=bus,
    )

    with pytest.raises(BoundaryDlqNotPersistedError) as excinfo:
        await callback(message)

    # Lineage is legible on the escape hatch too, not just in a log line.
    assert str(fx["ingress_correlation_id"]) in str(excinfo.value)
    bus._publish_raw_to_dlq.assert_awaited_once()
    assert bus._publish_raw_to_dlq.await_args.kwargs["correlation_id"] == UUID(
        str(fx["ingress_correlation_id"])
    )
