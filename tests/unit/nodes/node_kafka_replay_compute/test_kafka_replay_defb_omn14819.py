# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""def-B canonicalization proof for node_kafka_replay_compute (OMN-14819).

Tests-as-proof for the hand-flip (verbatim boundary-move) that relocates the
``ModelEventEnvelope`` deserialization boundary out of the handler core so the node
classifies canonical under the canonical handler-shape ratchet (OMN-14355):

* ``test_handler_core_is_ccore_clean`` -- the C-core guard. RED on the pre-flip
  handler module (which imports ``ModelEventEnvelope``), GREEN on the flip.
* ``test_handle_is_defb_shape`` -- the def-B entrypoint takes a single typed request.
* ``test_defb_parity`` -- behavior equivalence: every corpus case replays to the
  same event counts, offsets, correlation chain, and failed-offset evidence.
* ``test_default_deserializer_yields_correlation_id`` -- the relocated default
  deserializer still surfaces ``correlation_id`` through ``ProtocolReplayEnvelope``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_kafka_replay_compute.deserializers.deserializer_default import (
    default_envelope_deserializer,
)
from omnibase_infra.nodes.node_kafka_replay_compute.handlers import handler_replay
from omnibase_infra.nodes.node_kafka_replay_compute.handlers.handler_replay import (
    HandlerKafkaReplay,
)
from tests.unit.nodes.node_kafka_replay_compute._defb_corpus import (
    FakeReplayConsumer,
    ReplayCase,
    replay_cases,
    run_case,
)


@pytest.mark.unit
def test_handler_core_is_ccore_clean() -> None:
    """The canonical handler core must not reference the envelope type (C-core).

    This is the RED->GREEN flip proof: on the pre-flip module this string is present
    (``envelope_in_core``); the flip relocates the boundary so it is absent.
    """
    source = Path(handler_replay.__file__).read_text(encoding="utf-8")
    assert "ModelEventEnvelope" not in source, (
        "handler core still references ModelEventEnvelope -> fails C-core "
        "(canonical handler-shape ratchet, OMN-14355)"
    )


@pytest.mark.unit
def test_handle_is_defb_shape() -> None:
    """def-B: ``handle`` takes exactly one non-self typed request parameter."""
    params = [
        name
        for name in inspect.signature(HandlerKafkaReplay.handle).parameters
        if name != "self"
    ]
    assert params == ["command"], params


@pytest.mark.unit
@pytest.mark.parametrize("case", replay_cases(), ids=lambda c: c.name)
def test_defb_parity(case: ReplayCase) -> None:
    """Behavior equivalence for the def-B handler over the full parity corpus."""
    if case.raises:
        with pytest.raises(ValueError, match="event count mismatch"):
            run_case(case)
        return

    result = run_case(case)
    assert result.events_replayed == case.expected_events
    assert result.last_offset_per_topic == case.expected_last_offset
    assert len(result.correlation_id_chain) == case.expected_correlation_len
    assert result.failed_event_offsets == case.expected_failed_offsets


@pytest.mark.unit
def test_default_deserializer_yields_correlation_id() -> None:
    """The relocated default deserializer surfaces correlation_id via the protocol."""
    correlation_id = uuid4()
    payload = ModelEventEnvelope[dict[str, object]](
        payload={"ordinal": 0},
        correlation_id=correlation_id,
        event_type="dispatch_worker-completed.v1",
    ).model_dump_json()

    envelope = default_envelope_deserializer(payload.encode("utf-8"))

    assert envelope.correlation_id == correlation_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_contract_declared_default_constructor_wiring_replays_isolated_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real contract handler wiring plus default constructor path stays executable."""
    case = replay_cases()[0]
    contract_path = Path(handler_replay.__file__).parents[1] / "contract.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    handler_ref = contract["handler_routing"]["handlers"][0]["handler"]

    assert handler_ref == {
        "name": "HandlerKafkaReplay",
        "module": (
            "omnibase_infra.nodes.node_kafka_replay_compute.handlers.handler_replay"
        ),
    }

    constructed: list[dict[str, object]] = []

    class RecordingProductionConsumer(FakeReplayConsumer):
        def __init__(self, *args: object, **kwargs: object) -> None:
            constructed.append({"args": args, "kwargs": kwargs})
            super().__init__(case.records)

    monkeypatch.setattr(handler_replay, "AIOKafkaConsumer", RecordingProductionConsumer)
    monkeypatch.setattr(handler_replay, "build_aiokafka_auth_kwargs_from_env", dict)

    handler = HandlerKafkaReplay()
    result = await handler.handle(case.command)

    assert result.events_replayed == case.expected_events
    assert constructed == [
        {
            "args": (),
            "kwargs": {
                "bootstrap_servers": case.command.target_cluster_bootstrap,
                "group_id": case.command.target_consumer_group,
                "enable_auto_commit": False,
                "auto_offset_reset": "earliest",
            },
        }
    ]
