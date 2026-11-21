# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev runtime logged "no delivery context" for every projection dispatch (OMN-26404).

Recorded incident. The dev ``omninode-runtime`` container started
2026-11-21T20:24:04Z logged ``Projection writer dispatched with no delivery
context`` on every consumed message. The container's own log tail is kept by
the boot guard as ``incident-a7474c42-...-omninode-runtime.log`` on the dev host
and is the fixture here: ``tests/fixtures/omn26404/
dev_runtime_a7474c42_no_delivery_context.log`` holds its traceback and every
such line in the kept tail (38 lines, 12 handler/topic pairs).

Cause, read off the recorded traceback. The Kafka consume loop built the
delivery coordinates and handed them to ``dispatch_scoped``. The engine decides
at registration whether a dispatcher takes ``delivery`` by inspecting its
signature (``MessageDispatchEngine._dispatcher_accepts_delivery``). Since
OMN-26250 every attached contract dispatcher is wrapped by
``measure_contract_dispatch`` (``contract_dispatch_cost.py``), and the wrapper
declared only ``envelope``. The engine inspected the wrapper, found no
``delivery`` parameter, and called the wrapper without it. The wrapper called
the projection callback with ``delivery=None``, so the callback logged the error
and injected no ``_partition`` or ``_offset`` (OMN-18905: the serving cache then
drops every delta after the first).

These tests drive the real projection callback through the real registration
function and the real engine, once per handler and topic pair the incident
recorded.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.dispatch.model_message_delivery_context import (
    ModelMessageDeliveryContext,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus, EnumMessageCategory
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    PROJECTION_INPROCESS_DISPATCH_ATTR,
    PreparedWiring,
    _commit_handler_wiring,
    _make_projection_dispatch_callback,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.helpers.application_db_topology import projection_database_target

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "omn26404"
    / "dev_runtime_a7474c42_no_delivery_context.log"
)
_RECORDED_LINE = re.compile(
    r"Projection writer dispatched with no delivery context: "
    r"handler=(?P<handler>\w+) topic=(?P<topic>\S+?)\. Its snapshot deltas"
)
_PARTITION = 0
_OFFSET = 7346333
_DSN = "postgresql://omninode_runtime:***REDACTED***@host:5432/omninode_internal"


def _recorded_pairs() -> list[tuple[str, str]]:
    pairs = {
        (m["handler"], m["topic"])
        for m in map(_RECORDED_LINE.search, _FIXTURE.read_text().splitlines())
        if m
    }
    return sorted(pairs)


class _RecordingWriter:
    """An in-process projection writer that records the payload it was handed."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        self.payloads.append(dict(input_data))
        return {"rows_upserted": 1}


def test_the_fixture_is_the_recorded_incident() -> None:
    lines = _FIXTURE.read_text().splitlines()
    recorded = [line for line in lines if _RECORDED_LINE.search(line)]

    assert len(recorded) == 38
    assert len(_recorded_pairs()) == 12
    assert any("contract_dispatch_cost.py" in line for line in lines)
    assert any("return await dispatcher(envelope)" in line for line in lines)


@pytest.mark.parametrize(("handler_name", "topic"), _recorded_pairs())
async def test_no_delivery_context_incident_writer_receives_its_coordinates(
    handler_name: str,
    topic: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OMNINODE_INTERNAL_DB_URL", _DSN)
    writer = type(
        handler_name,
        (_RecordingWriter,),
        {PROJECTION_INPROCESS_DISPATCH_ATTR: True},
    )()
    callback = _make_projection_dispatch_callback(
        writer,
        projection_database_target(
            "consumer_flow_windows",
            "topic_produce_windows",
            schema="omninode_internal",
        ),
        (topic,),
        contract_name=f"node_{handler_name}",
    )
    dispatcher_id = f"dispatcher.{handler_name}"
    # The recorded topics span two kinds: events, and the runtime tick intent.
    category = (
        EnumMessageCategory.INTENT
        if topic.startswith("onex.intent.")
        else EnumMessageCategory.EVENT
    )
    engine = MessageDispatchEngine()
    # The registration function the runtime uses for every attached contract.
    _commit_handler_wiring(
        PreparedWiring(
            dispatcher_id=dispatcher_id,
            dispatcher=callback,
            category=category,
            message_types=None,
            routes=[
                ModelDispatchRoute(
                    route_id=f"route.{handler_name}",
                    topic_pattern=topic,
                    message_category=category,
                    handler_id=dispatcher_id,
                )
            ],
        ),
        engine,
        owner_contract_name=f"node_{handler_name}",
    )
    engine.freeze()

    with caplog.at_level(logging.ERROR):
        result = await engine.dispatch_scoped(
            topic,
            ModelEventEnvelope(payload={"fact": "value"}),
            allowed_dispatcher_ids={dispatcher_id},
            delivery=ModelMessageDeliveryContext(
                topic=topic, partition=_PARTITION, offset=_OFFSET
            ),
        )

    assert result.status is EnumDispatchStatus.SUCCESS, result.error_message
    assert "no delivery context" not in caplog.text
    assert len(writer.payloads) == 1
    assert writer.payloads[0]["_partition"] == _PARTITION
    assert writer.payloads[0]["_offset"] == _OFFSET
