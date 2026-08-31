# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test for terminal event emission from projection callbacks (OMN-11187).

Verifies that _make_projection_dispatch_callback emits a terminal event envelope
to the declared terminal_event topic after each successful DB projection. This is
the integration-layer proof that the runtime wiring correctly bridges the projection
handler's DB write to the event bus observable by Pattern-B consumers.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from omnibase_infra.errors import ProjectionNotMaterializedError
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDispatchSinks,
    _make_projection_dispatch_callback,
)
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)


@pytest.fixture(autouse=True)
def _configured_projection_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_projection_dsns(monkeypatch)


_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter"
)
_PATCH_ENVIRON_GET = "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get"

TERMINAL_TOPIC = "onex.evt.omnimarket.projection-delegation-applied.v1"
DB_TARGET = projection_database_target("delegation_events")
SUBSCRIBE_TOPICS = ("onex.evt.omniclaude.task-delegated.v1",)


@pytest.mark.integration
def test_terminal_event_emitted_after_successful_projection() -> None:
    """After a successful DB projection, a terminal envelope is published to event_bus.

    This is the integration-level proof for OMN-11187: the projection callback
    wires handler.handle() → event_bus.publish(terminal_event, ...) when both
    event_bus and terminal_event are configured at the call site.
    """
    published: list[tuple[str, object, bytes]] = []
    correlation_id = uuid.uuid4()

    class FakeDelegationHandler:
        def handle(self, input_data: dict) -> dict:
            return {"rows_upserted": 1}

    class FakeEventBus:
        async def publish(self, topic: str, key: object, value: bytes) -> None:
            published.append((topic, key, value))

    callback = _make_projection_dispatch_callback(
        FakeDelegationHandler(),
        DB_TARGET,
        SUBSCRIBE_TOPICS,
        sinks=ProjectionDispatchSinks(
            event_bus=FakeEventBus(),
            terminal_event=TERMINAL_TOPIC,
        ),
    )

    envelope = MagicMock()
    envelope.topic = SUBSCRIBE_TOPICS[0]
    envelope.payload = {"task_type": "code-review", "delegated_to": "smoke-agent"}
    envelope.correlation_id = correlation_id

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://postgres:test@localhost:5436/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
            asyncio.run(callback(envelope))

    assert len(published) == 1, "Exactly one terminal event must be published"
    topic, _, raw = published[0]
    assert topic == TERMINAL_TOPIC

    parsed = json.loads(raw.decode("utf-8"))
    assert parsed["event_type"] == TERMINAL_TOPIC
    assert parsed["correlation_id"] == str(correlation_id), (
        "correlation_id must propagate from source envelope to terminal event"
    )
    # OMN-16875: the applied event now carries the handler's own result, not a
    # hardcoded literal. The ``projected`` ack is preserved for existing
    # Pattern-B consumers; ``rows_upserted`` is the fact the handler produced,
    # which previously never reached the bus at all.
    assert parsed["payload"] == {"rows_upserted": 1, "projected": True}


@pytest.mark.integration
def test_write_path_failure_emits_no_terminal_and_withholds_the_offset() -> None:
    """A WRITE-PATH failure raises instead of quarantining (OMN-17379).

    This test previously asserted that ``RuntimeError("DB write failed")``
    produced exactly one quarantine publish and that the callback returned
    normally. Returning normally IS an ACK: the consume boundary reads "no
    exception" as success and the offset advances past an event that was never
    written. Live proof on the .201 dev lane 2026-08-31 — ``pr_merged_events``
    held 28 rows whose newest was 2026-08-03 while its consumer group sat at
    ``Stable / TOTAL-LAG 0 / CURRENT-OFFSET 97 = LOG-END``; every write had been
    failing ``InsufficientPrivilege`` and every offset committed anyway, for 230
    merged PRs.

    A ``RuntimeError`` from the write is the RUNTIME's defect, not the event's:
    the payload is well-formed and still owed a row, and the remedy is an
    operator repair followed by redelivery. So the callback raises
    ``ProjectionNotMaterializedError``, which
    ``EventBusKafka._dispatch_to_subscriber`` classifies offset-unsafe and
    rewinds the partition on.

    No dead-letter copy is asserted **positively**, not merely omitted: on a
    withheld-offset path a DLQ leg publishes one copy per redelivery for as long
    as the write path stays broken, which is how the quarantine sink reached
    8,878,932 messages under OMN-16690. The CONTENT class keeps its quarantine
    leg — that is the sibling row below.
    """
    published: list[tuple] = []

    class FailingHandler:
        def handle(self, input_data: dict) -> dict:
            raise RuntimeError("DB write failed")

    class FakeEventBus:
        async def publish(self, topic: str, key: object, value: bytes) -> None:
            published.append((topic, key, value))

    callback = _make_projection_dispatch_callback(
        FailingHandler(),
        DB_TARGET,
        SUBSCRIBE_TOPICS,
        sinks=ProjectionDispatchSinks(
            event_bus=FakeEventBus(),
            terminal_event=TERMINAL_TOPIC,
        ),
    )

    envelope = MagicMock()
    envelope.topic = SUBSCRIBE_TOPICS[0]
    envelope.payload = {}
    envelope.correlation_id = uuid.uuid4()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://postgres:test@localhost:5436/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
            with pytest.raises(ProjectionNotMaterializedError) as excinfo:
                asyncio.run(callback(envelope))

    assert "DB write failed" in str(excinfo.value), (
        "the raised error must carry the originating failure so an operator can "
        f"tell WHICH write path broke; got {excinfo.value!s}"
    )
    assert published == [], (
        "a write-path failure must publish nothing at all — not a terminal "
        "event (the projection did not write) and not a dead-letter copy (the "
        "record is preserved by the uncommitted offset, and a copy per "
        f"redelivery is the 8.9M-message shape). Published: {published!r}"
    )


@pytest.mark.integration
def test_content_failure_still_quarantines_and_lets_the_offset_advance() -> None:
    """The other half of the OMN-17379 split, kept honest.

    A malformed payload is the EVENT's own defect. Redelivering the identical
    bytes reproduces the identical failure forever, so DLQ-and-advance stays
    correct — otherwise one bad record wedges the partition permanently. This
    row exists so the fix above cannot be widened into "every projection failure
    stalls the feed" without turning red.
    """
    from omnibase_infra.event_bus.topic_constants import build_dlq_topic

    published: list[tuple] = []

    class _Payload(BaseModel):
        task_type: str

    class ValidatingHandler:
        def handle(self, input_data: dict) -> dict:
            _Payload.model_validate(input_data)
            return {"rows_upserted": 1}

    class FakeEventBus:
        async def publish(self, topic: str, key: object, value: bytes) -> None:
            published.append((topic, key, value))

    callback = _make_projection_dispatch_callback(
        ValidatingHandler(),
        DB_TARGET,
        SUBSCRIBE_TOPICS,
        sinks=ProjectionDispatchSinks(
            event_bus=FakeEventBus(),
            terminal_event=TERMINAL_TOPIC,
        ),
    )

    envelope = MagicMock()
    envelope.topic = SUBSCRIBE_TOPICS[0]
    envelope.payload = {}  # missing the required task_type
    envelope.correlation_id = uuid.uuid4()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://postgres:test@localhost:5436/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
            asyncio.run(callback(envelope))

    assert not any(topic == TERMINAL_TOPIC for topic, _key, _value in published), (
        "No terminal event must be published when the handler raises"
    )
    assert len(published) == 1, "Content failure must publish one quarantine DLQ"
    topic, _key, raw = published[0]
    assert topic == build_dlq_topic("quarantine")

    dlq = json.loads(raw.decode("utf-8"))
    assert dlq["failure_class"] == "consumer_error"
    assert dlq["quarantine_fallback"] is True
    assert "ValidationError" in dlq["failure_reason"]


@pytest.mark.integration
def test_no_terminal_event_without_event_bus() -> None:
    """Projection callbacks without an event_bus configured emit no terminal event.

    Existing projection handlers that were wired before OMN-11187 pass event_bus=None
    (the default). This test ensures backward compatibility — those callbacks must
    not crash and must still successfully project to the DB.
    """
    call_count = [0]

    class FakeDelegationHandler:
        def handle(self, input_data: dict) -> dict:
            call_count[0] += 1
            return {"rows_upserted": 1}

    callback = _make_projection_dispatch_callback(
        FakeDelegationHandler(),
        DB_TARGET,
        SUBSCRIBE_TOPICS,
        sinks=ProjectionDispatchSinks(
            event_bus=None,
            terminal_event=TERMINAL_TOPIC,
        ),
    )

    envelope = MagicMock()
    envelope.topic = SUBSCRIBE_TOPICS[0]
    envelope.payload = {"task_type": "code-review"}
    envelope.correlation_id = uuid.uuid4()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://postgres:test@localhost:5436/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
            result = asyncio.run(callback(envelope))

    assert result is None
    assert call_count[0] == 1, "Handler must still be called when bus is None"
