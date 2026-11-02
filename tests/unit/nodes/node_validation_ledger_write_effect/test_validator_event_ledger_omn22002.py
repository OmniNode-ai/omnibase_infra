# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The validation ledger consumes the validator run and triggered events (OMN-22002).

omnibase_core publishes one run event per report and one triggered event per
refusing validator. The ledger projection node subscribes to both through its
contract and projects each event into a ``validation_ledger.append`` intent that
the existing append handler persists, so the ledger table is fed one way.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumResponseStatus
from omnibase_infra.enums.generated.enum_platform_topic import EnumPlatformTopic
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.handlers.handler_db import HandlerDb
from omnibase_infra.handlers.handler_validator_event_ledger_project import (
    HandlerValidatorEventLedgerProject,
)
from omnibase_infra.handlers.models import ModelDbQueryPayload, ModelDbQueryResponse
from omnibase_infra.nodes.node_validation_ledger_write_effect.handlers.handler_validation_ledger_append import (
    HandlerValidationLedgerAppend,
)

pytestmark = pytest.mark.unit

TRIGGERED_TOPIC = EnumPlatformTopic.EVT_VALIDATOR_TRIGGERED_V1.value
RUN_TOPIC = EnumPlatformTopic.EVT_VALIDATOR_RUN_V1.value
_SRC = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra"
_CONTRACT = (
    _SRC / "nodes" / "node_validation_ledger_projection_compute" / "contract.yaml"
)
_HANDLER_SOURCE = _SRC / "handlers" / "handler_validator_event_ledger_project.py"
_HANDLER_MODULE = "omnibase_infra.handlers.handler_validator_event_ledger_project"
_STAMP = datetime(2026, 11, 2, 12, 0, 0, tzinfo=UTC)
_RUN_ID = UUID("5b7a1e0c-4c0e-4d6c-9f55-3c7d8f0a1b21")


def _message(
    topic: str, body: dict[str, Any] | bytes, *, offset: int, partition: int = 0
) -> ModelEventMessage:
    value = body if isinstance(body, bytes) else json.dumps(body).encode()
    return ModelEventMessage(
        topic=topic,
        value=value,
        headers=ModelEventHeaders(
            source="omnibase_core",
            event_type=topic,
            correlation_id=_RUN_ID,
            message_id=uuid4(),
            timestamp=_STAMP,
        ),
        offset=str(offset),
        partition=partition,
    )


def _triggered(validator: str = "ci_inheritance_check") -> dict[str, Any]:
    return {
        "validator_name": validator,
        "severity": "FAIL",
        "finding_count": 2,
        "locations": ["a.py:1", "b.py:9"],
        "repository": "omnibase_infra",
        "pull_request": 5580,
        "head_sha": "f30b9f461344ba16dbec05ab21e29b40f11dbfc2",
        "correlation_id": str(_RUN_ID),
    }


def _run_event() -> dict[str, Any]:
    return {
        "kind": "run",
        "validator_name": "ci_inheritance_check",
        "report_ref": str(_RUN_ID),
        "emitted_at": "2026-11-02T11:59:00+00:00",
        "verdict": "FAIL",
        "fired_count": 2,
    }


class _ConflictModellingDb:
    """Models ``ON CONFLICT (topic, partition, offset) DO NOTHING RETURNING id``."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, int, int], list[object]] = {}

    async def execute(self, envelope: dict[str, Any]) -> MagicMock:
        params = envelope["payload"]["parameters"]
        key = (params[5], params[6], params[7])
        fresh = key not in self.rows
        if fresh:
            self.rows[key] = list(params)
        rows = [{"id": str(uuid4())}] if fresh else []
        wrapper = MagicMock()
        wrapper.result = ModelDbQueryResponse(
            status=EnumResponseStatus.SUCCESS,
            payload=ModelDbQueryPayload(rows=rows, row_count=len(rows)),
            correlation_id=uuid4(),
        )
        return wrapper


def _append_handler(db: _ConflictModellingDb) -> HandlerValidationLedgerAppend:
    handler = HandlerValidationLedgerAppend(MagicMock(spec=ModelONEXContainer))
    handler._db_handler = cast("HandlerDb", db)
    handler._initialized = True
    return handler


async def _persist(
    messages: list[ModelEventMessage], db: _ConflictModellingDb
) -> list[bool]:
    projector = HandlerValidatorEventLedgerProject()
    appender = _append_handler(db)
    duplicates = []
    for message in messages:
        output = await projector.handle(message)
        intent = output.result
        assert intent is not None
        result = await appender.append(intent.payload)
        duplicates.append(result.duplicate)
    return duplicates


def test_contract_subscribes_to_both_topics_and_routes_each_to_the_handler() -> None:
    contract = yaml.safe_load(_CONTRACT.read_text())
    subscribed = set(contract["event_bus"]["subscribe_topics"])
    assert {TRIGGERED_TOPIC, RUN_TOPIC} <= subscribed
    routed = {
        entry["topic"]
        for entry in contract["handler_routing"]["handlers"]
        if entry["handler"]["module"] == _HANDLER_MODULE
    }
    assert routed == {TRIGGERED_TOPIC, RUN_TOPIC}
    assert contract["descriptor"]["node_archetype"] == "compute"


def test_handler_source_carries_no_topic_literal() -> None:
    source = _HANDLER_SOURCE.read_text()
    assert re.findall(r"""['"]onex\.[^'"]*['"]""", source) == []


@pytest.mark.asyncio
async def test_triggered_event_projects_one_row_with_validator_repository_commit_verdict() -> (
    None
):
    db = _ConflictModellingDb()
    event = _triggered()
    await _persist([_message(TRIGGERED_TOPIC, event, offset=7)], db)

    assert len(db.rows) == 1
    (row,) = db.rows.values()
    run_id, repo_id, event_type, event_version, occurred_at = row[:5]
    assert (run_id, repo_id, event_type, event_version) == (
        _RUN_ID,
        "omnibase_infra",
        TRIGGERED_TOPIC,
        "v1",
    )
    assert occurred_at == _STAMP
    envelope = row[8]
    assert isinstance(envelope, bytes)
    assert row[9] == hashlib.sha256(envelope).hexdigest()
    stored = json.loads(envelope)
    assert stored["validator_name"] == "ci_inheritance_check"
    assert stored["repository"] == "omnibase_infra"
    assert stored["head_sha"] == event["head_sha"]
    assert stored["severity"] == "FAIL"


@pytest.mark.asyncio
async def test_triggered_replay_is_idempotent() -> None:
    db = _ConflictModellingDb()
    message = _message(TRIGGERED_TOPIC, _triggered(), offset=7)
    duplicates = await _persist([message, message], db)
    assert duplicates == [False, True]
    assert len(db.rows) == 1


@pytest.mark.asyncio
async def test_run_event_projects_one_row_keyed_by_its_report() -> None:
    db = _ConflictModellingDb()
    await _persist([_message(RUN_TOPIC, _run_event(), offset=3)], db)
    (row,) = db.rows.values()
    assert row[0] == _RUN_ID
    assert row[2] == RUN_TOPIC
    assert row[4] == datetime(2026, 11, 2, 11, 59, tzinfo=UTC)
    assert json.loads(row[8])["verdict"] == "FAIL"


@pytest.mark.asyncio
async def test_same_events_yield_same_rows_on_two_independent_runs() -> None:
    """The row is a pure function of the message, so the bus overlay cannot change it."""
    messages = [
        _message(RUN_TOPIC, _run_event(), offset=0),
        _message(TRIGGERED_TOPIC, _triggered("a_validator"), offset=0, partition=1),
        _message(TRIGGERED_TOPIC, _triggered("b_validator"), offset=1, partition=1),
    ]
    first, second = _ConflictModellingDb(), _ConflictModellingDb()
    await _persist(messages, first)
    await _persist(messages, second)
    assert first.rows == second.rows
    assert len(first.rows) == 3


@pytest.mark.asyncio
async def test_unparseable_event_is_kept_with_fallback_fields() -> None:
    db = _ConflictModellingDb()
    raw = b"not json at all"
    await _persist([_message(TRIGGERED_TOPIC, raw, offset=9)], db)
    (row,) = db.rows.values()
    assert row[1] == "unknown"
    assert row[8] == raw
    assert base64.b64decode(base64.b64encode(raw)) == raw


@pytest.mark.asyncio
async def test_payload_is_read_by_its_topic_model_only() -> None:
    """A run payload on the triggered topic is kept as fallback, never projected as a run."""
    db = _ConflictModellingDb()
    await _persist(
        [
            _message(TRIGGERED_TOPIC, _run_event(), offset=1),
            _message(RUN_TOPIC, _triggered(), offset=2),
        ],
        db,
    )
    rows = sorted(db.rows.values(), key=lambda row: row[7])
    assert [row[1] for row in rows] == ["unknown", "unknown"]
    # Kept under the message's correlation id, not the report_ref or the payload's.
    assert [row[0] for row in rows] == [_RUN_ID, _RUN_ID]
    assert all(row[2] in {TRIGGERED_TOPIC, RUN_TOPIC} for row in rows)


@pytest.mark.asyncio
async def test_run_event_with_naive_emitted_at_is_read_as_utc() -> None:
    db = _ConflictModellingDb()
    body = {**_run_event(), "emitted_at": "2026-11-02T11:59:00"}
    await _persist([_message(RUN_TOPIC, body, offset=4)], db)
    (row,) = db.rows.values()
    assert row[4] == datetime(2026, 11, 2, 11, 59, tzinfo=UTC)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("offset", "partition"),
    [(None, 0), ("5", None), ("not-a-number", 0)],
)
async def test_message_without_a_broker_position_is_refused(
    offset: str | None, partition: int | None
) -> None:
    message = _message(TRIGGERED_TOPIC, _triggered(), offset=0).model_copy(
        update={"offset": offset, "partition": partition}
    )
    with pytest.raises(ProtocolConfigurationError):
        await HandlerValidatorEventLedgerProject().handle(message)


@pytest.mark.asyncio
async def test_fallback_row_uses_message_correlation_id_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    db = _ConflictModellingDb()
    with caplog.at_level("WARNING", logger=_HANDLER_MODULE):
        await _persist([_message(TRIGGERED_TOPIC, b"not json", offset=2)], db)
    (row,) = db.rows.values()
    assert row[0] == _RUN_ID
    assert "kept under fallback identity" in caplog.text
    assert "not json" not in caplog.text


@pytest.mark.asyncio
async def test_naive_header_timestamp_is_read_as_utc_on_every_path() -> None:
    db = _ConflictModellingDb()
    naive = _STAMP.replace(tzinfo=None)
    messages = []
    for index, (topic, body) in enumerate(
        [(TRIGGERED_TOPIC, _triggered()), (TRIGGERED_TOPIC, b"not json")]
    ):
        message = _message(topic, body, offset=index)
        headers = message.headers.model_copy(update={"timestamp": naive})
        messages.append(message.model_copy(update={"headers": headers}))
    await _persist(messages, db)
    assert {row[4] for row in db.rows.values()} == {_STAMP}
