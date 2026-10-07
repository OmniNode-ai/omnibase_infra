# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18992 — the real dispatch callback classifies a zero-row return.

The unit suite pins the extractor and the branch shape in isolation. Neither
can fail if the callback the runtime actually builds never reaches the branch,
and that is exactly the class of gap this ticket came out of: the change that
surfaced it passed every test while injecting nothing on the path the
projection writers use.

So this drives the REAL callback from ``_make_projection_dispatch_callback``
over a real envelope, with only the database adapter and the DSN lookup
doubled, and reads the log records the runtime itself emitted. Both directions
are asserted, because only one of them is the defect:

* a handler reporting rows refused by its ordering guard -> INFO, no ERROR;
* a handler reporting a bare zero -> the ERROR line, unchanged.

A healthy writer's zero-row log does not establish why it returned zero.
B23 corrects that historical attribution. The real PostgreSQL fixture in
``tests/integration/migrations/test_omn18992_projection_zero_rows.py`` forces
an older-sequence refusal and separately drives a window-less heartbeat.

"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ROWS_REFUSED_KEY,
    _make_projection_dispatch_callback,
)
from tests.helpers.application_db_topology import (
    configure_projection_dsns,
    projection_database_target,
)

pytestmark = pytest.mark.integration

_WIRING_LOGGER = "omnibase_infra.runtime.auto_wiring.handler_wiring"
_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter"
)
_PATCH_ENVIRON_GET = "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get"
_TEST_DSN = "postgresql://user:pass@host:5432/omnidash_analytics"
_TOPIC = "onex.evt.platform.node-heartbeat.v1"


@pytest.fixture(autouse=True)
def _dsns(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_projection_dsns(monkeypatch, url=_TEST_DSN)


class _Writer:
    """A projection handler returning whatever outcome the case is about."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def handle(self, input_data: dict[str, Any]) -> dict[str, Any]:
        return self._result


def _heartbeat_envelope() -> MagicMock:
    """A node heartbeat, the topic the four live refusals were measured on."""
    envelope = MagicMock()
    envelope.topic = _TOPIC
    envelope.payload = {
        "node_id": "omninode-runtime",
        "topic": _TOPIC,
        "window_start": "2026-09-21T10:18:00Z",
        "window_end": "2026-09-21T10:19:00Z",
        "ingest_sequence": 41,
    }
    envelope.correlation_id = "omn-18992-heartbeat"
    return envelope


def _dispatch(
    result: dict[str, Any],
    *,
    contract_name: str = "",
    payload: dict[str, Any] | None = None,
) -> list[logging.LogRecord]:
    """Run the real callback and return what the wiring logged."""
    callback = _make_projection_dispatch_callback(
        _Writer(result),
        projection_database_target("pr_merged_events", schema="omninode_internal"),
        (_TOPIC,),
        contract_name=contract_name,
    )
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger(_WIRING_LOGGER)
    handler = _Capture(level=logging.INFO)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with patch(_PATCH_ENVIRON_GET, return_value=_TEST_DSN):
            with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
                envelope = _heartbeat_envelope()
                if payload is not None:
                    envelope.payload = payload
                asyncio.run(callback(envelope))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return records


def _messages(records: list[logging.LogRecord], level: int) -> list[str]:
    return [r.getMessage() for r in records if r.levelno == level]


def test_a_guard_refusal_reaches_the_runtime_as_info() -> None:
    """An explicit reported refusal, driven through the real callback."""
    records = _dispatch(
        {"rows_upserted": 0, "flow_rows": [], ROWS_REFUSED_KEY: 1},
    )

    info = _messages(records, logging.INFO)
    errors = _messages(records, logging.ERROR)

    assert any("ORDERING_GUARD_REFUSED" in message for message in info), (
        f"no INFO line naming the guard; INFO records were {info}"
    )
    assert not any("wrote zero rows (no terminal emitted)" in m for m in errors), (
        f"the zero-rows ERROR still fired on a guard refusal; ERRORs were {errors}"
    )


def test_a_bare_zero_still_reaches_the_runtime_as_error() -> None:
    """The half that must not be softened: a writer that wrote nothing."""
    records = _dispatch({"rows_upserted": 0, "flow_rows": []})

    errors = _messages(records, logging.ERROR)

    assert any("wrote zero rows (no terminal emitted)" in m for m in errors), (
        f"the genuine zero-write lost its ERROR; ERRORs were {errors}"
    )
    assert not any("ordering guard" in m for m in _messages(records, logging.INFO))


def test_a_writer_that_never_heard_of_the_field_is_unchanged() -> None:
    """AC2 through the real path, not just through the extractor.

    This is what lets the consumer land in this repo before the producer in
    omnimarket. If it ever fails, the two halves have become order-dependent
    and the landing sequence is unsafe.
    """
    records = _dispatch({"projected": False})

    assert _messages(records, logging.ERROR) == [
        "Projection handler wrote zero rows (no terminal emitted): "
        f"handler=_Writer topic={_TOPIC} event_type=heartbeat "
        "rows_upserted=0 result={'projected': False}"
    ]


def test_a_successful_write_logs_neither_line() -> None:
    """The positive control: the callback is reached and can stay quiet.

    Without this, every assertion above would also pass against a callback
    that never ran at all.
    """
    records = _dispatch({"rows_upserted": 2, "flow_rows": [{"consumer_group": "g"}]})

    assert not any(
        "wrote zero rows" in m or "ordering guard" in m
        for m in _messages(records, logging.ERROR) + _messages(records, logging.INFO)
    )


def test_windowless_consumer_flow_heartbeat_has_its_own_token() -> None:
    records = _dispatch(
        {"rows_upserted": 0, "flow_rows": [], ROWS_REFUSED_KEY: 0},
        contract_name="projection_consumer_flow",
        payload={"node_id": "omninode-runtime"},
    )
    assert any("WINDOW_LESS_HEARTBEAT" in m for m in _messages(records, logging.INFO))
    assert not _messages(records, logging.ERROR)
    assert not any("ORDERING_GUARD_REFUSED" in r.getMessage() for r in records)


@pytest.mark.parametrize(
    ("contract_name", "payload", "result"),
    [
        (
            "projection_consumer_flow",
            {"flow_window": {}},
            {"rows_upserted": 0, "flow_rows": [], ROWS_REFUSED_KEY: 0},
        ),
        (
            "projection_lab_lane_health",
            {},
            {"rows_upserted": 0, "flow_rows": [], ROWS_REFUSED_KEY: 0},
        ),
        ("projection_consumer_flow", {}, {"rows_upserted": 0, "flow_rows": []}),
        (
            "projection_consumer_flow",
            {},
            {"rows_upserted": 0, "flow_rows": [], ROWS_REFUSED_KEY: "invalid"},
        ),
        ("projection_consumer_flow", {}, {"rows_upserted": 0, ROWS_REFUSED_KEY: 0}),
    ],
)
def test_unproven_noop_still_logs_error(
    contract_name: str, payload: dict[str, Any], result: dict[str, Any]
) -> None:
    records = _dispatch(result, contract_name=contract_name, payload=payload)
    assert any(
        "wrote zero rows (no terminal emitted)" in m
        for m in _messages(records, logging.ERROR)
    )
    assert not any("WINDOW_LESS_HEARTBEAT" in r.getMessage() for r in records)


@pytest.mark.parametrize("refused", [False, 0.0, None, "0", -1])
def test_malformed_empty_heartbeat_result_keeps_error(refused: object) -> None:
    records = _dispatch(
        {"rows_upserted": 0, "flow_rows": [], ROWS_REFUSED_KEY: refused},
        contract_name="projection_consumer_flow",
        payload={},
    )
    assert _messages(records, logging.ERROR)
    assert not any("WINDOW_LESS_HEARTBEAT" in r.getMessage() for r in records)
