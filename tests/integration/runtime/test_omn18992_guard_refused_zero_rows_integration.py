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

The second is the one that must not be softened. An ordering guard refusing a
redelivery is correct behaviour and fires routinely -- 4 times in 33 minutes on
the .201 dev lane while the writer emitting them was serving 500 rows and
updating every few seconds -- and logging it as ERROR trains people to skip the
class, at which point a writer that genuinely wrote nothing goes unseen.
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


def _dispatch(result: dict[str, Any]) -> list[logging.LogRecord]:
    """Run the real callback and return what the wiring logged."""
    callback = _make_projection_dispatch_callback(
        _Writer(result),
        projection_database_target("pr_merged_events", schema="omninode_internal"),
        (_TOPIC,),
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
                asyncio.run(callback(_heartbeat_envelope()))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return records


def _messages(records: list[logging.LogRecord], level: int) -> list[str]:
    return [r.getMessage() for r in records if r.levelno == level]


def test_a_guard_refusal_reaches_the_runtime_as_info() -> None:
    """The case the four live lines are, driven through the real callback."""
    records = _dispatch(
        {"rows_upserted": 0, "flow_rows": [], ROWS_REFUSED_KEY: 1},
    )

    info = _messages(records, logging.INFO)
    errors = _messages(records, logging.ERROR)

    assert any("ordering guard" in message for message in info), (
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

    assert any(
        "wrote zero rows (no terminal emitted)" in m
        for m in _messages(records, logging.ERROR)
    )


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
