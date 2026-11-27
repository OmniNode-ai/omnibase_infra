# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-26410 — the real dispatch callback honours a handler's zero-row attestation.

``tests/unit/runtime/auto_wiring/test_projection_noop_attestation_omn26410.py``
pins every attestation shape. This file drives the callback the runtime builds
from ``_make_projection_dispatch_callback`` over a real envelope with only the
database adapter and the DSN lookup doubled, and reads the log records the
wiring itself emitted, so it fails if the classifier is never reached on the
path the projection writers use. Both directions are asserted:

* the EvalResults writer's runtime tick, attested ``tick`` -> INFO, no ERROR;
* the same tick unattested, and a judged event whose upsert returned no row
  -> the zero-rows ERROR, unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
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
_TEST_DSN = "postgresql://user:***REDACTED***@host:5432/omnidash_analytics"
_TICK_TOPIC = "onex.intent.platform.runtime-tick.v1"
_JUDGED_TOPIC = "onex.evt.omnimarket.delegation-acceptance-judged.v1"
_NOOP_KEY = "projection_noop_reason"
_ZERO_ROWS_ERROR = "wrote zero rows (no terminal emitted)"

_TICK_RESULT: dict[str, Any] = {
    "rows_upserted": 0,
    "snapshot": {"published": False, "reason": "interval"},
}
_JUDGED_NO_ROW_RESULT: dict[str, Any] = {
    "rows_upserted": 0,
    "items_written": 0,
    "refused": [],
    "dlq_routed": 0,
}


@pytest.fixture(autouse=True)
def _dsns(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_projection_dsns(monkeypatch, url=_TEST_DSN)


class _Writer:
    """A projection handler returning whatever outcome the case is about."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def handle(self, input_data: dict[str, Any]) -> dict[str, Any]:
        return self._result


def _dispatch(result: dict[str, Any], *, topic: str) -> list[logging.LogRecord]:
    """Run the real callback over one envelope on ``topic`` and return its log."""
    callback = _make_projection_dispatch_callback(
        _Writer(result),
        projection_database_target("pr_merged_events", schema="omninode_internal"),
        (topic,),
        contract_name="projection_eval_results",
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
        envelope = MagicMock()
        envelope.topic = topic
        envelope.payload = {"now": "2026-11-23T00:02:33+00:00"}
        envelope.correlation_id = "omn-26410-integration"
        with patch(_PATCH_ENVIRON_GET, return_value=_TEST_DSN):
            with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
                asyncio.run(callback(envelope))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return records


def _messages(records: list[logging.LogRecord], level: int) -> list[str]:
    return [r.getMessage() for r in records if r.levelno == level]


def test_an_attested_tick_reaches_the_runtime_as_info() -> None:
    records = _dispatch({**_TICK_RESULT, _NOOP_KEY: "tick"}, topic=_TICK_TOPIC)

    errors = _messages(records, logging.ERROR)
    info = _messages(records, logging.INFO)

    assert not any(_ZERO_ROWS_ERROR in m for m in errors), (
        f"an attested tick still logged the zero-rows ERROR: {errors}"
    )
    assert any("NOOP_ATTESTED" in m and "reason=tick" in m for m in info), (
        f"no INFO line naming the attested reason; INFO records were {info}"
    )


def test_an_unattested_tick_zero_still_reaches_the_runtime_as_error() -> None:
    """The half that must not be softened: a zero nobody vouched for."""
    records = _dispatch(dict(_TICK_RESULT), topic=_TICK_TOPIC)

    assert any(_ZERO_ROWS_ERROR in m for m in _messages(records, logging.ERROR))
    assert not any("NOOP_ATTESTED" in r.getMessage() for r in records)


def test_a_judged_event_whose_upsert_returned_no_row_still_logs_the_error() -> None:
    """A source event is owed a row, whatever the result claims about ticks."""
    records = _dispatch(
        {**_JUDGED_NO_ROW_RESULT, _NOOP_KEY: "tick"}, topic=_JUDGED_TOPIC
    )

    assert any(_ZERO_ROWS_ERROR in m for m in _messages(records, logging.ERROR))
    assert not any("NOOP_ATTESTED" in r.getMessage() for r in records)


def test_a_real_write_logs_neither_line() -> None:
    """Positive control: the callback is reached and can stay quiet."""
    records = _dispatch({"rows_upserted": 2, _NOOP_KEY: "tick"}, topic=_TICK_TOPIC)

    assert not any(
        _ZERO_ROWS_ERROR in m or "NOOP_ATTESTED" in m
        for m in _messages(records, logging.ERROR) + _messages(records, logging.INFO)
    )
