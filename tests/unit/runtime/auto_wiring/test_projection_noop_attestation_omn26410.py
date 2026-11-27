# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-26410: a handler can attest a designed zero-row no-op, and only that logs INFO.

The dev runtime logged ``Projection handler wrote zero rows (no terminal
emitted)`` at ERROR about once a second for ``EvalResultsProjectionWriter`` on
``onex.intent.platform.runtime-tick.v1``. A tick returns ``rows_upserted: 0`` by
design (it republishes a snapshot once per interval), and the classifier could
not tell that from a lost write, so every tick read as a failed write.

This drives the REAL callback from ``_make_projection_dispatch_callback``, as
``test_omn18992_guard_refused_zero_rows_integration`` does, and reads the log
records the runtime emitted. Both directions are asserted, because only one is
the defect:

* a result carrying a valid attestation for its topic -> INFO, no ERROR;
* a zero with no attestation, a malformed one, a tick attestation on a source
  topic, and a valid judged event whose upsert returned no row -> the ERROR.

The attestation values are spelled as the wire strings a producer sends, so
this file reads the same against a classifier that predates the field (RED) and
one that honours it (GREEN).
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

pytestmark = pytest.mark.unit

_WIRING_LOGGER = "omnibase_infra.runtime.auto_wiring.handler_wiring"
_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter"
)
_PATCH_ENVIRON_GET = "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get"
_TEST_DSN = "postgresql://user:pass@host:5432/omnidash_analytics"
_TICK_TOPIC = "onex.intent.platform.runtime-tick.v1"
_JUDGED_TOPIC = "onex.evt.omnimarket.delegation-acceptance-judged.v1"
_NOOP_KEY = "projection_noop_reason"
_ZERO_ROWS_ERROR = "wrote zero rows (no terminal emitted)"

#: The two result shapes recorded on the dev runtime (container omninode-runtime,
#: 2026-11-23 00:02Z to 00:32Z, 1,259 and 3 lines on the tick topic).
_TICK_INTERVAL_RESULT: dict[str, Any] = {
    "rows_upserted": 0,
    "snapshot": {"published": False, "reason": "interval"},
}
_TICK_PUBLISHED_RESULT: dict[str, Any] = {
    "rows_upserted": 0,
    "snapshot": {"published": True, "published_at": "2026-11-23T00:02:33+00:00"},
}
#: A judged event the writer consumed and refused to write, with nothing attested.
_JUDGED_NO_ROW_RESULT: dict[str, Any] = {
    "rows_upserted": 0,
    "items_written": 0,
    "traces_written": 0,
    "run_written": 0,
    "rates_written": 0,
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
        envelope.correlation_id = "omn-26410"
        with patch(_PATCH_ENVIRON_GET, return_value=_TEST_DSN):
            with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
                asyncio.run(callback(envelope))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return records


def _messages(records: list[logging.LogRecord], level: int) -> list[str]:
    return [r.getMessage() for r in records if r.levelno == level]


class TestAnAttestedNoopIsInfo:
    """The change: a handler that says why it wrote nothing is not a failed write."""

    @pytest.mark.parametrize(
        "snapshot_result", [_TICK_INTERVAL_RESULT, _TICK_PUBLISHED_RESULT]
    )
    def test_an_attested_tick_logs_info_and_no_error(
        self, snapshot_result: dict[str, Any]
    ) -> None:
        records = _dispatch({**snapshot_result, _NOOP_KEY: "tick"}, topic=_TICK_TOPIC)

        errors = _messages(records, logging.ERROR)
        info = _messages(records, logging.INFO)

        assert not any(_ZERO_ROWS_ERROR in m for m in errors), (
            f"an attested tick still logged the zero-rows ERROR: {errors}"
        )
        assert any("NOOP_ATTESTED" in m and "reason=tick" in m for m in info), (
            f"no INFO line naming the attested reason; INFO records were {info}"
        )

    def test_an_attested_skip_on_a_source_topic_logs_info(self) -> None:
        records = _dispatch(
            {"rows_upserted": 0, "skipped": "not goal-scoped", _NOOP_KEY: "skipped"},
            topic="onex.evt.omnimarket.dod-verify-completed.v1",
        )

        assert not any(_ZERO_ROWS_ERROR in m for m in _messages(records, logging.ERROR))
        assert any(
            "NOOP_ATTESTED" in m and "reason=skipped" in m
            for m in _messages(records, logging.INFO)
        )


class TestAnythingElseStaysAnError:
    """The half that must not be softened."""

    @pytest.mark.parametrize(
        "snapshot_result", [_TICK_INTERVAL_RESULT, _TICK_PUBLISHED_RESULT]
    )
    def test_an_unattested_tick_zero_still_logs_the_error(
        self, snapshot_result: dict[str, Any]
    ) -> None:
        records = _dispatch(dict(snapshot_result), topic=_TICK_TOPIC)

        assert any(_ZERO_ROWS_ERROR in m for m in _messages(records, logging.ERROR))
        assert not any("NOOP_ATTESTED" in r.getMessage() for r in records)

    def test_a_valid_judged_event_whose_upsert_returns_no_row_still_logs_the_error(
        self,
    ) -> None:
        records = _dispatch(dict(_JUDGED_NO_ROW_RESULT), topic=_JUDGED_TOPIC)

        assert any(_ZERO_ROWS_ERROR in m for m in _messages(records, logging.ERROR))
        assert not any("NOOP_ATTESTED" in r.getMessage() for r in records)

    def test_a_tick_attestation_on_a_source_topic_still_logs_the_error(self) -> None:
        """A tick is the timer topic; a source event is owed a row whatever it claims."""
        records = _dispatch(
            {**_JUDGED_NO_ROW_RESULT, _NOOP_KEY: "tick"}, topic=_JUDGED_TOPIC
        )

        assert any(_ZERO_ROWS_ERROR in m for m in _messages(records, logging.ERROR))
        assert not any("NOOP_ATTESTED" in r.getMessage() for r in records)

    @pytest.mark.parametrize(
        "reason", ["TICK", "Tick ", "interval", "", None, 1, True, ["tick"], {}]
    )
    def test_an_unknown_or_malformed_reason_still_logs_the_error(
        self, reason: object
    ) -> None:
        records = _dispatch(
            {**_TICK_INTERVAL_RESULT, _NOOP_KEY: reason}, topic=_TICK_TOPIC
        )

        assert any(_ZERO_ROWS_ERROR in m for m in _messages(records, logging.ERROR))
        assert not any("NOOP_ATTESTED" in r.getMessage() for r in records)

    def test_an_attestation_beside_a_nonzero_count_is_just_a_write(self) -> None:
        """Positive control: the callback runs and a real write logs neither line."""
        records = _dispatch({"rows_upserted": 2, _NOOP_KEY: "tick"}, topic=_TICK_TOPIC)

        assert not any(
            _ZERO_ROWS_ERROR in m or "NOOP_ATTESTED" in m
            for m in _messages(records, logging.ERROR)
            + _messages(records, logging.INFO)
        )
