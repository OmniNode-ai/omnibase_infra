# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18326: cover mapping-envelope and non-UTC-offset branches of the extractor."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _extract_projection_envelope_timestamp,
)


@pytest.mark.unit
def test_mapping_envelope_timestamp_is_returned() -> None:
    dt = datetime(2026, 9, 13, 17, 44, 8, tzinfo=UTC)

    assert _extract_projection_envelope_timestamp({"envelope_timestamp": dt}) is dt


@pytest.mark.unit
def test_mapping_envelope_timestamp_naive_is_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dt = datetime(2026, 9, 13, 17, 44, 8)
    logger_name = "omnibase_infra.runtime.auto_wiring.handler_wiring"

    with caplog.at_level(logging.WARNING, logger=logger_name):
        result = _extract_projection_envelope_timestamp({"envelope_timestamp": dt})

    assert result is None
    assert any(
        record.name == logger_name
        and record.levelno == logging.WARNING
        and "timezone-naive" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "envelope",
    [
        pytest.param({}, id="absent"),
        pytest.param({"envelope_timestamp": None}, id="none"),
        pytest.param({"envelope_timestamp": "2026-09-13T17:44:08Z"}, id="string"),
        pytest.param({"envelope_timestamp": 1789000000}, id="integer"),
    ],
)
def test_mapping_envelope_timestamp_absent_or_untyped_is_none(
    envelope: dict[str, object],
) -> None:
    assert _extract_projection_envelope_timestamp(envelope) is None


@pytest.mark.unit
def test_envelope_timestamp_non_utc_offset_is_kept_unchanged() -> None:
    dt = datetime(
        2026, 9, 13, 23, 14, 8, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )
    envelope = SimpleNamespace(envelope_timestamp=dt)

    result = _extract_projection_envelope_timestamp(envelope)

    assert result is dt
    assert result.utcoffset() == timedelta(hours=5, minutes=30)
