# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""AC2 (OMN-26249): every field of the reading is required, none is optional."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_runtime_loop_reading import (
    ModelRuntimeLoopReading,
)

_START = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)


def _fields() -> dict[str, object]:
    return {
        "member_id": "dev.svc.node.v1:42",
        "host": "h201",
        "window_start": _START,
        "window_end": _START + timedelta(seconds=15),
        "max_gap_ms": 12.5,
        "tick_lag_ms": 40.0,
        "tick_lag_absent_reason": None,
        "abandoned_count": 0,
    }


@pytest.mark.unit
def test_a_complete_reading_validates() -> None:
    assert ModelRuntimeLoopReading(**_fields()).max_gap_ms == 12.5  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("omitted", sorted(_fields()))
def test_omitting_any_field_is_a_validation_error(omitted: str) -> None:
    fields = _fields()
    del fields[omitted]
    with pytest.raises(ValidationError):
        ModelRuntimeLoopReading(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_a_non_subscriber_states_why_it_has_no_tick_lag() -> None:
    fields = _fields() | {"tick_lag_ms": None, "tick_lag_absent_reason": "not_a_tick_subscriber"}
    reading = ModelRuntimeLoopReading(**fields)  # type: ignore[arg-type]
    assert reading.tick_lag_absent_reason == "not_a_tick_subscriber"


@pytest.mark.unit
def test_tick_lag_and_reason_together_or_neither_are_refused() -> None:
    both = _fields() | {"tick_lag_absent_reason": "not_a_tick_subscriber"}
    neither = _fields() | {"tick_lag_ms": None}
    for fields in (both, neither):
        with pytest.raises(ValidationError):
            ModelRuntimeLoopReading(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_a_window_that_ends_before_it_starts_is_refused() -> None:
    fields = _fields() | {"window_end": _START - timedelta(seconds=1)}
    with pytest.raises(ValidationError):
        ModelRuntimeLoopReading(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_naive_timestamps_are_refused() -> None:
    fields = _fields() | {"window_start": datetime(2026, 11, 20, 12, 0, 0)}  # noqa: DTZ001
    with pytest.raises(ValidationError):
        ModelRuntimeLoopReading(**fields)  # type: ignore[arg-type]
