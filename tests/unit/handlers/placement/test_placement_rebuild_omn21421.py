# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21421 AC3: a new primary folds from the stored offsets to the old primary's rows."""

from __future__ import annotations

import json

import pytest

from omnibase_infra.handlers.placement import (
    ModelPlacementReplayEvent,
    fold_projection_from_offsets,
)

pytestmark = pytest.mark.unit


def _fold(rows: dict[str, str], event: ModelPlacementReplayEvent) -> dict[str, str]:
    """A reducer-style fold: a running count and the last value per key."""
    current = json.loads(rows.get(event.key, '{"count": 0, "last": ""}'))
    updated = {"count": current["count"] + 1, "last": event.value}
    return {**rows, event.key: json.dumps(updated, sort_keys=True)}


def _log() -> list[ModelPlacementReplayEvent]:
    events = []
    for offset in range(40):
        for partition in (0, 1):
            events.append(
                ModelPlacementReplayEvent(
                    partition=partition,
                    offset=offset,
                    key=f"k{(offset * 7 + partition) % 5}",
                    value=f"v{partition}-{offset}",
                )
            )
    return events


def test_new_primary_fold_from_stored_offsets_matches_old_primary_rows() -> None:
    log = _log()
    # The old primary folds everything, storing rows and next offsets at a checkpoint.
    checkpoint_at = {0: 17, 1: 23}
    stored_rows, stored_offsets = fold_projection_from_offsets(
        rows={},
        offsets={},
        events=[e for e in log if e.offset < checkpoint_at[e.partition]],
        fold=_fold,
    )
    assert stored_offsets == checkpoint_at
    old_rows, old_offsets = fold_projection_from_offsets(
        rows={}, offsets={}, events=log, fold=_fold
    )

    # The new primary starts from the stored table and offsets and replays the
    # whole topic from the beginning: events below the stored offsets are skipped.
    new_rows, new_offsets = fold_projection_from_offsets(
        rows=stored_rows, offsets=stored_offsets, events=log, fold=_fold
    )

    diff = {
        key
        for key in old_rows.keys() | new_rows.keys()
        if old_rows.get(key) != new_rows.get(key)
    }
    assert diff == set()
    assert new_offsets == old_offsets == {0: 40, 1: 40}


def test_fold_does_not_mutate_the_stored_rows() -> None:
    log = _log()
    stored = {"k0": json.dumps({"count": 99, "last": "x"}, sort_keys=True)}
    snapshot = dict(stored)
    fold_projection_from_offsets(rows=stored, offsets={}, events=log, fold=_fold)
    assert stored == snapshot


def test_an_event_below_the_stored_offset_is_not_folded_twice() -> None:
    event = ModelPlacementReplayEvent(partition=0, offset=3, key="a", value="v")
    rows, offsets = fold_projection_from_offsets(
        rows={}, offsets={0: 4}, events=[event], fold=_fold
    )
    assert rows == {}
    assert offsets == {0: 4}
