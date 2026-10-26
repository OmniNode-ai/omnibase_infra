# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection rebuild for a new primary (OMN-21421 AC3, rebuild-then-consume).

A newly elected primary folds its projection from the stored projection table
and the stored committed offsets before it joins the consumer group, so it
serves the same rows the old primary served.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from omnibase_infra.handlers.placement.model_placement_replay_event import (
    ModelPlacementReplayEvent,
)


def fold_projection_from_offsets(
    *,
    rows: Mapping[str, str],
    offsets: Mapping[int, int],
    events: Iterable[ModelPlacementReplayEvent],
    fold: Callable[[dict[str, str], ModelPlacementReplayEvent], dict[str, str]],
) -> tuple[dict[str, str], dict[int, int]]:
    """Fold replay events into ``rows``, skipping events already stored.

    ``offsets`` maps partition -> the next offset to consume, following Kafka
    committed-offset semantics; a partition missing from the mapping means its
    next offset is 0.

    The fold starts from copies of ``rows`` and ``offsets``; the inputs are
    never mutated. Events are replayed in order: an event whose offset is below
    its partition's current next offset is skipped because it is already folded
    into the stored rows, otherwise it is folded and that partition's next
    offset advances to the event's offset plus one.

    This is what lets a new primary replay its topic from the very start:
    events below the stored offsets are already in the stored rows, so
    replaying the topic from its start reaches the same rows as the old
    primary, and the returned offsets match the old primary's checkpoints.

    Returns a new ``(rows, offsets)`` pair of plain dicts: the folded
    projection and each partition's next offset to consume.
    """
    merged: dict[str, str] = dict(rows)
    next_offsets: dict[int, int] = dict(offsets)
    for event in events:
        if event.offset < next_offsets.get(event.partition, 0):
            continue
        merged = fold(merged, event)
        next_offsets[event.partition] = event.offset + 1
    return merged, next_offsets


__all__ = ["fold_projection_from_offsets"]
