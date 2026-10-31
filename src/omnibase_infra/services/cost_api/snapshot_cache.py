# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Process-local cache for latest cost projection snapshots.

Entries are keyed by topic and window; the topics come from the registry API node
contract (``config.cost_snapshot_topics``) and are injected by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

SnapshotPayload = Mapping[str, object]


_LATEST: dict[tuple[str, str], dict[str, object]] = {}


def store_latest_snapshot(topic: str, window: str, payload: SnapshotPayload) -> None:
    """Store the latest snapshot for an API route fallback boundary."""
    _LATEST[(topic, window)] = deepcopy(dict(payload))


def get_latest_snapshot(topic: str, window: str) -> dict[str, object] | None:
    """Return a copy of the latest snapshot payload for a topic/window."""
    payload = _LATEST.get((topic, window))
    if payload is None:
        return None
    return deepcopy(payload)


def clear_latest_snapshots() -> None:
    """Clear cached snapshots; intended for tests."""
    _LATEST.clear()


__all__ = [
    "clear_latest_snapshots",
    "get_latest_snapshot",
    "store_latest_snapshot",
]
