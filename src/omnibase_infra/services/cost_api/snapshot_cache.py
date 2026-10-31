# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Process-local cache for latest cost projection snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from omnibase_infra.enums.enum_cost_snapshot_kind import EnumCostSnapshotKind

SnapshotPayload = Mapping[str, object]

_LATEST: dict[tuple[EnumCostSnapshotKind, str], dict[str, object]] = {}


def store_latest_snapshot(
    kind: EnumCostSnapshotKind, window: str, payload: SnapshotPayload
) -> None:
    """Store the latest snapshot for an API route fallback boundary."""
    _LATEST[(kind, window)] = deepcopy(dict(payload))


def get_latest_snapshot(
    kind: EnumCostSnapshotKind, window: str
) -> dict[str, object] | None:
    """Return a copy of the latest snapshot payload for a kind/window."""
    payload = _LATEST.get((kind, window))
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
