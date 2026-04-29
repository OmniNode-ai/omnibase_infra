# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Process-local cache for latest cost projection snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

SnapshotPayload = Mapping[str, object]

TOPIC_COST_SUMMARY: Final[str] = "onex.snapshot.projection.cost.summary.v1"  # noqa: RUF100  # noqa: topic-naming-lint
TOPIC_COST_BY_REPO: Final[str] = "onex.snapshot.projection.cost.by_repo.v1"  # noqa: RUF100  # noqa: topic-naming-lint
TOPIC_COST_TOKEN_USAGE: Final[str] = "onex.snapshot.projection.cost.token_usage.v1"  # noqa: RUF100  # noqa: topic-naming-lint

_LATEST: dict[tuple[str, str], dict[str, object]] = {}


def store_latest_snapshot(topic: str, window: str, payload: SnapshotPayload) -> None:
    """Store the latest snapshot for an API route fallback boundary."""
    _LATEST[(topic, window)] = dict(payload)


def get_latest_snapshot(topic: str, window: str) -> dict[str, object] | None:
    """Return a copy of the latest snapshot payload for a topic/window."""
    payload = _LATEST.get((topic, window))
    if payload is None:
        return None
    return dict(payload)


def clear_latest_snapshots() -> None:
    """Clear cached snapshots; intended for tests."""
    _LATEST.clear()


__all__ = [
    "TOPIC_COST_BY_REPO",
    "TOPIC_COST_SUMMARY",
    "TOPIC_COST_TOKEN_USAGE",
    "clear_latest_snapshots",
    "get_latest_snapshot",
    "store_latest_snapshot",
]
