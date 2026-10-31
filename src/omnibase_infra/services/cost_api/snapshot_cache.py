# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Process-local cache for latest cost projection snapshots; topics from cost_api_topics.yaml."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from omnibase_infra.models.cost_api import load_cost_api_topics

SnapshotPayload = Mapping[str, object]

TOPIC_COST_SUMMARY = load_cost_api_topics().summary
TOPIC_COST_BY_REPO = load_cost_api_topics().by_repo
TOPIC_COST_TOKEN_USAGE = load_cost_api_topics().token_usage

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
    "TOPIC_COST_BY_REPO",
    "TOPIC_COST_SUMMARY",
    "TOPIC_COST_TOKEN_USAGE",
    "clear_latest_snapshots",
    "get_latest_snapshot",
    "store_latest_snapshot",
]
