# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The cost snapshot cache is keyed by snapshot kind and window, never by a topic string."""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_cost_snapshot_kind import EnumCostSnapshotKind
from omnibase_infra.services.cost_api import snapshot_cache


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    snapshot_cache.clear_latest_snapshots()


@pytest.mark.unit
def test_snapshots_are_kept_per_kind_and_window() -> None:
    snapshot_cache.store_latest_snapshot(
        EnumCostSnapshotKind.SUMMARY, "24h", {"total_tokens": 1}
    )
    snapshot_cache.store_latest_snapshot(
        EnumCostSnapshotKind.BY_REPO, "24h", {"total_tokens": 2}
    )

    assert snapshot_cache.get_latest_snapshot(EnumCostSnapshotKind.SUMMARY, "24h") == {
        "total_tokens": 1
    }
    assert snapshot_cache.get_latest_snapshot(EnumCostSnapshotKind.BY_REPO, "24h") == {
        "total_tokens": 2
    }
    assert (
        snapshot_cache.get_latest_snapshot(EnumCostSnapshotKind.TOKEN_USAGE, "24h")
        is None
    )
    assert (
        snapshot_cache.get_latest_snapshot(EnumCostSnapshotKind.SUMMARY, "7d") is None
    )


@pytest.mark.unit
def test_kind_values_hold_no_topic_string() -> None:
    assert all(not kind.value.startswith("onex.") for kind in EnumCostSnapshotKind)
