# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21424 AC6: check_fence answers unknown only for an unreachable store, never ok."""

from __future__ import annotations

import socket
from typing import cast

import httpx
import pytest

from omnibase_infra.handlers.lease.handler_lease_fence import HandlerLeaseFence
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import EnumLeaseFenceStatus

pytestmark = pytest.mark.unit


async def test_ac6_unreachable_etcd_is_unknown() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    adapter = HandlerPlacementEtcd(
        f"http://127.0.0.1:{port}", httpx.AsyncClient(timeout=0.5)
    )
    try:
        status = await HandlerLeaseFence(adapter).check_fence(
            kind="k", resource="r", fence=5
        )
    finally:
        await adapter.aclose()
    assert status is EnumLeaseFenceStatus.UNKNOWN
    assert status is not EnumLeaseFenceStatus.OK


class _FakeAdapter:
    def __init__(self, revisions: tuple[int, ...]) -> None:
        self._revisions = revisions

    async def get_prefix_entries(self, prefix: str) -> list[tuple[str, str, int, int]]:
        return [(f"{prefix}{i}", "v", rev, 1) for i, rev in enumerate(self._revisions)]


@pytest.mark.parametrize(
    ("fence", "expected"),
    [
        (7, EnumLeaseFenceStatus.OK),
        (9, EnumLeaseFenceStatus.OK),
        (5, EnumLeaseFenceStatus.STALE),
        (8, EnumLeaseFenceStatus.STALE),
        (12, EnumLeaseFenceStatus.EXPIRED),
    ],
)
async def test_a_fence_above_every_live_grant_is_expired_not_unknown(
    fence: int, expected: EnumLeaseFenceStatus
) -> None:
    adapter = cast("HandlerPlacementEtcd", _FakeAdapter((7, 9)))
    status = await HandlerLeaseFence(adapter).check_fence(
        kind="k", resource="r", fence=fence
    )
    assert status is expected
