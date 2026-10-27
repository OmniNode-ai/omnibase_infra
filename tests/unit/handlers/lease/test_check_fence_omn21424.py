# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21424 AC6 (unreachable store): check_fence answers unknown, never ok."""

from __future__ import annotations

import socket
from typing import cast

import httpx
import pytest

from omnibase_infra.handlers.lease.handler_lease_fence import check_fence
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
        status = await check_fence(adapter, kind="k", resource="r", fence=5)
    finally:
        await adapter.aclose()
    assert status is EnumLeaseFenceStatus.UNKNOWN
    assert status is not EnumLeaseFenceStatus.OK


class _Entries:
    """A prefix read returning fixed slot create_revisions."""

    def __init__(self, revisions: tuple[int, ...]) -> None:
        self._revisions = revisions

    async def get_prefix_entries(self, prefix: str) -> list[tuple[str, str, int, int]]:
        return [(f"{prefix}{i}", "{}", rev, 1) for i, rev in enumerate(self._revisions)]


@pytest.mark.parametrize(
    ("live", "fence", "expected"),
    [
        ((), 5, EnumLeaseFenceStatus.EXPIRED),
        ((5,), 5, EnumLeaseFenceStatus.OK),
        ((9,), 5, EnumLeaseFenceStatus.STALE),
        ((5,), 9, EnumLeaseFenceStatus.EXPIRED),
        ((3, 9), 9, EnumLeaseFenceStatus.OK),
        ((3, 9), 5, EnumLeaseFenceStatus.STALE),
        ((3, 9), 12, EnumLeaseFenceStatus.EXPIRED),
    ],
)
async def test_ac6_a_readable_store_never_answers_unknown(
    live: tuple[int, ...], fence: int, expected: EnumLeaseFenceStatus
) -> None:
    """A fence missing from the linearizable read names no live grant."""
    adapter = cast("HandlerPlacementEtcd", _Entries(live))
    status = await check_fence(adapter, kind="k", resource="r", fence=fence)
    assert status is expected
