# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21424 AC6 (unreachable store): check_fence answers unknown, never ok."""

from __future__ import annotations

import socket

import pytest

from omnibase_infra.handlers.lease.handler_lease_fence import check_fence
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import EnumLeaseFenceStatus

pytestmark = pytest.mark.unit


async def test_ac6_unreachable_etcd_is_unknown() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    adapter = HandlerPlacementEtcd(f"http://127.0.0.1:{port}", timeout_s=0.5)
    try:
        status = await check_fence(adapter, kind="k", resource="r", fence=5)
    finally:
        await adapter.aclose()
    assert status is EnumLeaseFenceStatus.UNKNOWN
    assert status is not EnumLeaseFenceStatus.OK
