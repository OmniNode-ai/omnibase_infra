# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Original and injected etcd clients round-trip a real lease on the lab runtime."""

from __future__ import annotations

import httpx
import pytest

from omnibase_infra.handlers.placement import HandlerPlacementEtcd

from .conftest import EtcdProcess

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]


@pytest.mark.parametrize("mode", ["default", "timeout", "injected"])
async def test_constructor_lease_round_trip(etcd: EtcdProcess, mode: str) -> None:
    if mode == "default":
        adapter = HandlerPlacementEtcd(etcd.endpoint)
    elif mode == "timeout":
        adapter = HandlerPlacementEtcd(etcd.endpoint, timeout_s=2.0)
    else:
        adapter = HandlerPlacementEtcd(etcd.endpoint, httpx.AsyncClient(timeout=2.0))
    try:
        lease_id, ttl = await adapter.lease_grant(5)
        assert lease_id != 0
        assert ttl == 5
        created, revision = await adapter.create_if_absent(
            "/compatibility", "held", lease_id=lease_id
        )
        assert created and revision > 0
        assert await adapter.get("/compatibility") == ("held", revision)
        assert await adapter.lease_keepalive(lease_id) > 0
        await adapter.lease_revoke(lease_id)
        assert await adapter.get("/compatibility") is None
    finally:
        await adapter.aclose()
