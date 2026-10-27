# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Placement session setup publishes state only after membership succeeds."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
)
from omnibase_infra.handlers.placement.handler_placement_session import (
    MEMBERS_PREFIX,
    HandlerPlacementSession,
)

pytestmark = pytest.mark.unit


class _FakeAdapter(HandlerPlacementEtcd):
    def __init__(
        self,
        *,
        granted_ttl_s: int = 4,
        put_error: Exception | None = None,
        revoke_error: Exception | None = None,
        on_put: Callable[[], None] | None = None,
    ) -> None:
        self.granted_ttl_s = granted_ttl_s
        self.put_error = put_error
        self.revoke_error = revoke_error
        self.on_put = on_put
        self.grants: list[int] = []
        self.revoked: list[int] = []
        self.members: dict[str, tuple[str, int]] = {}

    async def lease_grant(self, ttl_s: int) -> tuple[int, int]:
        self.grants.append(ttl_s)
        return 42, self.granted_ttl_s

    async def put(self, key: str, value: str, *, lease_id: int) -> int:
        if self.on_put is not None:
            self.on_put()
        if self.put_error is not None:
            raise self.put_error
        self.members[key] = (value, lease_id)
        return 1

    async def lease_revoke(self, lease_id: int) -> None:
        self.revoked.append(lease_id)
        if self.revoke_error is not None:
            raise self.revoke_error


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_put_failure_keeps_session_closed_and_revokes_lease(
    cleanup_fails: bool,
) -> None:
    original = InfraConnectionError("membership put failed")
    adapter = _FakeAdapter(
        put_error=original,
        revoke_error=RuntimeError("revoke failed") if cleanup_fails else None,
    )
    session = HandlerPlacementSession(
        adapter=adapter,
        runtime_id="runtime-a",
        node_names=["node-a"],
        clock=lambda: 100.0,
    )

    with pytest.raises(InfraConnectionError) as raised:
        await session.open()

    assert raised.value is original
    assert not session.within_deadline()
    assert session.lease_id == 0
    assert session._deadline == 0.0
    assert adapter.members == {}
    assert adapter.revoked == [42]


@pytest.mark.parametrize("granted_ttl_s", [1, 2])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_ttl_at_or_below_margin_revokes_lease(
    granted_ttl_s: int, cleanup_fails: bool
) -> None:
    adapter = _FakeAdapter(
        granted_ttl_s=granted_ttl_s,
        revoke_error=RuntimeError("revoke failed") if cleanup_fails else None,
    )
    session = HandlerPlacementSession(
        adapter=adapter,
        runtime_id="runtime-a",
        node_names=["node-a"],
        margin_s=2.0,
        clock=lambda: 100.0,
    )

    with pytest.raises(InfraConnectionError, match="etcd granted TTL"):
        await session.open()

    assert not session.within_deadline()
    assert session.lease_id == 0
    assert session._deadline == 0.0
    assert adapter.members == {}
    assert adapter.revoked == [42]


async def test_success_publishes_state_after_put_with_grant_send_deadline() -> None:
    now = 100.0

    def on_put() -> None:
        nonlocal now
        assert session.lease_id == 0
        assert not session.within_deadline()
        assert session._deadline == 0.0
        now = 101.0

    adapter = _FakeAdapter(on_put=on_put)
    session = HandlerPlacementSession(
        adapter=adapter,
        runtime_id="runtime-a",
        node_names=["node-a", "node-b"],
        clock=lambda: now,
    )

    await session.open()

    assert adapter.grants == [4]
    value, lease_id = adapter.members[f"{MEMBERS_PREFIX}runtime-a"]
    assert json.loads(value) == {
        "runtime_id": "runtime-a",
        "nodes": ["node-a", "node-b"],
    }
    assert lease_id == session.lease_id == 42
    assert adapter.revoked == []
    assert session.within_deadline()
    now = 102.5
    assert not session.within_deadline()
