# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Failed acquires clean up unused grants and preserve committed transactions."""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.lease import HandlerLeaseStore
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import ModelLeaseAcquireRequest, ModelLeaseKindPolicy

pytestmark = pytest.mark.unit


class FailingEtcd(HandlerPlacementEtcd):
    """Record a grant, a timed-out transaction and its cleanup."""

    def __init__(
        self,
        *,
        committed: bool = False,
        revoke_fails: bool = False,
        recovery_read_fails: bool = False,
    ) -> None:
        self.committed = committed
        self.revoke_fails = revoke_fails
        self.recovery_read_fails = recovery_read_fails
        self.error = InfraConnectionError("etcd transaction failed")
        self.reads: list[str] = []
        self.grants: list[int] = []
        self.revoked: list[int] = []

    async def get(self, key: str) -> tuple[str, int] | None:
        self.reads.append(key)
        if self.recovery_read_fails and len(self.reads) > 1:
            raise InfraConnectionError("etcd recovery read failed")
        if self.committed and len(self.reads) > 1:
            return (
                '{"holder": "h", "lease_id": "00000000-0000-0000-0000-000000000001"}',
                7,
            )
        return None

    async def lease_grant(self, ttl_s: int) -> tuple[int, int]:
        self.grants.append(123)
        return 123, ttl_s

    async def txn(
        self,
        *,
        compare: list[dict[str, object]],
        success: list[dict[str, object]],
        failure: list[dict[str, object]],
    ) -> tuple[bool, int, list[dict[str, object]]]:
        raise self.error

    async def lease_revoke(self, lease_id: int) -> None:
        self.revoked.append(lease_id)
        if self.revoke_fails:
            raise InfraConnectionError("etcd revoke failed")


@pytest.fixture
def acquire_request() -> ModelLeaseAcquireRequest:
    return ModelLeaseAcquireRequest(
        kind="slot", resource="host", holder="h", request_id=UUID(int=1)
    )


@pytest.mark.parametrize("revoke_fails", [False, True])
async def test_failed_txn_revokes_grant_and_preserves_original_error(
    acquire_request: ModelLeaseAcquireRequest, revoke_fails: bool
) -> None:
    adapter = FailingEtcd(revoke_fails=revoke_fails)
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="slot")])
    with pytest.raises(InfraConnectionError) as caught:
        await store.acquire(acquire_request)
    assert caught.value is adapter.error
    assert adapter.grants == [123]
    assert (
        adapter.reads
        == ["/onex/lease-request/slot/host/" + str(acquire_request.request_id)] * 2
    )
    assert adapter.revoked == [123]


async def test_timed_out_committed_txn_is_not_revoked(
    acquire_request: ModelLeaseAcquireRequest,
) -> None:
    adapter = FailingEtcd(committed=True)
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="slot")])
    with pytest.raises(InfraConnectionError) as caught:
        await store.acquire(acquire_request)
    assert caught.value is adapter.error
    assert len(adapter.reads) == 2
    assert adapter.revoked == []


async def test_recovery_read_and_revoke_failures_preserve_original_error(
    acquire_request: ModelLeaseAcquireRequest,
) -> None:
    adapter = FailingEtcd(recovery_read_fails=True, revoke_fails=True)
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="slot")])
    with pytest.raises(InfraConnectionError) as caught:
        await store.acquire(acquire_request)
    assert caught.value is adapter.error
    assert adapter.revoked == [123]


async def test_failure_before_txn_revokes_grant(
    acquire_request: ModelLeaseAcquireRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FailingEtcd()
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="slot")])

    def fail_uuid() -> UUID:
        raise adapter.error

    monkeypatch.setattr(
        "omnibase_infra.handlers.lease.handler_lease_store.uuid4", fail_uuid
    )
    with pytest.raises(InfraConnectionError) as caught:
        await store.acquire(acquire_request)
    assert caught.value is adapter.error
    assert adapter.revoked == [123]
