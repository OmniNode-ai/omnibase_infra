# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unused etcd leases are cleaned up without losing grants or original errors."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.lease.handler_lease_store import HandlerLeaseStore
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
    unb64,
)
from omnibase_infra.models.lease import ModelLeaseAcquireRequest, ModelLeaseKindPolicy

pytestmark = pytest.mark.unit


class FakeEtcd(HandlerPlacementEtcd):
    """A single transaction, optionally committed before its response is lost."""

    def __init__(self) -> None:
        self.txn_error: Exception | None = None
        self.revoke_error: Exception | None = None
        self.read_error: Exception | None = None
        self.commit = False
        self.replay = False
        self.revoked: list[int] = []
        self.entries: list[tuple[str, str, int, int]] = []
        self.request_key = ""

    async def get(self, key: str) -> tuple[str, int] | None:
        self.request_key = key
        return None

    async def lease_grant(self, ttl_s: int) -> tuple[int, int]:
        return 123, ttl_s

    async def lease_revoke(self, lease_id: int) -> None:
        self.revoked.append(lease_id)
        if self.revoke_error is not None:
            raise self.revoke_error

    async def get_prefix_entries(self, prefix: str) -> list[tuple[str, str, int, int]]:
        assert prefix == self.request_key
        if self.read_error is not None:
            raise self.read_error
        return self.entries

    async def txn(
        self,
        *,
        compare: list[dict[str, object]],
        success: list[dict[str, object]],
        failure: list[dict[str, object]],
    ) -> tuple[bool, int, list[dict[str, object]]]:
        put = success[1]["requestPut"]
        assert isinstance(put, dict)
        if self.commit:
            self.entries = [(self.request_key, unb64(put["value"]), 42, 123)]
        if self.txn_error is not None:
            raise self.txn_error
        responses: list[dict[str, object]] = []
        if self.replay:
            responses = [
                {
                    "response_range": {
                        "kvs": [{"value": put["value"], "create_revision": "41"}]
                    }
                }
            ]
        return self.commit, 42, responses


def _request() -> ModelLeaseAcquireRequest:
    return ModelLeaseAcquireRequest(
        kind="exclusive", resource="r", holder="worker", request_id=uuid4()
    )


def _store(adapter: FakeEtcd) -> HandlerLeaseStore:
    return HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="exclusive")])


@pytest.mark.parametrize("error_type", [InfraConnectionError, TimeoutError])
@pytest.mark.parametrize("revoke_fails", [False, True])
async def test_txn_failure_revokes_and_preserves_original_error(
    error_type: type[Exception], revoke_fails: bool
) -> None:
    adapter = FakeEtcd()
    original = error_type("transaction failed")
    adapter.txn_error = original
    if revoke_fails:
        adapter.revoke_error = InfraConnectionError("revoke failed")
    with pytest.raises(error_type) as raised:
        await _store(adapter).acquire(_request())
    assert raised.value is original
    assert adapter.revoked == [123]


async def test_a_remotely_committed_grant_is_replayed_without_revocation() -> None:
    adapter = FakeEtcd()
    adapter.commit = True
    adapter.txn_error = InfraConnectionError("transaction response lost")
    request = _request()
    reply = await _store(adapter).acquire(request)
    assert reply.outcome == "lease-granted"
    assert reply.reason == "replayed request_id"
    assert reply.fence == 42
    assert reply.request_id == request.request_id
    assert str(reply.lease_id) in adapter.entries[0][1]
    assert adapter.revoked == []


async def test_an_unreadable_commit_preserves_original_error_and_lease() -> None:
    adapter = FakeEtcd()
    original = InfraConnectionError("transaction failed")
    adapter.txn_error = original
    adapter.read_error = InfraConnectionError("read failed")
    with pytest.raises(InfraConnectionError) as raised:
        await _store(adapter).acquire(_request())
    assert raised.value is original
    assert adapter.revoked == []


@pytest.mark.parametrize("replay", [False, True])
async def test_cleanup_failure_does_not_replace_refusal_or_replay(replay: bool) -> None:
    adapter = FakeEtcd()
    adapter.replay = replay
    adapter.revoke_error = InfraConnectionError("revoke failed")
    reply = await _store(adapter).acquire(_request())
    assert reply.outcome == ("lease-granted" if replay else "lease-refused")
    assert adapter.revoked == [123]


async def test_an_accepted_grant_retains_its_lease() -> None:
    adapter = FakeEtcd()
    adapter.commit = True
    reply = await _store(adapter).acquire(_request())
    assert reply.outcome == "lease-granted"
    assert adapter.revoked == []
