# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lease request refusals and cleanup after failed acquires."""

from __future__ import annotations

from uuid import UUID

import httpx
import pytest
from pydantic import ValidationError

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.lease import HandlerLeaseStore
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
    unb64,
)
from omnibase_infra.models.lease import (
    ModelLeaseAcquireRequest,
    ModelLeaseKindPolicy,
    ModelLeaseReleaseRequest,
    ModelLeaseRenewRequest,
)

pytestmark = pytest.mark.unit


def _unexpected_request(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected etcd request {request.url.path}")


class _FakeEtcd(HandlerPlacementEtcd):
    """Records calls; an operation the store should not use fails the test."""

    def __init__(self) -> None:
        super().__init__(
            "http://etcd.test",
            httpx.AsyncClient(transport=httpx.MockTransport(_unexpected_request)),
        )
        self.calls: list[tuple[str, str | int]] = []
        self.txn_error = InfraConnectionError("transaction failed")
        self.revoke_error: InfraConnectionError | None = None
        self.probe_error: InfraConnectionError | None = None
        self.committed = False
        self.kv: dict[str, str] = {}

    async def get(self, key: str) -> tuple[str, int] | None:
        self.calls.append(("get", key))
        if self.probe_error is not None and any(c[0] == "txn" for c in self.calls):
            raise self.probe_error
        if key in self.kv:
            return self.kv[key], 42
        return None

    async def lease_grant(self, ttl_s: int) -> tuple[int, int]:
        self.calls.append(("grant", ttl_s))
        return 123, ttl_s

    async def lease_revoke(self, lease_id: int) -> None:
        self.calls.append(("revoke", lease_id))
        if self.revoke_error is not None:
            raise self.revoke_error

    async def txn(
        self,
        *,
        compare: list[dict[str, object]],
        success: list[dict[str, object]],
        failure: list[dict[str, object]],
    ) -> tuple[bool, int, list[dict[str, object]]]:
        self.calls.append(("txn", 123))
        if self.committed:
            # The transaction took effect but its answer was lost.
            for op in success:
                put = op["requestPut"]
                assert isinstance(put, dict)
                self.kv[unb64(put["key"])] = unb64(put["value"])
        raise self.txn_error

    async def get_prefix_entries(self, prefix: str) -> list[tuple[str, str, int, int]]:
        self.calls.append(("get_prefix_entries", prefix))
        return []


def _request(kind: str = "worker") -> ModelLeaseAcquireRequest:
    return ModelLeaseAcquireRequest(
        kind=kind, resource="resource", holder="holder", request_id=UUID(int=1)
    )


@pytest.mark.parametrize("operation", ["acquire", "renew", "release"])
async def test_unknown_kind_is_refused_without_etcd_calls(operation: str) -> None:
    adapter = _FakeEtcd()
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="worker")])
    request = _request("undeclared-kind")

    if operation == "acquire":
        reply = await store.acquire(request)
        assert reply.request_id == request.request_id
    elif operation == "renew":
        reply = await store.renew(request.kind, request.resource, UUID(int=2))
    else:
        reply = await store.release(request.kind, request.resource, UUID(int=2))

    assert reply.outcome == "lease-refused"
    assert reply.kind == request.kind
    assert reply.resource == request.resource
    assert reply.fence == 0
    assert request.kind in reply.reason
    assert "not declared" in reply.reason
    assert adapter.calls == []


async def test_refusal_survives_a_failed_revoke() -> None:
    class _Refusing(_FakeEtcd):
        async def txn(
            self,
            *,
            compare: list[dict[str, object]],
            success: list[dict[str, object]],
            failure: list[dict[str, object]],
        ) -> tuple[bool, int, list[dict[str, object]]]:
            self.calls.append(("txn", 123))
            return False, 7, []

    adapter = _Refusing()
    adapter.revoke_error = InfraConnectionError("revoke failed")
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="worker")])

    reply = await store.acquire(_request())

    assert reply.outcome == "lease-refused"
    assert ("revoke", 123) in adapter.calls


@pytest.mark.parametrize("revoke_fails", [False, True])
async def test_txn_failure_revokes_grant_and_preserves_original_error(
    revoke_fails: bool,
) -> None:
    adapter = _FakeEtcd()
    if revoke_fails:
        adapter.revoke_error = InfraConnectionError("revoke failed")
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="worker")])
    request = _request()
    request_key = f"/onex/lease-request/worker/resource/{request.request_id}"

    with pytest.raises(InfraConnectionError) as raised:
        await store.acquire(request)

    assert raised.value is adapter.txn_error
    assert adapter.calls == [
        ("get", request_key),
        ("grant", 30),
        ("txn", 123),
        ("get", request_key),
        ("revoke", 123),
    ]


async def test_txn_failure_after_remote_commit_does_not_revoke_grant() -> None:
    adapter = _FakeEtcd()
    adapter.committed = True
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="worker")])
    request = _request()
    request_key = f"/onex/lease-request/worker/resource/{request.request_id}"

    with pytest.raises(InfraConnectionError) as raised:
        await store.acquire(request)

    assert raised.value is adapter.txn_error
    assert adapter.calls == [
        ("get", request_key),
        ("grant", 30),
        ("txn", 123),
        ("get", request_key),
    ]
    assert request_key in adapter.kv


async def test_commit_probe_on_another_key_does_not_read_as_committed() -> None:
    adapter = _FakeEtcd()
    adapter.kv["/onex/lease-request/worker/resource/other"] = "{}"
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="worker")])

    with pytest.raises(InfraConnectionError):
        await store.acquire(_request())

    assert adapter.calls[-1] == ("revoke", 123)


async def test_txn_failure_with_failed_commit_probe_keeps_grant_for_its_ttl() -> None:
    adapter = _FakeEtcd()
    adapter.probe_error = InfraConnectionError("probe failed")
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="worker")])
    request = _request()
    request_key = f"/onex/lease-request/worker/resource/{request.request_id}"

    with pytest.raises(InfraConnectionError) as raised:
        await store.acquire(request)

    assert raised.value is adapter.txn_error
    assert adapter.calls == [
        ("get", request_key),
        ("grant", 30),
        ("txn", 123),
        ("get", request_key),
    ]


async def test_failure_before_txn_revokes_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeEtcd()
    store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="worker")])
    original = InfraConnectionError("lease preparation failed")

    def fail_uuid() -> UUID:
        raise original

    monkeypatch.setattr(
        "omnibase_infra.handlers.lease.handler_lease_store.uuid4", fail_uuid
    )

    with pytest.raises(InfraConnectionError) as raised:
        await store.acquire(_request())

    assert raised.value is original
    assert [call[0] for call in adapter.calls] == ["get", "grant", "revoke"]


@pytest.mark.parametrize("kind", ["a/b", "../exclusive", "Exclusive", "-x"])
def test_request_kind_outside_the_policy_pattern_is_refused(kind: str) -> None:
    lease_id = UUID(int=1)
    with pytest.raises(ValidationError):
        ModelLeaseAcquireRequest(
            kind=kind, resource="r", holder="h", request_id=lease_id
        )
    with pytest.raises(ValidationError):
        ModelLeaseRenewRequest(kind=kind, resource="r", lease_id=lease_id)
    with pytest.raises(ValidationError):
        ModelLeaseReleaseRequest(kind=kind, resource="r", lease_id=lease_id)
