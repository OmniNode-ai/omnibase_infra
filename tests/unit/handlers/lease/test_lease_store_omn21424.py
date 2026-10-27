# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerLeaseStore fence source and etcd lease cleanup (OMN-21424)."""

from __future__ import annotations

import asyncio
import json
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.lease.handler_lease_store import HandlerLeaseStore
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
    b64,
    unb64,
)
from omnibase_infra.models.lease import (
    ModelLeaseAcquireRequest,
    ModelLeaseKindPolicy,
    ModelLeaseReleaseRequest,
    ModelLeaseRenewRequest,
)

pytestmark = pytest.mark.unit

_SLOT_REVISION = 41
_HEADER_REVISION = 97
_REQUEST_REVISION = 63


class _FakeEtcd:
    """The adapter calls the store makes; revisions differ per key on purpose."""

    def __init__(self, *, txn_raises: bool = False) -> None:
        self.txn_raises = txn_raises
        self.revoked: list[int] = []
        self.slots: dict[str, tuple[str, int, int]] = {}
        self.requests: dict[str, str] = {}
        self.read_keys: list[str] = []

    async def get(self, key: str) -> tuple[str, int] | None:
        self.read_keys.append(key)
        value = self.requests.get(key)
        if value is None:
            return None
        return value, _REQUEST_REVISION

    async def lease_grant(self, ttl_s: int) -> tuple[int, int]:
        return 7, ttl_s

    async def lease_revoke(self, lease_id: int) -> None:
        self.revoked.append(lease_id)

    async def txn(
        self,
        *,
        compare: list[dict[str, object]],
        success: list[dict[str, object]],
        failure: list[dict[str, object]],
    ) -> tuple[bool, int, list[dict[str, object]]]:
        if self.txn_raises:
            raise InfraConnectionError("etcd unreachable")
        put = success[0]["requestPut"]
        assert isinstance(put, dict)
        key = unb64(put["key"])
        value = unb64(put["value"])
        self.slots[key] = (value, _SLOT_REVISION, int(str(put["lease"])))
        request_put = success[1]["requestPut"]
        assert isinstance(request_put, dict)
        self.requests[unb64(request_put["key"])] = unb64(request_put["value"])
        return (
            True,
            _HEADER_REVISION,
            [
                {"response_put": {}},
                {"response_put": {}},
                {
                    "response_range": {
                        "kvs": [
                            {
                                "key": b64(key),
                                "value": b64(value),
                                "create_revision": str(_SLOT_REVISION),
                            }
                        ]
                    }
                },
            ],
        )

    async def get_prefix_entries(self, prefix: str) -> list[tuple[str, str, int, int]]:
        return [
            (key, value, revision, lease)
            for key, (value, revision, lease) in self.slots.items()
            if key.startswith(prefix)
        ]


def _store(fake: _FakeEtcd) -> HandlerLeaseStore:
    return HandlerLeaseStore(
        cast("HandlerPlacementEtcd", fake),
        [ModelLeaseKindPolicy(kind="exclusive", capacity=1, ttl_s=30)],
    )


def _request() -> ModelLeaseAcquireRequest:
    return ModelLeaseAcquireRequest(
        kind="exclusive", resource="r1", holder="runtime-a", request_id=uuid4()
    )


async def test_a_raising_slot_txn_revokes_the_fresh_etcd_lease() -> None:
    fake = _FakeEtcd(txn_raises=True)
    with pytest.raises(InfraConnectionError):
        await _store(fake).acquire(_request())
    assert fake.revoked == [7]


async def test_the_grant_fence_is_the_slot_key_create_revision() -> None:
    fake = _FakeEtcd()
    reply = await _store(fake).acquire(_request())
    assert reply.outcome == "lease-granted"
    assert reply.fence == _SLOT_REVISION
    assert fake.revoked == []


async def test_a_replayed_request_is_fenced_by_its_slot_not_its_request_key() -> None:
    fake = _FakeEtcd()
    store = _store(fake)
    request = _request()
    granted = await store.acquire(request)
    held = next(iter(fake.slots.values()))[0]
    replayed = await store.acquire(request)
    assert (
        fake.read_keys == [f"/onex/lease-request/exclusive/r1/{request.request_id}"] * 2
    )
    assert replayed.reason == "replayed request_id"
    assert replayed.lease_id == granted.lease_id
    assert replayed.fence == _SLOT_REVISION
    assert UUID(json.loads(held)["lease_id"]) == granted.lease_id


@pytest.mark.parametrize(
    "failure",
    [
        InfraConnectionError("txn failed"),
        ValueError("invalid txn"),
        asyncio.CancelledError(),
    ],
)
@pytest.mark.parametrize("revoke_fails", [False, True])
async def test_cleanup_preserves_the_original_acquire_failure(
    failure: BaseException, revoke_fails: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeEtcd()
    revoke = AsyncMock(
        side_effect=InfraConnectionError("revoke failed") if revoke_fails else None
    )
    monkeypatch.setattr(fake, "txn", AsyncMock(side_effect=failure))
    monkeypatch.setattr(fake, "lease_revoke", revoke)
    with pytest.raises(type(failure)) as caught:
        await _store(fake).acquire(_request())
    assert caught.value is failure
    revoke.assert_awaited_once_with(7)
    if revoke_fails:
        assert any("InfraConnectionError" in note for note in failure.__notes__)


@pytest.mark.parametrize("revoke_fails", [False, True])
async def test_a_refusal_survives_a_failed_cleanup_of_its_unused_lease(
    revoke_fails: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeEtcd()
    revoke = AsyncMock(
        side_effect=InfraConnectionError("revoke failed") if revoke_fails else None
    )
    monkeypatch.setattr(fake, "txn", AsyncMock(return_value=(False, 0, [])))
    monkeypatch.setattr(fake, "lease_revoke", revoke)
    reply = await _store(fake).acquire(_request())
    assert reply.outcome == "lease-refused"
    revoke.assert_awaited_once_with(7)


@pytest.mark.parametrize("kind", ["a/b", "exclusive/r1", "../x", "Upper"])
def test_a_kind_outside_the_key_alphabet_is_refused(kind: str) -> None:
    with pytest.raises(ValidationError):
        ModelLeaseAcquireRequest(
            kind=kind, resource="r1", holder="runtime-a", request_id=uuid4()
        )
    with pytest.raises(ValidationError):
        ModelLeaseRenewRequest(kind=kind, resource="r1", lease_id=uuid4())
    with pytest.raises(ValidationError):
        ModelLeaseReleaseRequest(kind=kind, resource="r1", lease_id=uuid4())
