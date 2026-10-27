# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lost transaction reply and failed probe must not revoke a committed grant."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.lease import HandlerLeaseStore, check_fence
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import (
    EnumLeaseFenceStatus,
    ModelLeaseAcquireRequest,
    ModelLeaseKindPolicy,
)

from .conftest import EtcdProcess

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]


async def test_lost_txn_and_probe_replies_preserve_committed_grant(
    etcd: EtcdProcess, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with httpx.AsyncClient(timeout=5) as client:
        adapter = HandlerPlacementEtcd(etcd.endpoint, client)
        store = HandlerLeaseStore(adapter, [ModelLeaseKindPolicy(kind="worker")])
        request = ModelLeaseAcquireRequest(
            kind="worker", resource="r", holder="a", request_id=uuid4()
        )
        original_post = adapter._post
        lost_reply = InfraConnectionError("transaction reply lost")
        committed = False

        async def lose_replies(
            path: str, body: Mapping[str, object]
        ) -> dict[str, object]:
            nonlocal committed
            if path == "/v3/kv/range" and committed:
                raise InfraConnectionError("confirmation read unavailable")
            result = await original_post(path, body)
            if path == "/v3/kv/txn":
                committed = True
                raise lost_reply
            return result

        monkeypatch.setattr(adapter, "_post", lose_replies)
        with pytest.raises(InfraConnectionError) as raised:
            await store.acquire(request)
        assert raised.value is lost_reply
        monkeypatch.setattr(adapter, "_post", original_post)

        replay = await store.acquire(request)
        assert replay.outcome == "lease-granted"
        assert replay.reason == "replayed request_id"
        assert replay.holder == request.holder
        assert (
            await check_fence(
                adapter,
                kind=request.kind,
                resource=request.resource,
                fence=replay.fence,
            )
            == EnumLeaseFenceStatus.OK
        )
        competitor = await store.acquire(
            request.model_copy(update={"request_id": uuid4(), "holder": "b"})
        )
        assert competitor.outcome == "lease-refused"
        assert len(await store.show(request.kind, request.resource)) == 1
