# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A revoke of a lease etcd already expired is a revoke, not a transport error (OMN-21428).

A refused acquire revokes the etcd lease it granted for the attempt. Under load
that lease can expire first; etcd then answers the revoke with 404 (lease not
found), and the acquire must still answer instead of raising.
"""

from __future__ import annotations

import httpx
import pytest

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.placement import HandlerPlacementEtcd

pytestmark = pytest.mark.unit


def _adapter(status: int) -> HandlerPlacementEtcd:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v3/lease/revoke"
        return httpx.Response(status, json={"error": "requested lease not found"})

    return HandlerPlacementEtcd(
        "http://etcd.invalid:2379",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_revoke_of_an_expired_lease_returns() -> None:
    adapter = _adapter(404)
    try:
        await adapter.lease_revoke(7)
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("status", [400, 500, 503])
async def test_revoke_other_errors_still_raise(status: int) -> None:
    adapter = _adapter(status)
    try:
        with pytest.raises(InfraConnectionError):
            await adapter.lease_revoke(7)
    finally:
        await adapter.aclose()
