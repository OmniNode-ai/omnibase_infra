# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ``StoreGatewaySessionMemory.put_if_present`` (OMN-15918 R3).

``put_if_present`` is the atomicity primitive that closes the heartbeat
resurrection race: an unconditional ``put`` after a read-then-await gap can
silently recreate a session a concurrent detach just removed.
``test_handlers.py::test_heartbeat_does_not_resurrect_concurrently_detached_session``
covers the full handler-level proof; this file covers the store primitive in
isolation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_status import (
    EnumGatewaySessionStatus,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.store_gateway_session_memory import (
    StoreGatewaySessionMemory,
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _session(**overrides: object) -> ModelGatewaySession:
    now = datetime.now(UTC)
    base: dict[str, object] = {
        "session_id": uuid4(),
        "tenant_id": TENANT_ID,
        "tenant_slug": "acme",
        "principal_id": "t-acme",
        "keycloak_client_id": "gw-tenant-acme",
        "edge_instance_id": "edge-201",
        "status": EnumGatewaySessionStatus.ACTIVE,
        "attached_at": now,
        "last_heartbeat_at": now,
        "expires_at": now + timedelta(hours=1),
    }
    base.update(overrides)
    return ModelGatewaySession(**base)  # type: ignore[arg-type]


async def test_put_if_present_overwrites_an_existing_session() -> None:
    store = StoreGatewaySessionMemory()
    session = _session()
    await store.put(session)

    refreshed = session.model_copy(update={"status": EnumGatewaySessionStatus.DEGRADED})
    result = await store.put_if_present(refreshed)

    assert result is True
    stored = await store.get(session.session_id)
    assert stored is not None
    assert stored.status is EnumGatewaySessionStatus.DEGRADED


async def test_put_if_present_returns_false_and_does_not_resurrect_absent_session() -> (
    None
):
    """The exact bug OMN-15918 R3 closes: a session absent from the store
    (e.g. concurrently detached) must never be recreated by a stale write."""
    store = StoreGatewaySessionMemory()
    session = _session()
    # Deliberately never put() -- session_id has never existed in the store,
    # simulating "detached between another handler's read and this write."

    result = await store.put_if_present(session)

    assert result is False
    assert await store.get(session.session_id) is None
    assert store._sessions == {}


async def test_put_if_present_after_delete_does_not_resurrect() -> None:
    store = StoreGatewaySessionMemory()
    session = _session()
    await store.put(session)
    await store.delete(session.session_id)

    result = await store.put_if_present(session)

    assert result is False
    assert await store.get(session.session_id) is None
