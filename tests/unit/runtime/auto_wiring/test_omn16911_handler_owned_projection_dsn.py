# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A handler-owned projection pool must use the topology-resolved DSN [OMN-16911].

``ConsumerFlowProjectionWriter`` declares ``db_io.db_tables`` in the
``omninode_internal`` domain, so the runtime resolves the
``omninode_runtime_service`` binding for it and proves that binding's principal
(``omninode_runtime``) holds USAGE on the schema plus the table privileges the
declared access needs. That proof then bound nothing: the handler opens its own
asyncpg pool in ``__init__`` from an omnimarket-side settings fallback that
prefers ``OMNIDASH_ANALYTICS_DB_URL`` — the dashboard-facing
``tenant_projection_writer`` identity, which has no USAGE on
``omninode_internal``. Every statement the writer issued on the .201 dev lane
died with ``InsufficientPrivilegeError: permission denied for schema
omninode_internal``, DLQ'ing ~6 heartbeats/min while
``consumer_flow_windows`` held 0 rows.

The runtime now hands a handler that owns its adapter the same DSN it resolved
and proved, and refuses to wire one that offers no seam to hand it to. A
role/grant mismatch is a wiring-time failure, not a per-heartbeat one.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    PROJECTION_INPROCESS_DISPATCH_ATTR,
    PROJECTION_OWNED_DATABASE_BIND_ATTR,
    _make_projection_dispatch_callback,
)
from tests.helpers.application_db_topology import projection_database_target

_INTERNAL_DSN = "postgresql://omninode_runtime:pw@host:5432/omnidash_analytics"
_ANALYTICS_DSN = "postgresql://role_omnidash:pw@host:5432/omnidash_analytics"
# OMN-15425: the `tenant_projection` binding is a THIRD login role on the same
# physical database, resolved from its own DSN env. It used to share
# OMNIDASH_ANALYTICS_DB_URL with `app_dashboard`, which is unsatisfiable under
# this ticket's own per-binding `current_user` attestation.
_TENANT_DSN = "postgresql://tenant_projection_writer:pw@host:5432/omnidash_analytics"


@pytest.fixture(autouse=True)
def _distinct_binding_dsns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each topology binding a DSN that names its own login role.

    The two DSNs differ by role on purpose: this defect is invisible when both
    binding envs carry the same string, which is how the wiring tests that
    predate it kept passing while the lane denied every write.
    """
    monkeypatch.setenv("OMNINODE_INTERNAL_DB_URL", _INTERNAL_DSN)
    monkeypatch.setenv("OMNIDASH_ANALYTICS_DB_URL", _ANALYTICS_DSN)
    monkeypatch.setenv("ONEX_TENANT_DB_URL", _TENANT_DSN)


class _OwnedAdapter:
    """Minimal stand-in for the handler-owned async adapter (connect/close)."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def connect(self) -> None:  # pragma: no cover - shape only
        return None

    async def close(self) -> None:  # pragma: no cover - shape only
        return None


class _OwnedDatabaseWriter:
    """Runner-shaped handler dispatched in-process that owns its own pool."""

    onex_runtime_inprocess_dispatch = True

    def __init__(self, dsn: str = _ANALYTICS_DSN) -> None:
        self.db = _OwnedAdapter(dsn)
        self.topics = ["onex.evt.platform.node-heartbeat.v1"]
        self.bound: list[str] = []

    def bind_projection_database_url(self, dsn: str) -> None:
        self.bound.append(dsn)
        self.db.dsn = dsn

    async def project_event(self, *args: object, **kwargs: object) -> bool:
        return True  # pragma: no cover - shape only

    def run(self) -> None:  # pragma: no cover - shape only
        return None

    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        return {"rows_upserted": 1}


class _OwnedDatabaseWriterWithoutSeam(_OwnedDatabaseWriter):
    """Same shape, but offers the runtime no way to hand it the resolved DSN."""

    bind_projection_database_url = None  # type: ignore[assignment]


class _StandaloneRunner(_OwnedDatabaseWriter):
    """Runner shape WITHOUT the in-process declaration: the runtime skips it."""

    onex_runtime_inprocess_dispatch = False


def _internal_target() -> object:
    """The real consumer-flow declaration: both tables in omninode_internal."""
    return projection_database_target(
        "consumer_flow_windows",
        "topic_produce_windows",
        schema="omninode_internal",
    )


@pytest.mark.unit
def test_handler_owned_pool_is_bound_to_the_internal_binding_dsn() -> None:
    """AC5/AC1: the owned pool takes the DSN the runtime resolved and proved.

    Falsified if the handler keeps the analytics DSN — that is the live defect,
    byte-for-byte: ``role_omnidash`` has no USAGE on ``omninode_internal``.
    """
    handler = _OwnedDatabaseWriter()
    _make_projection_dispatch_callback(
        handler,
        _internal_target(),
        ("onex.evt.platform.node-heartbeat.v1",),
    )
    assert handler.bound == [_INTERNAL_DSN]
    assert handler.db.dsn == _INTERNAL_DSN
    assert _ANALYTICS_DSN not in handler.bound


@pytest.mark.unit
def test_the_bound_dsn_names_the_principal_the_topology_proved() -> None:
    """The DSN handed over is the resolved binding's, not a second lookup."""
    target = _internal_target()
    bindings = target.bindings  # type: ignore[attr-defined]
    assert [binding.binding_ref for binding in bindings] == ["omninode_runtime_service"]
    assert bindings[0].principal == "omninode_runtime"
    assert bindings[0].dsn_env == "OMNINODE_INTERNAL_DB_URL"

    handler = _OwnedDatabaseWriter()
    _make_projection_dispatch_callback(
        handler, target, ("onex.evt.platform.node-heartbeat.v1",)
    )
    assert handler.bound == [_INTERNAL_DSN]


@pytest.mark.unit
def test_owning_a_pool_without_a_bind_seam_fails_wiring_not_a_heartbeat() -> None:
    """AC5: the mismatch must fail a gate. Wiring refuses, loudly and by name."""
    handler = _OwnedDatabaseWriterWithoutSeam()
    with pytest.raises(ValueError, match=PROJECTION_OWNED_DATABASE_BIND_ATTR):
        _make_projection_dispatch_callback(
            handler,
            _internal_target(),
            ("onex.evt.platform.node-heartbeat.v1",),
        )


@pytest.mark.unit
def test_one_owned_pool_cannot_serve_two_workload_identities() -> None:
    """A single pool is a single login role; two bindings must fail closed."""
    target = projection_database_target(
        "consumer_flow_windows", schema="omninode_internal"
    )
    mixed = projection_database_target("agent_routing_decisions", schema="tenant")
    combined = target.__class__(  # type: ignore[attr-defined]
        tables=target.tables + mixed.tables,  # type: ignore[attr-defined]
        table_targets=target.table_targets + mixed.table_targets,  # type: ignore[attr-defined]
        physical_database=target.physical_database,  # type: ignore[attr-defined]
    )
    handler = _OwnedDatabaseWriter()
    with pytest.raises(ValueError, match="single workload identity"):
        _make_projection_dispatch_callback(
            handler, combined, ("onex.evt.platform.node-heartbeat.v1",)
        )


@pytest.mark.unit
def test_a_standalone_runner_is_left_alone() -> None:
    """OMN-16874 branch is untouched: the runtime never dispatches these."""
    handler = _StandaloneRunner()
    callback = _make_projection_dispatch_callback(
        handler,
        _internal_target(),
        ("onex.evt.platform.node-heartbeat.v1",),
    )
    assert handler.bound == []
    envelope = MagicMock()
    envelope.topic = "onex.evt.platform.node-heartbeat.v1"
    assert asyncio.run(callback(envelope)) is None


@pytest.mark.unit
def test_an_injection_only_handler_needs_no_bind_seam() -> None:
    """A handler with no adapter of its own is unaffected by this rule."""

    class _InjectionOnly:
        def handle(self, input_data: dict[str, object]) -> dict[str, object]:
            return {"rows_upserted": 1}

    _make_projection_dispatch_callback(
        _InjectionOnly(),
        _internal_target(),
        ("onex.evt.platform.node-heartbeat.v1",),
    )


@pytest.mark.unit
def test_the_dispatch_and_bind_attributes_are_declared_names() -> None:
    """Both seams are declared constants the handler side can spell exactly."""
    assert PROJECTION_INPROCESS_DISPATCH_ATTR == "onex_runtime_inprocess_dispatch"
    assert PROJECTION_OWNED_DATABASE_BIND_ATTR == "bind_projection_database_url"
