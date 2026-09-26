# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17454: each resolved standalone binding connects as the principal it names.

``resolve_standalone_projection_bindings`` returns, per topology binding, a
principal and a DSN read from that binding's carrier. The unit tests check the
mapping; this test checks the claim a runner will rely on, on a real
PostgreSQL: open the DSN a binding returns and ``current_user`` is that
binding's principal. A resolver that paired a principal with the other
binding's carrier would pass every unit test and still connect a tenant writer
as ``omninode_runtime`` -- the exact identity mix-up shape A exists to remove.

The server is the throwaway ``ephemeral_postgres`` cluster (initdb/pg_ctl),
holding only the two login roles the shipped ``local`` topology names.
"""

from __future__ import annotations

from urllib.parse import quote

import psycopg2
import pytest

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.standalone_projection_bindings import (
    StandaloneProjectionBindings,
    resolve_standalone_projection_bindings,
)
from omnibase_infra.topology import load_topology_profile
from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = pytest.mark.integration

_TENANT_ROLE = "tenant_projection_writer"
_INTERNAL_ROLE = "omninode_runtime"


def _table(name: str, schema: str) -> ModelDbTableDeclaration:
    return ModelDbTableDeclaration(
        name=name,
        database_ref="application",
        schema=schema,
        migration=f"proof/{name}.sql",
        access="read_write",
        role=name,
    )


# A writer spanning both domains: a tenant relation, an omninode_internal one,
# and (added by the resolver) the runner's own watermark.
_MIXED_WRITER = (
    _table("delegation_events", "public"),
    _table("generation_events", "omninode_internal"),
)


def _dsn(pg: EphemeralPostgres, role: str, dbname: str) -> str:
    return (
        f"postgresql://{role}@/{dbname}"
        f"?host={quote(pg.socket_dir, safe='')}&port={pg.port}"
    )


@pytest.fixture
def lane(ephemeral_postgres: EphemeralPostgres) -> EphemeralPostgres:
    admin = ephemeral_postgres.connect()
    admin.autocommit = True
    with admin.cursor() as cur:
        for role in (_TENANT_ROLE, _INTERNAL_ROLE):
            cur.execute(f"CREATE ROLE {role} LOGIN")
        cur.execute("CREATE DATABASE omnidash_analytics")
    admin.close()
    return ephemeral_postgres


def _resolve(
    pg: EphemeralPostgres,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tenant_role: str,
    internal_role: str,
) -> StandaloneProjectionBindings:
    monkeypatch.setenv(
        "ONEX_TENANT_DB_URL", _dsn(pg, tenant_role, "omnidash_analytics")
    )
    monkeypatch.setenv(
        "OMNINODE_INTERNAL_DB_URL", _dsn(pg, internal_role, "omnidash_analytics")
    )
    return resolve_standalone_projection_bindings(
        _MIXED_WRITER, load_topology_profile("local")
    )


def _identity_mismatches(resolved: StandaloneProjectionBindings) -> list[str]:
    """Open every resolved DSN; name each binding whose session is someone else."""
    wrong = []
    for ref, binding in sorted(resolved.bindings.items()):
        conn = psycopg2.connect(binding.dsn.get_secret_value())
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT current_user")
                row = cur.fetchone()
        finally:
            conn.close()
        assert row is not None
        if row[0] != binding.principal:
            wrong.append(
                f"binding {ref!r} names principal {binding.principal!r} "
                f"but its DSN connects as {row[0]!r}"
            )
    return wrong


def test_each_resolved_binding_connects_as_its_declared_principal(
    lane: EphemeralPostgres, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved = _resolve(
        lane, monkeypatch, tenant_role=_TENANT_ROLE, internal_role=_INTERNAL_ROLE
    )

    assert set(resolved.bindings) == {"tenant_projection", "omninode_runtime_service"}
    assert _identity_mismatches(resolved) == []
    # And the table map sends each write through the identity that owns its domain.
    principal_for = {
        table: resolved.bindings[resolved.write_binding_for(table)].principal
        for table in resolved.tables
    }
    assert principal_for == {
        "delegation_events": _TENANT_ROLE,
        "generation_events": _INTERNAL_ROLE,
        "projection_watermarks": _INTERNAL_ROLE,
    }


def test_swapped_carriers_are_caught_naming_both_bindings(
    lane: EphemeralPostgres, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Positive control for the check above: carriers pointing at the wrong
    # roles must be reported, binding by binding, so an empty mismatch list
    # there cannot come from a check that never compares anything.
    resolved = _resolve(
        lane, monkeypatch, tenant_role=_INTERNAL_ROLE, internal_role=_TENANT_ROLE
    )

    wrong = _identity_mismatches(resolved)

    assert len(wrong) == 2, wrong
    assert any("'tenant_projection'" in line for line in wrong), wrong
    assert any("'omninode_runtime_service'" in line for line in wrong), wrong
