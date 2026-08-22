# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16239: emitted projection SQL must name the PHYSICAL schema.

``ProjectionTableTarget`` carried the raw contract schema straight into
``_execute_upsert``/``_execute_query``'s schema-qualified SQL while the grant
check ran the same declaration through
:func:`physical_grant_schema_for_table`. For every relation still under the
OMN-15359 physical-relocation bridge the two disagreed: grant validation
passed against ``public`` while the statement that actually executed named
``tenant``/``omninode_internal``.

That is not a latent risk, it is a live one. Verified on the stability-test
lane's ``omnidash_analytics`` (2026-08-19): the database has exactly three
non-system schemas -- ``information_schema``, ``omninode_internal``,
``public`` -- so there is **no ``tenant`` schema at all**, and
``omninode_internal`` holds exactly one relation (``live_events``, the single
family copied out by migration 099) against 71 in ``public``. Every bridged
declaration therefore emitted SQL against a schema that resolves nowhere.

These tests drive the real adapter SQL-emission path -- a genuine
``_execute_upsert``/``_execute_query`` call against a DB-API double whose
cursor records the statement -- not a mock of the resolution itself. The
assertions read the emitted SQL string.

The final test is the degrade-to-no-op proof: once OMN-15359 finishes copying
each family, its table leaves the bridge set,
``physical_grant_schema_for_table`` returns identity, and this fix becomes
invisible. ``live_events`` is already in that post-migration state and is
asserted here as the live control.
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDatabaseTarget,
    _build_projection_db_adapter,
    _resolve_projection_database_target,
)
from omnibase_infra.topology.physical_schema_mapping import (
    INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
    TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
    physical_grant_schema_for_table,
)
from tests.helpers.application_db_topology import (
    application_topology,
    projection_database_target,
    projection_database_urls,
)
from tests.helpers.projection_tenant_authority import verified_tenant_dispatch

pytestmark = pytest.mark.unit


def _connection(principal: str) -> tuple[MagicMock, MagicMock]:
    """Return a DB-API double whose cursor records every executed statement."""
    cursor = MagicMock()
    cursor.fetchone.return_value = (principal, "omnidash_analytics")
    cursor.fetchall.return_value = []
    cursor_context = MagicMock()
    cursor_context.__enter__.return_value = cursor
    conn = MagicMock()
    conn.closed = False
    conn.autocommit = True
    conn.cursor.return_value = cursor_context
    return conn, cursor


def _adapter(
    target: ProjectionDatabaseTarget,
    *,
    tenant_id: UUID | None = None,
) -> object:
    authority, event = (
        verified_tenant_dispatch(tenant_id) if tenant_id is not None else (None, None)
    )
    return _build_projection_db_adapter(
        projection_database_urls(target, "postgresql://fixture"),
        target,
        authority,
        event,
    )


def _emitted_sql(cursor: MagicMock, statement: str) -> str:
    """Return the single recorded statement starting with ``statement``.

    Searched rather than indexed: the adapter emits identity and
    ``set_config`` preamble statements whose count differs per domain, and a
    hardcoded index would silently assert against the wrong call.
    """
    matches = [
        call.args[0]
        for call in cursor.execute.call_args_list
        if isinstance(call.args[0], str) and call.args[0].startswith(statement)
    ]
    assert len(matches) == 1, (
        f"expected exactly one {statement!r} statement, got {len(matches)}: "
        f"{[call.args[0] for call in cursor.execute.call_args_list]!r}"
    )
    return matches[0]


def _declaration(
    name: str,
    schema: str,
    access: Literal["read", "write", "read_write"],
) -> ModelDbTableDeclaration:
    return ModelDbTableDeclaration(
        name=name,
        database_ref="application",
        schema=schema,
        migration=f"docker/migrations/forward/nodes/{name}.sql",
        access=access,
        role=name,
    )


def test_bridged_tenant_table_is_still_declared_bridged() -> None:
    """Guard the premise: these fixtures must actually be under the bridge.

    If OMN-15359 relocates these families the mapping returns identity and the
    SQL assertions below would pass vacuously against the declared schema.
    Fail loudly here instead, so the tests get re-pointed rather than silently
    stopping to prove anything.
    """
    assert "delegation_events" in TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359
    assert "generation_events" in INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359
    assert physical_grant_schema_for_table("tenant", "delegation_events") == "public"
    assert (
        physical_grant_schema_for_table("omninode_internal", "generation_events")
        == "public"
    )


def test_upsert_sql_names_the_physical_schema_for_a_bridged_tenant_table() -> None:
    """The INSERT must target ``public``, the schema that physically holds it.

    Pre-fix this emitted ``INSERT INTO "tenant"."delegation_events"`` against a
    database with no ``tenant`` schema whatsoever.
    """
    tenant_id = uuid4()
    target = projection_database_target("delegation_events", schema="tenant")
    conn, cursor = _connection("tenant_projection_writer")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target, tenant_id=tenant_id)
        assert adapter.upsert(  # type: ignore[attr-defined]
            "delegation_events",
            "event_id",
            {"event_id": uuid4(), "tenant_id": str(tenant_id)},
        )

    insert_sql = _emitted_sql(cursor, "INSERT INTO")
    assert insert_sql.startswith('INSERT INTO "public"."delegation_events"')
    assert '"tenant"."delegation_events"' not in insert_sql


def test_upsert_sql_names_the_physical_schema_for_a_bridged_internal_table() -> None:
    """Same seam on the OMNINODE_INTERNAL domain, which needs no tenant authority."""
    target = projection_database_target("generation_events", schema="omninode_internal")
    conn, cursor = _connection("omninode_runtime")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target)
        assert adapter.upsert("generation_events", "event_id", {"event_id": uuid4()})  # type: ignore[attr-defined]

    insert_sql = _emitted_sql(cursor, "INSERT INTO")
    assert insert_sql.startswith('INSERT INTO "public"."generation_events"')
    assert '"omninode_internal"."generation_events"' not in insert_sql


def test_query_sql_names_the_physical_schema_for_a_bridged_internal_table() -> None:
    """The read path emits its own schema-qualified SQL and must agree."""
    target = projection_database_target(
        "generation_events", schema="omninode_internal", access="read"
    )
    conn, cursor = _connection("omninode_runtime")

    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target)
        assert adapter.query("generation_events") == []  # type: ignore[attr-defined]

    select_sql = _emitted_sql(cursor, "SELECT * FROM")
    assert select_sql.startswith('SELECT * FROM "public"."generation_events"')
    assert '"omninode_internal"."generation_events"' not in select_sql


def test_grant_check_and_emitted_sql_resolve_to_one_schema() -> None:
    """The seam itself: one resolution feeds grant validation and SQL alike.

    A single resolved value on the target is what makes divergence structurally
    impossible, rather than two call sites that merely happen to agree today.
    """
    for schema, name in (
        ("tenant", "delegation_events"),
        ("omninode_internal", "generation_events"),
    ):
        target = projection_database_target(name, schema=schema)
        table_target = target.table_targets[0]

        assert table_target.physical_schema == physical_grant_schema_for_table(
            schema, name
        )
        assert table_target.physical_schema == "public"
        assert table_target.table.schema == schema, (
            "the declaration itself must stay logical -- domain resolution and "
            "operator-facing contract errors depend on it"
        )


def test_unbridged_table_keeps_its_declared_schema_so_the_fix_degrades_to_a_noop() -> (
    None
):
    """``live_events`` already completed its OMN-15359 copy (migration 099).

    It is therefore absent from the bridge set, the mapping returns identity,
    and the emitted SQL must still name ``omninode_internal`` -- proving this
    fix vanishes once every family is relocated, rather than pinning writes to
    ``public`` forever.
    """
    assert "live_events" not in INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359

    declaration = _declaration("live_events", "omninode_internal", "write")
    target = _resolve_projection_database_target((declaration,), application_topology())
    assert target.table_targets[0].physical_schema == "omninode_internal"

    conn, cursor = _connection("omninode_runtime")
    with patch("psycopg2.connect", return_value=conn):
        adapter = _adapter(target)
        assert adapter.upsert("live_events", "event_id", {"event_id": uuid4()})  # type: ignore[attr-defined]

    insert_sql = _emitted_sql(cursor, "INSERT INTO")
    assert insert_sql.startswith('INSERT INTO "omninode_internal"."live_events"')
