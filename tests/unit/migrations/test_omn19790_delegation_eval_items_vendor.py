# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19790 vendor identity for the delegation eval projection migrations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_delegation_eval"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CREATE = "0000_create_delegation_eval_items.sql"
_GRANT = "0001_grant_tenant_projection_writer_delegation_eval_items.sql"
_SHA256 = {
    _CREATE: "cc2f88c63762bc818e60f558fa7a91a0c7132b5d3c67700db5b354ed63782246",
    _GRANT: "d363bc416cf0ca25062242a6b0343818e2d7e4c442167b3e0198dd71743c2688",
}
_TABLE = "delegation_eval_items"
_CURSOR_SEQUENCE = "delegation_eval_items_projection_cursor_seq"


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in (_VENDOR / filename).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
def test_vendor_bytes_and_manifest_binding_are_exact(filename: str) -> None:
    artifact_path = f"nodes/{_NODE}/{filename}"
    assert (
        hashlib.sha256((_VENDOR / filename).read_bytes()).hexdigest()
        == _SHA256[filename]
    )
    rows = {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }
    assert rows[artifact_path] == [
        artifact_path,
        f"node:{_NODE}",
        f"node:{_NODE}",
        "tenant",
        f"node:{_NODE}:{filename}",
        _SHA256[filename],
    ]


def test_migration_classes_match_the_classifier() -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_CREATE}"] == "forward-only"
    assert classes[f"forward/nodes/{_NODE}/{_GRANT}"] == "expand-only"


def test_table_rls_and_exact_grants_are_present() -> None:
    create = _statements(_CREATE)
    grants = _statements(_GRANT)
    assert re.search(rf"CREATE TABLE IF NOT EXISTS public\.{_TABLE}\s*\(", create)
    assert "projection_cursor BIGSERIAL" in create
    assert f"ALTER TABLE public.{_TABLE} ENABLE ROW LEVEL SECURITY;" in create
    assert f"ALTER TABLE public.{_TABLE} FORCE ROW LEVEL SECURITY;" in create
    assert re.search(
        rf"CREATE POLICY tenant_isolation ON public\.{_TABLE}\s+"
        r"FOR ALL\s+USING \(tenant_id = current_setting\('app.tenant_id', true\)::uuid\)\s+"
        r"WITH CHECK \(tenant_id = current_setting\('app.tenant_id', true\)::uuid\);",
        create,
    )
    assert f"GRANT SELECT ON public.{_TABLE} TO app_dashboard;" in create
    assert re.search(
        rf"GRANT\s+SELECT,\s*INSERT,\s*UPDATE\s+"
        rf"ON public\.{_TABLE}\s+TO tenant_projection_writer;",
        grants,
    )
    assert re.search(
        rf"GRANT\s+USAGE\s+ON SEQUENCE public\.{_CURSOR_SEQUENCE}\s+"
        r"TO tenant_projection_writer;",
        grants,
    )
    assert "GRANT USAGE ON SCHEMA public TO tenant_projection_writer;" in grants
    assert "has_sequence_privilege(" in grants
    assert "'tenant_projection_writer'," in grants
    for sql in (create, grants):
        assert not re.search(
            r"GRANT\s+[^;]*\bDELETE\b[^;]*\bTO\s+tenant_projection_writer\b",
            sql,
            re.IGNORECASE,
        )


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
@pytest.mark.parametrize(
    "profile", ["local", "onex-dev", "onex-prod", "stability-test"]
)
def test_migrations_pass_the_application_database_sql_gate(
    filename: str, profile: str
) -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / filename).read_text(encoding="utf-8")
    violations = lint_application_database_sql(sql, load_topology_profile(profile))
    assert violations == (), f"{profile}/{filename}: {violations}"


def test_the_sql_gate_is_live_positive_control() -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / _CREATE).read_text(encoding="utf-8")
    broken = sql.replace("public.delegation_eval_items", "tenant.delegation_eval_items")
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()


@pytest.mark.parametrize("profile", ["local", "onex-dev", "onex-prod"])
def test_bridge_derives_the_shipped_table_grant(profile: str) -> None:
    from omnibase_core.enums.enum_database_grant_object_type import (
        EnumDatabaseGrantObjectType,
    )
    from omnibase_infra.topology import load_topology_profile
    from omnibase_infra.topology.table_grant_derivation import (
        LEGACY_MIGRATION_TABLE_DECLARATIONS,
        derive_table_grants,
    )

    bridges = tuple(
        entry
        for entry in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if entry.table.name == _TABLE
    )
    assert len(bridges) == 1
    bridge = bridges[0]
    assert bridge.table.access == "read_write"
    assert bridge.table.schema == "public"
    assert bridge.table.database_ref == "application"
    assert bridge.contract_path == (_VENDOR / _CREATE).relative_to(_ROOT)
    derived = derive_table_grants(load_topology_profile(profile), bridges)
    matching = [
        (principal, grant)
        for principal, grants in derived.grants.items()
        for grant in grants
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and grant.schema == "public"
        and _TABLE in grant.objects
    ]
    assert len(matching) == 1
    principal, grant = matching[0]
    assert principal == "tenant_projection_writer"
    assert {privilege.value for privilege in grant.privileges} == {
        "SELECT",
        "INSERT",
        "UPDATE",
    }
    instance = yaml.safe_load(
        (_ROOT / "src/omnibase_infra/topology/instances" / f"{profile}.yaml").read_text(
            encoding="utf-8"
        )
    )
    shipped = [
        entry
        for entry in instance["databases"]["application"]["principals"][principal][
            "grants"
        ]
        if entry["object_type"] == "TABLE"
        and entry["schema"] == "public"
        and _TABLE in entry["objects"]
    ]
    assert len(shipped) == 1
    assert set(shipped[0]["privileges"]) == {"SELECT", "INSERT", "UPDATE"}
