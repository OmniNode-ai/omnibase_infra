# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20242 vendor identity and grants for the disposition projection."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_delegation_disposition"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CREATE = "0000_create_delegation_dispositions.sql"
_GRANT = "0001_grant_tenant_projection_writer_delegation_dispositions.sql"
_FORCE_RLS = "0002_force_rls_delegation_dispositions.sql"
_SHA256 = {
    _CREATE: "3d180c385033e2ab29dbafdf84798c47b2274bbfd607fa828235a19801a73ebb",
    _GRANT: "24d3fe38a9ad8d1efcebd0d745c2242fa65000daebe488208590bc41bec9e149",
    _FORCE_RLS: "0c8f29714eb1441be4ab49e45d0ecade4ce7f2da441552708c346f859dedfb03",
}
_EXPECTED_CLASSES = {
    _CREATE: "forward-only",
    _GRANT: "expand-only",
    _FORCE_RLS: "forward-only",
}
_TABLE = "delegation_dispositions"


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in (_VENDOR / filename).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


@pytest.mark.parametrize("filename", _SHA256)
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
    assert {path.name for path in _VENDOR.glob("*.sql")} == set(_SHA256)
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    for filename, expected in _EXPECTED_CLASSES.items():
        assert classes[f"forward/nodes/{_NODE}/{filename}"] == expected


def test_only_the_force_rls_step_is_fenced() -> None:
    fence = yaml.safe_load(
        (_FORWARD / "fenced-node-migrations.yaml").read_text(encoding="utf-8")
    )["fenced_node_migrations"]
    entries = [entry for entry in fence if entry["id"].startswith(f"node:{_NODE}:")]
    assert entries == [{"id": f"node:{_NODE}:{_FORCE_RLS}", "ticket": "OMN-20242"}]


def test_table_rls_and_exact_grants_are_present() -> None:
    create = _statements(_CREATE)
    grants = _statements(_GRANT)
    force = _statements(_FORCE_RLS)
    assert re.search(rf"CREATE TABLE IF NOT EXISTS public\.{_TABLE}\s*\(", create)
    assert "PRIMARY KEY (tenant_id, delegation_correlation_id)" in create
    assert f"ALTER TABLE public.{_TABLE} ENABLE ROW LEVEL SECURITY;" in create
    assert f"ALTER TABLE public.{_TABLE} FORCE ROW LEVEL SECURITY;" not in create
    assert f"ALTER TABLE public.{_TABLE} FORCE ROW LEVEL SECURITY;" in force
    for sql in (create, force):
        assert re.search(
            rf"CREATE POLICY tenant_isolation ON public\.{_TABLE}\s+"
            r"FOR ALL\s+USING \(tenant_id = current_setting\('app.tenant_id', true\)::uuid\)\s+"
            r"WITH CHECK \(tenant_id = current_setting\('app.tenant_id', true\)::uuid\);",
            sql,
        )
        assert f"GRANT SELECT ON public.{_TABLE} TO app_dashboard;" in sql
    assert re.search(
        rf"GRANT\s+SELECT,\s*INSERT,\s*UPDATE\s+"
        rf"ON public\.{_TABLE}\s+TO tenant_projection_writer;",
        grants,
    )
    assert "GRANT USAGE ON SCHEMA public TO tenant_projection_writer;" in grants
    assert not re.search(r"\b(?:BIGSERIAL|SERIAL|CREATE SEQUENCE)\b", create)
    assert "ON SEQUENCE" not in grants
    for sql in (create, grants, force):
        assert not re.search(
            r"GRANT\s+[^;]*\bDELETE\b[^;]*\bTO\s+tenant_projection_writer\b",
            sql,
            re.IGNORECASE,
        )


@pytest.mark.parametrize("filename", _SHA256)
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


@pytest.mark.parametrize("filename", [_CREATE])
def test_the_sql_gate_is_live_positive_control(filename: str) -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / filename).read_text(encoding="utf-8")
    broken = sql.replace(f"public.{_TABLE}", f"tenant.{_TABLE}")
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()
