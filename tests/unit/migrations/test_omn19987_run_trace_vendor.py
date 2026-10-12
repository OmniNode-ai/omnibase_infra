# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19987 vendor identity for the run-trace projection migrations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_run_trace"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CREATE = "0000_create_run_trace.sql"
_GRANT = "0001_grant_tenant_projection_writer_run_trace.sql"
_FORCE_RLS = "0002_force_rls_run_trace.sql"
_SHA256 = {
    _CREATE: "ede5d70d8eb0a321943c48d544f0db3856975551e9cfaaba7473a911244220bf",
    _GRANT: "70377e2c384a00ad8168f379df1668a8a6435515387dc19e96768e2e9a358778",
    _FORCE_RLS: "9ee20a03252c1a9b77fe64fa8c6fb57664f4ac717dbf41de687f7186c1b5d43c",
}
_TABLE = "run_trace"
_CURSOR_SEQUENCE = "run_trace_projection_cursor_seq"
# Every topology instance this repo ships, read from the tree so a new instance
# is covered without editing this file.
_SHIPPED_PROFILES = sorted(
    path.stem
    for path in (_ROOT / "src" / "omnibase_infra" / "topology" / "instances").glob(
        "*.yaml"
    )
)


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in (_VENDOR / filename).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


@pytest.mark.parametrize("filename", [_CREATE, _GRANT, _FORCE_RLS])
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
    assert classes[f"forward/nodes/{_NODE}/{_FORCE_RLS}"] == "forward-only"


def test_table_rls_and_exact_grants_are_present() -> None:
    create = _statements(_CREATE)
    grants = _statements(_GRANT)
    force = _statements(_FORCE_RLS)
    assert re.search(rf"CREATE TABLE IF NOT EXISTS public\.{_TABLE}\s*\(", create)
    assert "projection_cursor BIGSERIAL" in create
    assert f"ALTER TABLE public.{_TABLE} ENABLE ROW LEVEL SECURITY;" in create
    assert f"ALTER TABLE public.{_TABLE} FORCE ROW LEVEL SECURITY;" not in create
    assert f"ALTER TABLE public.{_TABLE} FORCE ROW LEVEL SECURITY;" in force
    assert re.search(
        rf"CREATE POLICY tenant_isolation ON public\.{_TABLE}\s+"
        r"FOR ALL\s+USING \(tenant_id = current_setting\('app.tenant_id', true\)\)\s+"
        r"WITH CHECK \(tenant_id = current_setting\('app.tenant_id', true\)\);",
        create,
    )
    assert "PRIMARY KEY (tenant_id, correlation_id, step_sequence)" in create
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


@pytest.mark.parametrize("filename", [_CREATE, _GRANT, _FORCE_RLS])
@pytest.mark.parametrize("profile", _SHIPPED_PROFILES)
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


def test_the_shipped_profiles_are_found() -> None:
    """The parametrised gate above ran on something."""
    assert len(_SHIPPED_PROFILES) >= 3


def test_the_sql_gate_is_live_positive_control() -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / _CREATE).read_text(encoding="utf-8")
    broken = sql.replace("public.run_trace", "tenant.run_trace")
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()
