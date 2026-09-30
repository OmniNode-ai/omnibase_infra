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
_FORCE_RLS = "0002_force_rls_delegation_eval_items.sql"
_SHA256 = {
    _CREATE: "133b6b8746404b36069778132e13c0f01f1edb053024c3cae4c9765ab368dad2",
    _GRANT: "d363bc416cf0ca25062242a6b0343818e2d7e4c442167b3e0198dd71743c2688",
    _FORCE_RLS: "a317f7363fe8abbf0dbd380598fa6d694534b1de8998fc38c70d75d208ec2eca",
}
_TABLE = "delegation_eval_items"
_CURSOR_SEQUENCE = "delegation_eval_items_projection_cursor_seq"


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


@pytest.mark.parametrize("filename", [_CREATE, _GRANT, _FORCE_RLS])
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
