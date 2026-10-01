# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19793 vendor identity for the delegation eval-run projection migrations."""

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
_VERDICTS = "0003_create_delegation_eval_item_verdicts.sql"
_RESULTS = "0004_create_delegation_eval_results.sql"
_GRANT = "0005_grant_tenant_projection_writer_delegation_eval_run_tables.sql"
_FORCE_RLS = "0006_force_rls_delegation_eval_run_tables.sql"
_SHA256 = {
    _VERDICTS: "9cf639b139b772caa20b69b2920dc5302563c41643f96d16eff5c04009f343cd",
    _RESULTS: "9e10ff42d43e3f4964a2bfb243f0dd27580821f1ea07f5209cc4994ea38996cb",
    _GRANT: "6528f69377939f1ddbad4dbff4b0049665df7f944aa4d7c006d72bcdc717f12a",
    _FORCE_RLS: "393aa4e4bee0ee70d59b22bbbf2a42f49611357619179b2aef3036c3a95c3031",
}
_CREATES = {
    _VERDICTS: "delegation_eval_item_verdicts",
    _RESULTS: "delegation_eval_results",
}


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in (_VENDOR / filename).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


@pytest.mark.parametrize("filename", list(_SHA256))
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
    assert classes[f"forward/nodes/{_NODE}/{_VERDICTS}"] == "forward-only"
    assert classes[f"forward/nodes/{_NODE}/{_RESULTS}"] == "forward-only"
    assert classes[f"forward/nodes/{_NODE}/{_GRANT}"] == "expand-only"
    assert classes[f"forward/nodes/{_NODE}/{_FORCE_RLS}"] == "forward-only"


@pytest.mark.parametrize(("filename", "table"), list(_CREATES.items()))
def test_create_enables_rls_and_defers_force_to_the_fenced_migration(
    filename: str, table: str
) -> None:
    create = _statements(filename)
    force = _statements(_FORCE_RLS)
    assert re.search(rf"CREATE TABLE IF NOT EXISTS public\.{table}\s*\(", create)
    assert "projection_cursor BIGSERIAL" in create
    assert f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY;" in create
    assert f"ALTER TABLE public.{table} FORCE ROW LEVEL SECURITY;" not in create
    assert f"ALTER TABLE public.{table} FORCE ROW LEVEL SECURITY;" in force
    policy = (
        rf"CREATE POLICY tenant_isolation ON public\.{table}\s+"
        r"FOR ALL\s+USING \(tenant_id = current_setting\('app.tenant_id', true\)::uuid\)\s+"
        r"WITH CHECK \(tenant_id = current_setting\('app.tenant_id', true\)::uuid\);"
    )
    assert re.search(policy, create)
    assert re.search(policy, force)
    assert f"GRANT SELECT ON public.{table} TO app_dashboard;" in create


@pytest.mark.parametrize("table", list(_CREATES.values()))
def test_grant_is_exact_and_never_delete(table: str) -> None:
    grants = _statements(_GRANT)
    assert re.search(
        rf"GRANT\s+SELECT,\s*INSERT,\s*UPDATE\s+ON public\.{table}\s+"
        r"TO tenant_projection_writer;",
        grants,
    )
    assert re.search(
        rf"GRANT\s+USAGE\s+ON SEQUENCE public\.{table}_projection_cursor_seq\s+"
        r"TO tenant_projection_writer;",
        grants,
    )
    assert "GRANT USAGE ON SCHEMA public TO tenant_projection_writer;" in grants
    assert not re.search(
        r"GRANT\s+[^;]*\bDELETE\b[^;]*\bTO\s+tenant_projection_writer\b",
        grants,
        re.IGNORECASE,
    )


@pytest.mark.parametrize("filename", list(_SHA256))
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
