# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20381 vendor identity for the tenant-credentials 0008 migration.

omnimarket's node_projection_tenant_credentials gains 0008 (a nullable
workspace_id on each credential row, omnimarket#4171). The lane
runners apply only what is vendored here, and omnimarket's Sibling Release
Compatibility Gate compares each file's git blob against the pinned
omnibase_infra release, so the files must be the omnimarket bytes exactly,
bound in the application-migration ledger and classified.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_tenant_credentials"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CREATE = "0000_create_tenant_inference_credentials.sql"

# filename -> (git blob, sha256, column, type). The git blob is what omnimarket's
# Sibling Release Compatibility Gate compares (`git hash-object` of the source
# file against `git ls-tree` of the vendored copy); it is the blob of
# src/omnimarket/nodes/node_projection_tenant_credentials/migrations/<file> at
# omnimarket#4171 head 68e65a02c4dd. Byte-identical: a change on either side
# must move both.
_VENDORED = {
    "0008_tenant_credentials_workspace_id.sql": (
        "e5b9044b52256b26aad4363a2f91c5a977146721",
        "688d94ce82f950d757a5658dfeac55d018c5d70424b91585cb3018eaea2dc79e",
        "workspace_id",
        "TEXT",
    ),
}


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in (_VENDOR / filename).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


def _manifest_rows() -> dict[str, list[str]]:
    return {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }


@pytest.mark.parametrize("filename", sorted(_VENDORED))
def test_vendor_bytes_match_the_omnimarket_blob(filename: str) -> None:
    blob, sha256, _, _ = _VENDORED[filename]
    path = _VENDOR / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256
    data = path.read_bytes()
    # The git blob id, computed the way `git hash-object --no-filters` does.
    git_blob = hashlib.sha1(
        b"blob %d\0" % len(data) + data, usedforsecurity=False
    ).hexdigest()
    assert git_blob == blob


@pytest.mark.parametrize("filename", sorted(_VENDORED))
def test_manifest_binding_is_exact(filename: str) -> None:
    _, sha256, _, _ = _VENDORED[filename]
    artifact_path = f"nodes/{_NODE}/{filename}"
    assert _manifest_rows()[artifact_path] == [
        artifact_path,
        f"node:{_NODE}",
        f"node:{_NODE}",
        "tenant",
        f"node:{_NODE}:{filename}",
        sha256,
    ]


@pytest.mark.parametrize("filename", sorted(_VENDORED))
def test_migration_class_is_declared_expand_only(filename: str) -> None:
    """One nullable ADD COLUMN is additive, so declaring forward-only would make
    the file a rollback barrier for no reason."""
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{filename}"] == "expand-only"


@pytest.mark.parametrize("filename", sorted(_VENDORED))
def test_it_only_adds_one_nullable_column(filename: str) -> None:
    """Nullable with no default and no backfill: rows registered before the
    migration read NULL and nothing is rewritten, so it applies on every lane
    without a fence."""
    _, _, column, column_type = _VENDORED[filename]
    statements = _statements(filename)
    assert re.findall(
        r"ALTER TABLE (\w+) ADD COLUMN IF NOT EXISTS (\w+) (\w+);", statements
    ) == [("tenant_inference_credentials", column, column_type)]
    assert len([s for s in statements.split(";") if s.strip()]) == 1
    upper = statements.upper()
    for forbidden in ("NOT NULL", "DEFAULT", "DROP", "UPDATE", "DELETE", "TRUNCATE"):
        assert forbidden not in upper
    fenced = yaml.safe_load(
        (_FORWARD / "fenced-node-migrations.yaml").read_text(encoding="utf-8")
    )
    assert f"node:{_NODE}:{filename}" not in {
        entry["id"] for entry in fenced["fenced_node_migrations"]
    }


def test_the_table_they_alter_exists_before_them_in_runner_order() -> None:
    """run-forward-migrations.sh applies a node's files in lexical sort order, so
    on a fresh database 0008 runs after 0000..0007 but BEFORE 003 and
    004. That is safe only while they need nothing but the table 0000 creates."""
    order = sorted(path.name for path in _VENDOR.glob("*.sql"))
    for filename in _VENDORED:
        assert order.index(_CREATE) < order.index(filename)
    assert re.search(
        r"CREATE TABLE IF NOT EXISTS tenant_inference_credentials\s*\(",
        _statements(_CREATE),
    )


@pytest.mark.parametrize("filename", sorted(_VENDORED))
@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
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


@pytest.mark.parametrize("filename", sorted(_VENDORED))
@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_the_sql_gate_rejects_an_unowned_target(filename: str, profile: str) -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / filename).read_text(encoding="utf-8")
    broken = sql.replace(
        "ALTER TABLE tenant_inference_credentials",
        "ALTER TABLE unowned_schema.tenant_inference_credentials",
    )
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile(profile)) != ()
