# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19961 vendor identity for the lab container memory projection migrations.

omnimarket's node_projection_lab_container_memory migration 0000 creates
``omninode_internal.lab_container_memory_window`` (one row per lane container
per census window, projected from the lane container memory event of
OMN-19959) and 0001 grants its projection writer SELECT, INSERT and UPDATE
through ``omninode_runtime``. Both are vendored here FIRST, ahead of the
omnimarket source, per the node-migration vendor-parity ordering.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_lab_container_memory"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CREATE = "0000_create_lab_container_memory_window.sql"
_GRANT = "0001_grant_omninode_runtime_lab_container_memory_window.sql"
_SHA256 = {
    _CREATE: "dd318ad4c41b4b8f0b8c53aeddc443cb5c1a30fd4e2b3a3f4679acef6d0fe58d",
    _GRANT: "974b61d5078f68a8d9bde80229d21a038ccdb486b24782246faacf4bcb4757df",
}


def _sql(filename: str) -> str:
    return (_VENDOR / filename).read_text(encoding="utf-8")


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in _sql(filename).splitlines()
        if not line.lstrip().startswith("--")
    )


def _manifest_rows() -> dict[str, list[str]]:
    return {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
def test_vendor_bytes_and_manifest_binding_are_exact(filename: str) -> None:
    artifact_path = f"nodes/{_NODE}/{filename}"
    digest = hashlib.sha256((_VENDOR / filename).read_bytes()).hexdigest()
    assert digest == _SHA256[filename]
    assert _manifest_rows()[artifact_path] == [
        artifact_path,
        f"node:{_NODE}",
        f"node:{_NODE}",
        "omninode_internal",
        f"node:{_NODE}:{filename}",
        _SHA256[filename],
    ]


def test_both_migrations_are_declared_expand_only() -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    for filename in (_CREATE, _GRANT):
        assert classes[f"forward/nodes/{_NODE}/{filename}"] == "expand-only"


def test_the_table_is_keyed_on_record_key_and_the_grant_withholds_delete() -> None:
    create = _statements(_CREATE)
    assert re.search(
        r"CREATE TABLE IF NOT EXISTS omninode_internal\.lab_container_memory_window\s*\(",
        create,
    )
    assert "PRIMARY KEY (record_key)" in create

    grant = _statements(_GRANT)
    assert "GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;" in grant
    assert re.search(
        r"GRANT SELECT, INSERT, UPDATE\s+ON omninode_internal\.lab_container_memory_window"
        r"\s+TO omninode_runtime;",
        grant,
    )
    assert not re.search(
        r"GRANT\s+[^;]*\bDELETE\b[^;]*\bTO\s+omninode_runtime\b",
        grant,
        re.IGNORECASE,
    )


@pytest.mark.parametrize("filename", [_CREATE, _GRANT])
def test_migration_passes_the_application_database_sql_gate(filename: str) -> None:
    """Both files clear the OMN-15361 gate on every shipped profile."""
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _sql(filename)
    for profile in ("local", "onex-dev", "onex-prod", "stability-test"):
        violations = lint_application_database_sql(sql, load_topology_profile(profile))
        assert violations == (), f"{profile}: {violations}"


def test_the_linter_is_live_positive_control() -> None:
    """Retargeting the table to the retired tenant schema must be refused."""
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _sql(_CREATE)
    broken = sql.replace(
        "omninode_internal.lab_container_memory_window",
        "tenant.lab_container_memory_window",
    )
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()


def test_every_shipped_instance_grants_the_runtime_exactly_the_writers_privileges() -> (
    None
):
    from omnibase_infra.topology.application_database import load_topology_profile

    for profile in ("local", "onex-dev", "onex-prod"):
        topology = load_topology_profile(profile)
        granted = {
            privilege.value
            for database in topology.databases.values()
            for principal_name, principal in database.principals.items()
            if principal_name == "omninode_runtime"
            for grant in principal.grants
            if "lab_container_memory_window" in grant.objects
            for privilege in grant.privileges
        }
        assert granted == {"SELECT", "INSERT", "UPDATE"}, profile
