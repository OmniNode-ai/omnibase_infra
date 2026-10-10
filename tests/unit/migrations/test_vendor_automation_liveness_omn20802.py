# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20802 vendor identity for the automation-liveness projection.

omnimarket adds node_projection_automation_liveness, three omninode_internal
relations written by the runtime principal with SELECT, INSERT and UPDATE. The
node-migration vendor-parity gate refuses that pull request until the
byte-identical copies are on this repo's dev, and a runtime that installs the
node without the topology grants exits at wiring.
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
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"

_NODE = "node_projection_automation_liveness"
_CREATE = "0000_create_automation_liveness.sql"
_GRANT = "0001_grant_omninode_runtime_automation_liveness.sql"
_FILES = (_CREATE, _GRANT)
_SHA256 = {
    _CREATE: "c90716b8f1f019e6e79efa4eb2dded9561b43da50d9a3d40e67023c4b57f51dd",
    _GRANT: "50c9955046fe744d78d351e84ab897da17603e948bcca5abf7e60eb2bbf22300",
}
_TABLES = (
    "automation_liveness_state",
    "automation_run_history",
    "automation_alarm_episodes",
)


def _vendored(filename: str) -> Path:
    return _FORWARD / "nodes" / _NODE / filename


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in _vendored(filename).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


@pytest.mark.parametrize("filename", _FILES)
def test_vendor_bytes_and_manifest_binding_are_exact(filename: str) -> None:
    artifact_path = f"nodes/{_NODE}/{filename}"
    assert (
        hashlib.sha256(_vendored(filename).read_bytes()).hexdigest()
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
        "omninode_internal",
        f"node:{_NODE}:{filename}",
        _SHA256[filename],
    ]


@pytest.mark.parametrize("filename", _FILES)
def test_migration_classes_match_the_classifier(filename: str) -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    declared = classes[f"forward/nodes/{_NODE}/{filename}"]
    # The create adds a unique index on a created table, which the classifier
    # reads as forward-only; the grant file is additive.
    expected = "forward-only" if filename == _CREATE else "expand-only"
    assert declared.split(" #")[0] == expected


@pytest.mark.parametrize("table", _TABLES)
def test_each_relation_and_its_runtime_grant_are_present(table: str) -> None:
    create = _statements(_CREATE)
    grants = _statements(_GRANT)
    assert re.search(
        rf"CREATE TABLE IF NOT EXISTS omninode_internal\.{table}\s*\(", create
    )
    assert "GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;" in grants
    assert re.search(
        rf"GRANT\s+SELECT,\s*INSERT,\s*UPDATE\s+ON omninode_internal\.{table}\s+"
        r"TO omninode_runtime;",
        grants,
    )
    assert not re.search(
        r"GRANT\s+[^;]*\bDELETE\b[^;]*\bTO\s+omninode_runtime\b",
        grants,
        re.IGNORECASE,
    )


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_all_three_relations_are_granted_in_the_shipped_topology(profile: str) -> None:
    from omnibase_infra.topology.application_database import load_topology_profile

    text = yaml.safe_dump(
        load_topology_profile(profile).model_dump(mode="json"), sort_keys=True
    )
    for table in _TABLES:
        assert table in text, f"{profile} does not grant {table}"


@pytest.mark.parametrize("filename", _FILES)
@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_migrations_pass_the_application_database_sql_gate(
    filename: str, profile: str
) -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _vendored(filename).read_text(encoding="utf-8")
    violations = lint_application_database_sql(sql, load_topology_profile(profile))
    assert violations == (), f"{profile}/{filename}: {violations}"


def test_the_sql_gate_is_live_positive_control() -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _vendored(_CREATE).read_text(encoding="utf-8")
    broken = sql.replace(
        "omninode_internal.automation_liveness_state",
        "tenant.automation_liveness_state",
    )
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()
