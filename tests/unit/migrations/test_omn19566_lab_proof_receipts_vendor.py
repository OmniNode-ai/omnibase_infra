# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19566 vendor identity for the lab proof receipt projection migrations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_lab_proof_receipts"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CREATE = "0000_create_lab_proof_receipts.sql"
_GRANT = "0001_grant_omninode_runtime_lab_proof_receipts.sql"
_SHA256 = {
    _CREATE: "d3f8330b99a5795de241ee1e2d40801e19f67c888a52f6032fa8130f37ec9bd6",
    _GRANT: "673d00436921a386ddb2cdf1bf1ccb0c34816d6417e9fd8fa0b13cec823fad20",
}
_TABLES = ("lab_proof_receipts",)


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
        "omninode_internal",
        f"node:{_NODE}:{filename}",
        _SHA256[filename],
    ]


def test_migration_classes_match_the_classifier() -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_CREATE}"] == "forward-only"
    assert classes[f"forward/nodes/{_NODE}/{_GRANT}"] == "expand-only"


def test_table_and_exact_runtime_grants_are_present() -> None:
    create = _statements(_CREATE)
    grants = _statements(_GRANT)
    for table in _TABLES:
        assert re.search(
            rf"CREATE TABLE IF NOT EXISTS omninode_internal\.{table}\s*\(", create
        )
        assert re.search(
            rf"GRANT\s+SELECT,\s*INSERT,\s*UPDATE\s+"
            rf"ON omninode_internal\.{table}\s+TO omninode_runtime;",
            grants,
        )
        assert re.search(
            rf"GRANT\s+USAGE\s+ON SEQUENCE "
            rf"omninode_internal\.{table}_projection_cursor_seq\s+"
            r"TO omninode_runtime;",
            grants,
        )
    assert "GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;" in grants
    for sql in (create, grants):
        assert not re.search(
            r"GRANT\s+[^;]*\bDELETE\b[^;]*\bTO\s+omninode_runtime\b",
            sql,
            re.IGNORECASE,
        )


# The writer reads the stored proof and upserts the newer receipt. It needs
# SELECT, INSERT and UPDATE, cursor sequence USAGE, and schema USAGE.
_EXPECTED_GRANTS = frozenset(
    {
        ("USAGE", "SCHEMA omninode_internal"),
        *(
            (privilege, f"omninode_internal.{table}")
            for table in _TABLES
            for privilege in ("SELECT", "INSERT", "UPDATE")
        ),
        *(
            ("USAGE", f"SEQUENCE omninode_internal.{table}_projection_cursor_seq")
            for table in _TABLES
        ),
    }
)


def _grants(sql: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for statement in re.findall(
        r"^\s*GRANT\b[^;]*;", sql, re.IGNORECASE | re.MULTILINE
    ):
        match = re.fullmatch(
            r"GRANT\s+(?P<privs>[A-Z,\s]+?)\s+ON\s+(?P<target>.+?)\s+"
            r"TO\s+omninode_runtime\s*;",
            " ".join(statement.split()),
        )
        assert match, f"unparsed or non-omninode_runtime GRANT: {statement!r}"
        for privilege in match["privs"].split(","):
            found.add((privilege.strip(), match["target"]))
    return found


def test_the_grants_are_exactly_the_writer_scope() -> None:
    """Every GRANT in both files, parsed, equals the writer's scope and no more.

    No DELETE, TRUNCATE, REFERENCES, TRIGGER, CREATE or ALL; no schema-wide
    ``ON ALL TABLES IN SCHEMA``; no ``WITH GRANT OPTION``; no other grantee.
    A broader grant added later fails here by name.
    """
    assert _grants(_statements(_CREATE)) == set()
    assert _grants(_statements(_GRANT)) == _EXPECTED_GRANTS
    assert "WITH GRANT OPTION" not in _statements(_GRANT).upper()


def test_table_and_sequence_privileges_are_asserted() -> None:
    """The vendored migration asserts its table and cursor sequence grants."""
    grants = _statements(_GRANT)
    for table in _TABLES:
        for privilege in ("select", "insert", "update"):
            assert f"AS {table}_{privilege}_grant_assertion" in grants
        assert f"AS {table}_cursor_sequence_usage_assertion" in grants


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
    broken = sql.replace(
        "omninode_internal.lab_proof_receipts", "tenant.lab_proof_receipts"
    )
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()
