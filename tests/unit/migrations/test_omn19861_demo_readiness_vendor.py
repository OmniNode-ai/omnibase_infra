# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19861 vendor identity for the demo-readiness projection migrations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_demo_readiness"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CREATE = "0000_create_demo_readiness_latest.sql"
_GRANT = "0001_grant_omninode_runtime_demo_readiness_latest.sql"
_SHA256 = {
    _CREATE: "702b6c6acde9170124ad7a81a4b54e50149c750724169f9506b5d84ce8049760",
    _GRANT: "1373cc27ecd2e38f918d203b32b0427c07d81e38d62367986e11538556e95004",
}
_TABLE = "demo_readiness_latest"
_CURSOR_SEQUENCE = "demo_readiness_latest_projection_cursor_seq"


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
    assert re.search(
        rf"CREATE TABLE IF NOT EXISTS omninode_internal\.{_TABLE}\s*\(", create
    )
    assert re.search(
        rf"GRANT\s+SELECT,\s*INSERT,\s*UPDATE\s+"
        rf"ON omninode_internal\.{_TABLE}\s+TO omninode_runtime;",
        grants,
    )
    assert re.search(
        rf"GRANT\s+USAGE\s+ON SEQUENCE "
        rf"omninode_internal\.{_CURSOR_SEQUENCE}\s+"
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


# The writer's whole scope is SELECT, INSERT, and UPDATE on the latest-status
# projection, USAGE on its BIGSERIAL cursor sequence for nextval(), and USAGE
# on the schema so the role can resolve omninode_internal.*. Schema USAGE is
# name lookup only; it confers no privilege on any relation in the schema. This
# set is also exactly what the omninode_runtime principal receives in topology.
_EXPECTED_GRANTS = frozenset(
    {
        ("USAGE", "SCHEMA omninode_internal"),
        *(
            (privilege, f"omninode_internal.{_TABLE}")
            for privilege in ("SELECT", "INSERT", "UPDATE")
        ),
        ("USAGE", f"SEQUENCE omninode_internal.{_CURSOR_SEQUENCE}"),
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
        "omninode_internal.demo_readiness_latest", "tenant.demo_readiness_latest"
    )
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()
