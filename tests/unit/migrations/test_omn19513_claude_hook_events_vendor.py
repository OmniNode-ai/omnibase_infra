# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19513 vendor identity for the Claude hook event projection migrations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_claude_hook_events"
_VENDOR = _FORWARD / "nodes" / _NODE
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CREATE = "0000_create_claude_hook_events.sql"
_GRANT = "0001_grant_omninode_runtime_claude_hook_events.sql"
_GOAL = "0005_add_goal_id.sql"
_FIELDS = "0006_add_lane_model_host_exit_code.sql"
_SHA256 = {
    _CREATE: "388faf7f8d683a28341b306a5d67dc208b98aaf5b8f39789b97361cba4ed72ae",
    _GRANT: "c69298981a4a481d394c0f9cbf9c5374904644981073afa84ce87ae4ae4182d0",
    _GOAL: "14f1c5e67a51e1d30da1b21c5267de497e0bec7d2e443471a45d905b7de3708f",
    _FIELDS: "ec509c4d99d8daaf831c2ea4eebec209f64bfd45e57410b5916739c1e91b078b",
}
_TABLES = ("claude_agent_spans", "claude_hook_events")


def _statements(filename: str) -> str:
    return "\n".join(
        line
        for line in (_VENDOR / filename).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


@pytest.mark.parametrize("filename", [_CREATE, _GRANT, _GOAL, _FIELDS])
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
    assert classes[f"forward/nodes/{_NODE}/{_GOAL}"] == "expand-only"
    assert classes[f"forward/nodes/{_NODE}/{_FIELDS}"] == "expand-only"


def test_the_goal_migration_adds_nullable_uuid_columns_and_a_partial_index() -> None:
    """OMN-20031: the omnimarket 0005, vendored byte for byte, declares its shape."""
    sql = _statements(_GOAL)
    for column in ("goal_id", "parent_goal_id"):
        assert re.search(
            rf"ALTER TABLE omninode_internal\.claude_hook_events\s+"
            rf"ADD COLUMN IF NOT EXISTS {column} UUID;",
            sql,
        )
    assert re.search(
        r"CREATE INDEX IF NOT EXISTS idx_claude_hook_events_goal_id\s+"
        r"ON omninode_internal\.claude_hook_events \(goal_id\)\s+"
        r"WHERE goal_id IS NOT NULL;",
        sql,
    )
    assert _grants(sql) == set()


def test_both_tables_and_exact_runtime_grants_are_present() -> None:
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


# The writer's whole scope (omnimarket node_projection_claude_hook_events,
# handler_claude_hook_events_writer): SELECT for the parent-call lookup and the
# tool-call recount, INSERT for the event insert and the span upsert, UPDATE for
# the span upsert's DO UPDATE and the out-of-order parent repair, USAGE on each
# BIGSERIAL cursor sequence for nextval(), and USAGE on the schema so the role
# can resolve omninode_internal.* at all. Schema USAGE is name lookup only; it
# confers no privilege on any relation in the schema. This set is also exactly
# what the omninode_runtime principal is given in the topology instances.
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
    # A GRANT statement opens a line; the 0000 precondition's quoted
    # 'USAGE WITH GRANT OPTION' argument is a string literal, not a statement.
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


def test_every_issued_privilege_is_asserted_including_schema_usage() -> None:
    """A grant that did not take must fail the migration, not the first write."""
    grants = _statements(_GRANT)
    assert re.search(
        r"SELECT 1 / count\(\*\) AS omninode_runtime_schema_usage_grant_assertion\s+"
        r"WHERE has_schema_privilege\(\s*'omninode_runtime',\s*"
        r"'omninode_internal',\s*'USAGE'\s*\);",
        grants,
    )
    for table in _TABLES:
        for privilege in ("select", "insert", "update"):
            assert f"AS {table}_{privilege}_grant_assertion" in grants
        assert f"AS {table}_cursor_sequence_usage_assertion" in grants


@pytest.mark.parametrize("filename", [_CREATE, _GRANT, _GOAL, _FIELDS])
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
        "omninode_internal.claude_hook_events", "tenant.claude_hook_events"
    )
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()


def test_the_fields_migration_adds_four_nullable_columns() -> None:
    """OMN-17427: the omnimarket 0006, vendored byte for byte, declares its shape."""
    sql = _statements(_FIELDS)
    for column, kind in (
        ("lane", "TEXT"),
        ("model", "TEXT"),
        ("host", "TEXT"),
        ("exit_code", "INTEGER"),
    ):
        assert re.search(
            rf"ALTER TABLE omninode_internal\.claude_hook_events\s+"
            rf"ADD COLUMN IF NOT EXISTS {column} {kind};",
            sql,
        )
    assert _grants(sql) == set()
