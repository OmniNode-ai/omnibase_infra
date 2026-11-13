# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration 115 grants role_omnibase_infra the state_io tables (OMN-24037).

The runtime account ``role_omnibase_infra`` is declared in the topology
instances with a TABLE grant of INSERT, SELECT and UPDATE on the six
omnibase_infra state_io tables, but no migration ever issued that grant.
This module pins:

* the migration exists at ordinal 115 and no other file claims that ordinal;
* the GRANT names exactly the tables and privileges the topology declares;
* the GRANT carries no DELETE, no ALL and no wildcard;
* the post-condition reads ``has_table_privilege`` and guards on ``pg_roles``;
* the parser finds no grant in a migration without one (the failing control);
* the class is declared in ``config/migration_classes.yaml``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
MIGRATION = FORWARD_DIR / "115_grant_role_omnibase_infra_state_io_omn19998.sql"
INSTANCES_DIR = REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
SHIPPED_INSTANCES = tuple(sorted(path.stem for path in INSTANCES_DIR.glob("*.yaml")))
assert len(SHIPPED_INSTANCES) == 3, SHIPPED_INSTANCES

_GRANT_RE = re.compile(
    r"GRANT\s+(?P<privs>[A-Za-z,\s]+)\s+ON\s+TABLE\s+(?P<rels>[\w.,\s]+)\s+TO\s+role_omnibase_infra\s*;",
    re.IGNORECASE,
)


def _parse_grant(sql_text: str) -> tuple[set[str], set[str]] | None:
    match = _GRANT_RE.search(sql_text)
    if match is None:
        return None
    privileges = {
        p.strip().upper() for p in match.group("privs").split(",") if p.strip()
    }
    tables = {r.strip() for r in match.group("rels").split(",") if r.strip()}
    return privileges, tables


def _strip_comments(sql_text: str) -> str:
    return "\n".join(
        line for line in sql_text.splitlines() if not line.lstrip().startswith("--")
    )


@pytest.mark.unit
def test_the_migration_exists_at_ordinal_115() -> None:
    assert MIGRATION.is_file(), f"{MIGRATION.name} is missing"
    assert not any(
        path.name.startswith("115_") and path != MIGRATION
        for path in FORWARD_DIR.glob("*.sql")
    )


@pytest.mark.parametrize("instance", SHIPPED_INSTANCES)
@pytest.mark.unit
def test_the_grant_names_exactly_the_tables_the_topology_declares(
    instance: str,
) -> None:
    raw = yaml.safe_load((INSTANCES_DIR / f"{instance}.yaml").read_text("utf-8"))
    grants = raw["databases"]["omnibase_infra"]["principals"]["role_omnibase_infra"][
        "grants"
    ]
    table_grant = next(g for g in grants if g["object_type"] == "TABLE")
    declared_tables = {f"public.{name}" for name in table_grant["objects"]}
    declared_privileges = set(table_grant["privileges"])

    parsed = _parse_grant(MIGRATION.read_text(encoding="utf-8"))
    assert parsed is not None, "no GRANT statement found in the migration"
    privileges, tables = parsed

    assert tables == declared_tables
    assert privileges == {"SELECT", "INSERT", "UPDATE"}
    assert privileges == declared_privileges

    # Positive control: floor_fill_state is in the declared set and the set has at least 6 tables.
    assert "public.floor_fill_state" in declared_tables
    assert len(declared_tables) >= 6


@pytest.mark.unit
def test_the_grant_never_carries_delete_or_a_wildcard() -> None:
    stripped = _strip_comments(MIGRATION.read_text(encoding="utf-8"))
    assert "DELETE" not in stripped
    assert " ALL " not in stripped
    assert "TRUNCATE" not in stripped
    assert "TO PUBLIC" not in stripped


@pytest.mark.unit
def test_the_post_condition_reads_has_table_privilege() -> None:
    stripped = _strip_comments(MIGRATION.read_text(encoding="utf-8"))
    assert "has_table_privilege" in stripped
    assert "RAISE EXCEPTION" in stripped
    assert "pg_roles" in stripped


@pytest.mark.unit
def test_the_parser_finds_no_grant_in_a_migration_without_one() -> None:
    sql_text = MIGRATION.read_text(encoding="utf-8")
    without_grant = re.sub(_GRANT_RE, "", sql_text)
    assert _parse_grant(without_grant) is None
    assert _parse_grant("") is None


@pytest.mark.unit
def test_the_class_is_declared() -> None:
    classes_text = (REPO_ROOT / "config" / "migration_classes.yaml").read_text(
        encoding="utf-8"
    )
    assert (
        "  forward/115_grant_role_omnibase_infra_state_io_omn19998.sql: "
        in classes_text
    )
