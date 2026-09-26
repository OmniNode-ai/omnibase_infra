# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17886: the live-ACL diff for ``omninode_internal``, without a database.

The diff is pure over catalog rows shaped like the three LIVE_*_QUERY results,
so every rule is pinned here against rows built from the REAL onex-dev
declaration. The real-Postgres half (the queries against a live catalog) is
``tests/integration/migrations/test_omninode_internal_live_acl_omn17886.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.physical_schema_mapping import (
    physical_grant_schema_for_table,
)
from omnibase_infra.validation.omninode_internal_live_acl import (
    DeclaredAcl,
    declared_acl,
    diff_live_acl,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "validation" / "check_omninode_internal_live_acl.py"
OWNER = "postgres"


def _declared() -> DeclaredAcl:
    return declared_acl(load_topology_profile("onex-dev"))


def _rows_for(declared: DeclaredAcl) -> dict[str, list[dict[str, object]]]:
    """Live rows exactly matching ``declared``, owner rows included."""
    relation_rows: list[dict[str, object]] = []
    for relation in sorted(declared.relations):
        for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE"):
            relation_rows.append(
                {
                    "relation_name": relation,
                    "relkind": "r",
                    "owner": OWNER,
                    "grantee": OWNER,
                    "privilege_type": privilege,
                }
            )
    for principal, relation, privilege in sorted(declared.table_grants):
        relation_rows.append(
            {
                "relation_name": relation,
                "relkind": "r",
                "owner": OWNER,
                "grantee": principal,
                "privilege_type": privilege,
            }
        )
    schema_rows: list[dict[str, object]] = [
        {"owner": OWNER, "grantee": OWNER, "privilege_type": "USAGE"},
        {"owner": OWNER, "grantee": OWNER, "privilege_type": "CREATE"},
    ]
    schema_rows += [
        {"owner": OWNER, "grantee": principal, "privilege_type": privilege}
        for principal, privilege in sorted(declared.schema_grants)
    ]
    return {"relation": relation_rows, "schema": schema_rows, "default": []}


def _diff(rows: dict[str, list[dict[str, object]]], declared: DeclaredAcl):
    return diff_live_acl(
        declared,
        relation_rows=rows["relation"],
        schema_rows=rows["schema"],
        default_rows=rows["default"],
    )


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_omninode_internal_live_acl", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_declaration_is_the_real_one_and_physically_resolved() -> None:
    declared = _declared()
    assert ("omninode_runtime", "live_events", "INSERT") in declared.table_grants
    assert ("omninode_runtime", "USAGE") in declared.schema_grants
    for relation in declared.relations:
        assert physical_grant_schema_for_table("omninode_internal", relation) == (
            "omninode_internal"
        )


def test_live_catalog_matching_the_declaration_is_clean() -> None:
    declared = _declared()
    report = _diff(_rows_for(declared), declared)
    assert report.findings == ()
    # Positive control: the diff saw every declared relation, so "clean" is a
    # measurement rather than an empty read.
    assert report.observed_relation_count == len(declared.relations) > 0


def test_one_undeclared_and_one_missing_grant_are_both_named() -> None:
    declared = _declared()
    rows = _rows_for(declared)
    rows["relation"].append(
        {
            "relation_name": "live_events",
            "relkind": "r",
            "owner": OWNER,
            "grantee": "jake_ro",
            "privilege_type": "SELECT",
        }
    )
    rows["relation"] = [
        row
        for row in rows["relation"]
        if not (
            row["relation_name"] == "work_events"
            and row["grantee"] == "omninode_runtime"
            and row["privilege_type"] == "UPDATE"
        )
    ]
    report = _diff(rows, declared)
    assert report.findings == (
        "UNDECLARED_GRANT jake_ro SELECT ON omninode_internal.live_events",
        "MISSING_DECLARED_GRANT omninode_runtime UPDATE ON omninode_internal.work_events",
    )


def test_a_missing_relation_is_reported_once_not_per_privilege() -> None:
    declared = _declared()
    rows = _rows_for(declared)
    rows["relation"] = [
        r for r in rows["relation"] if r["relation_name"] != "live_events"
    ]
    report = _diff(rows, declared)
    assert report.findings == (
        "MISSING_DECLARED_RELATION omninode_internal.live_events",
    )


def test_the_owners_implicit_privileges_are_not_grants() -> None:
    declared = _declared()
    report = _diff(_rows_for(declared), declared)
    assert set(report.relation_owners.values()) == {OWNER}
    assert not any(OWNER in line for line in report.findings)


def test_a_default_privilege_rule_is_a_finding() -> None:
    declared = _declared()
    rows = _rows_for(declared)
    rows["default"] = [
        {
            "owner": OWNER,
            "object_type": "r",
            "grantee": "omninode_runtime",
            "privilege_type": "INSERT",
        }
    ]
    report = _diff(rows, declared)
    assert report.findings == (
        "UNDECLARED_DEFAULT_PRIVILEGE owner=postgres objtype=r omninode_runtime "
        "INSERT IN SCHEMA omninode_internal",
    )


def test_an_undeclared_schema_grant_is_a_finding() -> None:
    declared = _declared()
    rows = _rows_for(declared)
    rows["schema"].append(
        {"owner": OWNER, "grantee": "jake_ro", "privilege_type": "USAGE"}
    )
    report = _diff(rows, declared)
    assert report.findings == (
        "UNDECLARED_SCHEMA_GRANT jake_ro USAGE ON SCHEMA omninode_internal",
    )


def test_cli_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli()
    declared = _declared()
    clean = _rows_for(declared)
    dirty = _rows_for(declared)
    dirty["default"] = [
        {
            "owner": OWNER,
            "object_type": "r",
            "grantee": "jake_ro",
            "privilege_type": "SELECT",
        }
    ]
    monkeypatch.setenv("OMN17886_TEST_DSN", "postgresql://unused")

    monkeypatch.setattr(cli, "_read_live", lambda dsn, schema: clean)
    assert cli.main(["--profile", "onex-dev", "--dsn-env", "OMN17886_TEST_DSN"]) == 0

    monkeypatch.setattr(cli, "_read_live", lambda dsn, schema: dirty)
    assert cli.main(["--profile", "onex-dev", "--dsn-env", "OMN17886_TEST_DSN"]) == 1

    assert (
        cli.main(["--profile", "no-such-profile", "--dsn-env", "OMN17886_TEST_DSN"])
        == 2
    )
    monkeypatch.delenv("OMN17886_TEST_DSN")
    assert cli.main(["--profile", "onex-dev", "--dsn-env", "OMN17886_TEST_DSN"]) == 2
