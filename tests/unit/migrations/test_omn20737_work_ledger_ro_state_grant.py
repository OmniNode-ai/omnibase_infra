# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20737: pin the work_ledger_ro grant on omninode_internal.work_ledger_state."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_work_ledger"
_VENDOR = _FORWARD / "nodes" / _NODE
_GRANT = "0004_grant_work_ledger_ro_work_ledger_state.sql"
_SHA256 = "47ade9a1e05a81c32c096649d477d1b2389a632e10318ef30dd8c9d4a45d090d"


def _statements() -> str:
    return "\n".join(
        line
        for line in (_VENDOR / _GRANT).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


def test_bytes_and_ledger_binding_are_exact() -> None:
    artifact = f"nodes/{_NODE}/{_GRANT}"
    assert hashlib.sha256((_VENDOR / _GRANT).read_bytes()).hexdigest() == _SHA256
    rows = [
        row.split("\t")
        for row in (_FORWARD / "_ledger" / "application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if row.split("\t", 1)[0] == artifact
    ]
    assert rows == [
        [
            artifact,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "omninode_internal",
            f"node:{_NODE}:{_GRANT}",
            _SHA256,
        ]
    ]


def test_migration_class_is_expand_only() -> None:
    classes = yaml.safe_load(
        (_ROOT / "config" / "migration_classes.yaml").read_text(encoding="utf-8")
    )["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_GRANT}"] == "expand-only"


def test_the_only_grant_is_select_on_the_state_table_guarded_on_the_role() -> None:
    sql = " ".join(_statements().split())
    grants = re.findall(r"\bGRANT\b[^;]*;", sql)
    assert grants == [
        "GRANT SELECT ON TABLE omninode_internal.work_ledger_state TO work_ledger_ro;"
    ]
    assert "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'work_ledger_ro')" in sql
    assert not re.search(
        r"\b(CREATE|ALTER|DROP)\s+ROLE\b|\bPASSWORD\b|\bWITH GRANT OPTION\b|\bEXECUTE\b",
        sql,
        re.IGNORECASE,
    )


@pytest.mark.parametrize("profile", ["local", "test"])
def test_migration_passes_the_application_database_sql_gate(profile: str) -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / _GRANT).read_text(encoding="utf-8")
    violations = lint_application_database_sql(sql, load_topology_profile(profile))
    assert violations == (), f"{profile}: {violations}"
