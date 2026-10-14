# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20737: the parity check's read-only role may read work_ledger_state."""

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
_GRANT = "0004_grant_work_ledger_ro_work_ledger_state.sql"
_PATH = _FORWARD / "nodes" / _NODE / _GRANT
_ROLE = "work_ledger_ro"
_RELATION = "omninode_internal.work_ledger_state"


def _statements() -> list[str]:
    sql = re.sub(r"--[^\n]*", "", _PATH.read_text(encoding="utf-8"))
    return [" ".join(s.split()) for s in sql.split(";") if s.strip()]


def test_the_migration_is_the_next_file_of_the_work_ledger_lineage() -> None:
    names = sorted(path.name for path in _PATH.parent.glob("*.sql"))
    assert names[-1] == _GRANT
    assert names.index(_GRANT) == 4


def test_it_grants_select_on_the_state_table_to_the_read_only_role() -> None:
    joined = " ".join(_statements())
    assert f"GRANT SELECT ON TABLE {_RELATION} TO {_ROLE}" in joined


def test_the_grant_is_select_only_and_touches_no_credential() -> None:
    joined = " ".join(_statements())
    grants = re.findall(r"GRANT\s+([A-Z, ]+?)\s+ON\b", joined)
    assert grants == ["SELECT"]
    assert not re.search(
        r"\b(INSERT|UPDATE|DELETE|TRUNCATE|ALL|OPTION|CREATE ROLE|ALTER ROLE|PASSWORD)\b",
        joined,
    )


def test_the_role_is_provisioned_out_of_band_so_the_grant_is_guarded_on_it() -> None:
    joined = " ".join(_statements())
    assert f"rolname = '{_ROLE}'" in joined
    assert joined.index(f"rolname = '{_ROLE}'") < joined.index(f"TO {_ROLE}")


def test_the_migration_asserts_the_grant_when_the_role_exists() -> None:
    joined = " ".join(_statements())
    assert f"has_table_privilege('{_ROLE}', '{_RELATION}', 'SELECT')" in joined
    assert "RAISE EXCEPTION" in joined


def test_manifest_row_binds_the_exact_bytes() -> None:
    artifact = f"nodes/{_NODE}/{_GRANT}"
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
            hashlib.sha256(_PATH.read_bytes()).hexdigest(),
        ]
    ]


def test_migration_class_is_expand_only() -> None:
    classes = yaml.safe_load(
        (_ROOT / "config" / "migration_classes.yaml").read_text(encoding="utf-8")
    )["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_GRANT}"] == "expand-only"
