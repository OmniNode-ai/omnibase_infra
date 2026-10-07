# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20287 vendor identity for the delegation_events run-lineage migration.

omnimarket's node_projection_delegation ``0055_delegation_events_run_lineage.sql``
adds the nullable ``parent_correlation_id``, ``attempt_kind`` and
``parent_failure_cause`` columns to ``delegation_events`` and a partial index on
``parent_correlation_id``. It is vendored here under its own file name, so its
namespaced migration id (``node:<node>:<file>``) is distinct from the sibling
``0055_delegation_events_lineage.sql`` and ``0055_model_routing_local_call_share.sql``
that already sit at the same ordinal.

Failure modes pinned here:
  1. a vendored byte drifts from the ledger binding;
  2. the file is declared a class other than the class checker's own reading;
  3. the file fails the application-database SQL gate on a shipped profile, and
     that gate can still fail at all (positive control);
  4. the three 0055 files collide on a namespaced id, or the run-lineage file
     and the sibling lineage file disagree on the one index they both create,
     which would make the result depend on apply order.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_CLASS_CHECKER = _ROOT / "scripts" / "validation" / "check_migration_class.py"
_NODE = "node_projection_delegation"
_PROFILES = ("local", "onex-dev", "onex-prod", "stability-test")

_FILE = "0055_delegation_events_run_lineage.sql"
_SHA256 = "fc094083943fafe611391abc58fd822a113a499760e4a7ac990e5c0ba1c662aa"
_SIBLING = "0055_delegation_events_lineage.sql"

# The falsifier is an undeclared schema, and the control asserts which rule
# fired (see test_omn19860_caller_lane_vendor.py for why a bare relation or the
# `public` schema is not a live control).
_UNDECLARED_SCHEMA_TARGET = "undeclared_topology_schema.delegation_events"
_UNDECLARED_SCHEMA_RULE = "unknown topology schema"


def _sql(filename: str = _FILE) -> str:
    return (_FORWARD / "nodes" / _NODE / filename).read_text(encoding="utf-8")


def _class_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_migration_class", _CLASS_CHECKER
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vendor_bytes_and_manifest_binding_are_exact() -> None:
    artifact_path = f"nodes/{_NODE}/{_FILE}"
    assert hashlib.sha256((_FORWARD / artifact_path).read_bytes()).hexdigest() == (
        _SHA256
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
        "tenant",
        f"node:{_NODE}:{_FILE}",
        _SHA256,
    ]


def test_declared_class_is_expand_only_as_the_checker_reads_it() -> None:
    """Three nullable ADD COLUMNs and a CREATE INDEX are additive: old code
    redeployed over this schema still reads and writes delegation_events."""
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_FILE}"] == "expand-only"
    assert _class_checker().destructive_findings(_sql()) == []


def test_migration_passes_the_application_database_sql_gate() -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _sql()
    for profile in _PROFILES:
        violations = lint_application_database_sql(sql, load_topology_profile(profile))
        assert violations == (), f"{profile}: {violations}"


def test_the_linter_is_live_positive_control() -> None:
    """A zero from the linter means something only if it can return non-zero,
    on every profile the gate above clears."""
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _sql()
    anchor = "ALTER TABLE delegation_events"
    broken = sql.replace(anchor, f"ALTER TABLE {_UNDECLARED_SCHEMA_TARGET}", 1)
    assert broken != sql, f"the {anchor!r} anchor is no longer present in {_FILE}"
    for profile in _PROFILES:
        violations = lint_application_database_sql(
            broken, load_topology_profile(profile)
        )
        assert any(_UNDECLARED_SCHEMA_RULE in violation for violation in violations), (
            f"{profile}: retargeting at {_UNDECLARED_SCHEMA_TARGET!r} did not raise "
            f"{_UNDECLARED_SCHEMA_RULE!r}; got {violations}"
        )


def test_every_statement_is_idempotent_and_additive() -> None:
    """The file re-applies safely on a database the sibling 0055 already
    touched: every ADD COLUMN and CREATE INDEX is guarded and nothing is dropped."""
    sql = _sql()
    adds = re.findall(r"ALTER TABLE delegation_events ADD COLUMN (.+?);", sql)
    assert [add.split()[:4] for add in adds] == [
        ["IF", "NOT", "EXISTS", "parent_correlation_id"],
        ["IF", "NOT", "EXISTS", "attempt_kind"],
        ["IF", "NOT", "EXISTS", "parent_failure_cause"],
    ]
    assert re.findall(r"CREATE (?:UNIQUE )?INDEX (?!IF NOT EXISTS)", sql) == []
    assert not re.search(r"\b(?:DROP|RENAME|SET NOT NULL|DEFAULT)\b", sql)
    assert sql.count("BEGIN;") == 1
    assert sql.rindex("COMMIT;") > sql.rindex("CREATE INDEX")


def test_the_three_0055_files_have_distinct_namespaced_ids() -> None:
    files = sorted(
        path.name for path in (_FORWARD / "nodes" / _NODE).glob("0055_*.sql")
    )
    assert files == [
        _SIBLING,
        _FILE,
        "0055_model_routing_local_call_share.sql",
    ]
    manifest_ids = {
        row.split("\t")[4]
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.startswith(f"nodes/{_NODE}/0055_")
    }
    assert manifest_ids == {f"node:{_NODE}:{name}" for name in files}


def test_the_shared_index_is_defined_identically_in_both_lineage_files() -> None:
    """Both files create idx_delegation_events_parent_correlation_id with IF NOT
    EXISTS. Whichever applies first wins, so the two definitions must agree or
    the live index would depend on apply order."""

    def index_definition(filename: str) -> str:
        match = re.search(
            r"CREATE INDEX IF NOT EXISTS idx_delegation_events_parent_correlation_id"
            r"\s+ON delegation_events \(parent_correlation_id\)"
            r"\s+WHERE parent_correlation_id IS NOT NULL;",
            _sql(filename),
        )
        assert match is not None, f"{filename}: shared index definition not found"
        return re.sub(r"\s+", " ", match.group(0))

    assert index_definition(_FILE) == index_definition(_SIBLING)
