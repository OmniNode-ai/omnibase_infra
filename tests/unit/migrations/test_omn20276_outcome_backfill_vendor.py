# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20276 vendor identity for the delegation_events outcome backfill.

omnimarket's node_projection_delegation migration 0051 rewrites the historical
delegation_events rows that read terminal_ok=false with a typed cause while
operational_outcome='completed' (the OMN-19559 contradiction), keeping each
row's prior values in an audit table; rollback_node_projection_delegation_0051
restores them and drops the table. Vendored here FIRST, ahead of the omnimarket
source, per the node-migration vendor-parity ordering. The real-Postgres proof
(idempotent, reversible, FORCE-RLS-safe) lives beside the source in omnimarket.

Failure modes pinned here:
  1. a vendored byte drifts from the ledger binding, or the rollback drifts from
     its recorded down execution;
  2. the file is declared a class other than the class checker's own reading;
  3. the file fails the application-database SQL gate on a shipped profile, and
     that gate can still fail at all (positive control);
  4. the down execution that lifts the rollback barrier stops being a PASS bound
     to these exact bytes.
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

# The falsifier is an undeclared schema, and the control asserts which rule
# fired (see test_omn19860_caller_lane_vendor.py for why a bare relation or the
# `public` schema is not a live control).
_UNDECLARED_SCHEMA_TARGET = "undeclared_topology_schema.delegation_events"
_UNDECLARED_SCHEMA_RULE = "unknown topology schema"

#: (file, sha256, the statement a broken copy retargets for the control)
_VENDORED = (
    (
        "0051_delegation_events_failed_outcome_backfill.sql",
        "4f4ab00d16e56d30992bd08951a910ec9fda6282156b91b875d090d8c43a40cc",
        "UPDATE delegation_events",
    ),
)
_ROLLBACK = "rollback/rollback_node_projection_delegation_0051.sql"
_ROLLBACK_SHA256 = "4f6dca2c04e62df7dae954b48b5a6b596d1cadb6c59623ad0273d2cbd2f7ef14"
_EXECUTIONS = _ROOT / "config" / "migration_down_executions.yaml"
_IDS = [entry[0] for entry in _VENDORED]


def _sql(filename: str) -> str:
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


@pytest.mark.parametrize(("filename", "sha256", "anchor"), _VENDORED, ids=_IDS)
def test_vendor_bytes_and_manifest_binding_are_exact(
    filename: str, sha256: str, anchor: str
) -> None:
    artifact_path = f"nodes/{_NODE}/{filename}"
    vendored = _FORWARD / "nodes" / _NODE / filename
    assert hashlib.sha256(vendored.read_bytes()).hexdigest() == sha256
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
        f"node:{_NODE}:{filename}",
        sha256,
    ]


@pytest.mark.parametrize(("filename", "sha256", "anchor"), _VENDORED, ids=_IDS)
def test_declared_class_is_forward_only_as_the_checker_reads_it(
    filename: str, sha256: str, anchor: str
) -> None:
    """0051 rewrites rows: not additive by the checker's rules, so it is
    forward-only, a barrier lifted only by its recorded down execution."""
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{filename}"] == "forward-only"
    assert _class_checker().destructive_findings(_sql(filename)) != []


@pytest.mark.parametrize(("filename", "sha256", "anchor"), _VENDORED, ids=_IDS)
def test_migration_passes_the_application_database_sql_gate(
    filename: str, sha256: str, anchor: str
) -> None:
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _sql(filename)
    for profile in _PROFILES:
        violations = lint_application_database_sql(sql, load_topology_profile(profile))
        assert violations == (), f"{profile}: {violations}"


@pytest.mark.parametrize(("filename", "sha256", "anchor"), _VENDORED, ids=_IDS)
def test_the_linter_is_live_positive_control(
    filename: str, sha256: str, anchor: str
) -> None:
    """A zero from the linter means something only if it can return non-zero,
    on every profile the gate above clears."""
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _sql(filename)
    keyword = anchor.split(" delegation_events")[0]
    broken = sql.replace(anchor, f"{keyword} {_UNDECLARED_SCHEMA_TARGET}", 1)
    assert broken != sql, f"the {anchor!r} anchor is no longer present in {filename}"
    for profile in _PROFILES:
        violations = lint_application_database_sql(
            broken, load_topology_profile(profile)
        )
        assert any(_UNDECLARED_SCHEMA_RULE in violation for violation in violations), (
            f"{profile}: retargeting at {_UNDECLARED_SCHEMA_TARGET!r} did not raise "
            f"{_UNDECLARED_SCHEMA_RULE!r}; got {violations}"
        )


def test_down_execution_is_a_pass_bound_to_these_bytes() -> None:
    forward = _FORWARD / "nodes" / _NODE / _VENDORED[0][0]
    down = _ROOT / "docker" / "migrations" / _ROLLBACK
    assert hashlib.sha256(down.read_bytes()).hexdigest() == _ROLLBACK_SHA256
    executions = yaml.safe_load(_EXECUTIONS.read_text(encoding="utf-8"))["executions"]
    record = next(
        e
        for e in executions
        if e["migration"] == f"forward/nodes/{_NODE}/{_VENDORED[0][0]}"
    )
    assert record["down_script"] == _ROLLBACK
    assert record["outcome"] == "PASS"
    assert record["forward_sha256"] == hashlib.sha256(forward.read_bytes()).hexdigest()
    assert record["down_sha256"] == _ROLLBACK_SHA256
