# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19970 vendor identity for the delegation data_source migrations.

omnimarket's node_projection_delegation migration 0049 adds
``delegation_events.data_source`` ('real' | 'fixture', default 'real') so seeded
dev and demo rows are labelled, and 0050 re-creates
``projection_delegation_summary`` so measured savings exclude fixture rows. Both
are vendored here FIRST, ahead of the omnimarket source, per the node-migration
vendor-parity ordering. The real-Postgres proof of the column and the view lives
beside the source in omnimarket.

Failure modes pinned here:
  1. a vendored byte drifts from the ledger binding;
  2. a file is declared a class other than the class checker's own reading;
  3. a file fails the application-database SQL gate on a shipped profile, and
     that gate can still fail at all (positive control);
  4. 0050 replaces the summary view without re-setting ``security_invoker``.
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
        "0049_delegation_events_data_source.sql",
        "5c96cb3e2478e8079eed3944ff47a65ea16d921c0aa2d533166ad755d27cfd9f",
        "ALTER TABLE delegation_events",
    ),
    (
        "0050_delegation_summary_excludes_fixture_savings.sql",
        "b52806717c6e78ee789ecc4fd4c52b6e6201ed7cf3ae88545013b603cc76256e",
        "FROM delegation_events",
    ),
)
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
    """0049 adds a CHECK constraint and 0050 replaces a view: neither is additive
    by the checker's rules, so declaring either expand-only would let old code be
    redeployed over a schema the checker cannot prove it reads."""
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


def test_the_replaced_summary_view_keeps_security_invoker() -> None:
    """CREATE OR REPLACE VIEW resets a view's reloptions to the ones it names, so
    0050 must re-set security_invoker after replacing the view, before COMMIT."""
    sql = _sql("0050_delegation_summary_excludes_fixture_savings.sql")
    replaced = set(re.findall(r"CREATE OR REPLACE VIEW (\w+) AS", sql))
    reset = set(re.findall(r"ALTER VIEW (\w+) SET \(security_invoker = true\);", sql))
    assert replaced == {"projection_delegation_summary"}
    assert replaced <= reset
    assert sql.rindex("security_invoker") < sql.rindex("COMMIT;")
