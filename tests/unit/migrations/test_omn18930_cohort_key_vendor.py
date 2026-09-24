# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18930 vendor identity for the delegation_events cohort-key migration.

K3 of OMN-18925, projection half. omnimarket's node_projection_delegation
migration 0046 adds three nullable columns to ``delegation_events`` so every
row carries the delegation cohort key its terminal carried. It is vendored here
FIRST, ahead of the omnimarket source, per the node-migration vendor-parity
ordering. The real-Postgres proof of the columns and both writers lives beside
the source in omnimarket.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_VENDOR = _FORWARD / "nodes" / "node_projection_delegation"
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_FILENAME = "0046_delegation_events_cohort_key.sql"
_SHA256 = "933069368adc0ce83a627a859d5360e7970fba91bdfbd73848fde3820199dd1c"


def _manifest_rows() -> dict[str, list[str]]:
    return {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }


def test_vendor_bytes_and_manifest_binding_are_exact() -> None:
    artifact_path = f"nodes/node_projection_delegation/{_FILENAME}"
    assert hashlib.sha256((_VENDOR / _FILENAME).read_bytes()).hexdigest() == _SHA256
    assert _manifest_rows()[artifact_path] == [
        artifact_path,
        "node:node_projection_delegation",
        "node:node_projection_delegation",
        "tenant",
        f"node:node_projection_delegation:{_FILENAME}",
        _SHA256,
    ]


def test_migration_is_declared_expand_only_and_is_purely_additive() -> None:
    """The class line matches the bytes: three nullable columns, nothing else."""
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/node_projection_delegation/{_FILENAME}"] == (
        "expand-only"
    )
    sql = (_VENDOR / _FILENAME).read_text(encoding="utf-8")
    statements = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    added = re.findall(r"ADD COLUMN IF NOT EXISTS (\w+) (\w+)", statements)
    assert added == [
        ("cohort_key", "JSONB"),
        ("cohort_key_sha256", "TEXT"),
        ("cohort_key_refusal", "TEXT"),
    ]
    for forbidden in ("NOT NULL", "DROP", "UPDATE", "DELETE", "CREATE OR REPLACE"):
        assert forbidden not in statements.upper(), forbidden


def test_migration_passes_the_application_database_sql_gate() -> None:
    """0046 clears the OMN-15361 gate on every shipped profile."""
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / _FILENAME).read_text(encoding="utf-8")
    for profile in ("local", "onex-dev", "onex-prod", "stability-test"):
        violations = lint_application_database_sql(sql, load_topology_profile(profile))
        assert violations == (), f"{profile}: {violations}"


def test_the_linter_is_live_positive_control() -> None:
    """A zero from the linter means something only if it can return non-zero.

    The same file with its target unqualified away into a public-schema write
    of an unknown relation must not lint clean, or the test above proves
    nothing.
    """
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / _FILENAME).read_text(encoding="utf-8")
    broken = sql.replace("ALTER TABLE delegation_events", "ALTER TABLE public.pg_class")
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()
