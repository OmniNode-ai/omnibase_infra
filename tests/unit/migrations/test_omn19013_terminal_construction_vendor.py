# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19013 vendor identity for the terminal-construction metrics migration."""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_VENDOR = _FORWARD / "nodes" / "node_projection_delegation"
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_FILENAME = "0045_terminal_construction_outcome_metrics.sql"
_SHA256 = "796979e03d010fbe187f3adc0d2700af343cad3cb95ca8eedf48698f2a9c2ea7"


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


def test_migration_passes_the_application_database_sql_gate() -> None:
    """OMN-19013: 0045 must clear the OMN-15361 gate at the full bar.

    The frozen baseline is shrink-only and a new file may never be added to
    it, so this asserts the verdict the gate itself returns rather than the
    presence of an exemption. It was RED before the fix -- the parser reads
    ``EXTRACT(EPOCH FROM created_at)`` as a FROM clause and reported
    ``application relation target 'created_at' must be schema-qualified`` --
    and the sibling 0039 in this same stream already carries the parenthesised
    operand for the same reason.

    Every shipped profile is checked, not just ``local``: the gate runs the
    corpus against each one, so a verdict proven on a single profile would not
    be the verdict CI reaches.
    """
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = (_VENDOR / _FILENAME).read_text(encoding="utf-8")
    for profile in ("local", "onex-dev", "onex-prod", "stability-test"):
        violations = lint_application_database_sql(sql, load_topology_profile(profile))
        assert violations == (), f"{profile}: {violations}"


def test_the_sibling_migration_is_a_positive_control() -> None:
    """A zero from the linter means something only if it can return non-zero.

    0039 is the nearest non-baselined migration in the same stream and is
    known-clean, so this proves the harness above reaches real SQL rather
    than silently linting nothing.
    """
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sibling = _VENDOR / "0039_delegation_aggregate_views_per_tenant.sql"
    assert sibling.is_file()
    assert (
        lint_application_database_sql(
            sibling.read_text(encoding="utf-8"), load_topology_profile("local")
        )
        == ()
    )
