# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19550 vendor identity for the session-content projection migration.

omnimarket's node_projection_session_content migration 0001 creates
``omninode_internal.session_content`` and grants its projection writer access
through ``omninode_runtime``. It is vendored here FIRST, ahead of the
omnimarket source, per the node-migration vendor-parity ordering.
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
_VENDOR = _FORWARD / "nodes" / "node_projection_session_content"
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"
_FILENAME = "0001_create_session_content.sql"
_SHA256 = "a4246a5acd3c33c50f87abb32ff84e05a6e59b70f12986cfed2b620da27835a1"


def _sql() -> str:
    return (_VENDOR / _FILENAME).read_text(encoding="utf-8")


def _manifest_rows() -> dict[str, list[str]]:
    return {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }


def test_vendor_bytes_and_manifest_binding_are_exact() -> None:
    artifact_path = f"nodes/node_projection_session_content/{_FILENAME}"
    assert hashlib.sha256((_VENDOR / _FILENAME).read_bytes()).hexdigest() == _SHA256
    assert _manifest_rows()[artifact_path] == [
        artifact_path,
        "node:node_projection_session_content",
        "node:node_projection_session_content",
        "omninode_internal",
        f"node:node_projection_session_content:{_FILENAME}",
        _SHA256,
    ]


def test_migration_is_declared_expand_only_and_matches_the_vendored_bytes() -> None:
    """The class line covers the table creation and its runtime-role grants."""
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert (
        classes[f"forward/nodes/node_projection_session_content/{_FILENAME}"]
        == "expand-only"
    )

    statements = "\n".join(
        line for line in _sql().splitlines() if not line.lstrip().startswith("--")
    )
    assert re.search(
        r"CREATE TABLE IF NOT EXISTS omninode_internal\.session_content\s*\(",
        statements,
    )
    assert "GRANT USAGE ON SCHEMA omninode_internal TO omninode_runtime;" in statements
    assert (
        "GRANT SELECT, INSERT, UPDATE ON omninode_internal.session_content "
        "TO omninode_runtime;"
    ) in statements
    assert not re.search(
        r"GRANT\s+[^;]*\bDELETE\b[^;]*\bTO\s+omninode_runtime\b",
        statements,
        re.IGNORECASE,
    )


def test_migration_passes_the_application_database_sql_gate() -> None:
    """0001 clears the OMN-15361 gate on every shipped profile."""
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _sql()
    for profile in ("local", "onex-dev", "onex-prod", "stability-test"):
        violations = lint_application_database_sql(sql, load_topology_profile(profile))
        assert violations == (), f"{profile}: {violations}"


def test_the_linter_is_live_positive_control() -> None:
    """Retargeting the table to the retired tenant schema must be refused."""
    from omnibase_infra.topology.application_database import load_topology_profile
    from omnibase_infra.validation.application_database_domain_enforcement import (
        lint_application_database_sql,
    )

    sql = _sql()
    broken = sql.replace("omninode_internal.session_content", "tenant.session_content")
    assert broken != sql
    assert lint_application_database_sql(broken, load_topology_profile("local")) != ()
