# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""consumer_flow_windows is vacuumed on its own scale (OMN-17427, C28).

The consumer-flow exposure reads the newest window of every pair through
index-only skip-scan probes, which stay index-only only on pages VACUUM has
marked all-visible. At the default 20% thresholds an 11.5M-row table waited for
2.3M changes, and the cold read took 24 s against the projection API's 15 s
command timeout. Migration 0007 sets per-table thresholds; these tests pin what
it may and may not change.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NODE_DIR = (
    _REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_consumer_flow"
)
_MIGRATION = _NODE_DIR / "0007_autovacuum_consumer_flow_windows.sql"
_LEDGER = (
    _REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "_ledger"
    / "application-migrations.tsv"
)
_SET = re.compile(
    r"ALTER\s+TABLE\s+omninode_internal\.consumer_flow_windows\s+SET\s*\((?P<options>[^)]*)\)",
    re.IGNORECASE,
)
_EXPECTED = {
    "autovacuum_vacuum_scale_factor",
    "autovacuum_vacuum_insert_scale_factor",
    "autovacuum_analyze_scale_factor",
}


def _statements() -> list[str]:
    sql = "\n".join(
        line
        for line in _MIGRATION.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )
    return [statement.strip() for statement in sql.split(";") if statement.strip()]


def _options() -> dict[str, float]:
    statements = _statements()
    assert len(statements) == 1, statements
    match = _SET.fullmatch(statements[0])
    assert match is not None, statements[0]
    options: dict[str, float] = {}
    for pair in match["options"].split(","):
        name, _, value = pair.partition("=")
        options[name.strip().lower()] = float(value.strip())
    return options


def test_the_migration_only_sets_the_three_autovacuum_thresholds() -> None:
    assert set(_options()) == _EXPECTED


def test_every_threshold_is_well_under_the_default() -> None:
    """2% of the table, against the 20% default that let the read go cold."""
    for name, value in _options().items():
        assert 0 < value <= 0.05, f"{name}={value}"


def test_the_migration_is_declared_in_the_application_ledger() -> None:
    rows = [
        line.split("\t")
        for line in _LEDGER.read_text(encoding="utf-8").splitlines()
        if line.startswith("nodes/node_projection_consumer_flow/0007_")
    ]
    assert len(rows) == 1, rows
    assert rows[0][4] == (
        "node:node_projection_consumer_flow:0007_autovacuum_consumer_flow_windows.sql"
    )
