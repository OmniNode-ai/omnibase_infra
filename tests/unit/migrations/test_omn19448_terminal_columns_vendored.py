# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19448: every column delegation_events copies from the terminal is vendored.

The projection writer in omnimarket copies the terminal's own cause, operational
outcome and content verdict onto the row (AC1) and writes the trace, routing,
stop reason and timing columns from the terminal (AC2). On a lane deployed from
omnibase_infra those columns exist only if the node migrations that add them are
vendored here with a ledger row whose checksum is the file's. Failure modes this
is written against:

- a column the writer copies is added by no vendored migration, so the INSERT
  built from the row's keys fails on the lane;
- a vendored file has no ledger row, or its checksum no longer matches the file;
- a column gains a default or NOT NULL, so a terminal that does not carry the
  value is stored as a value it never sent instead of NULL;
- terminal_failure_cause gains a CHECK list, so a cause added to the vocabulary
  later (refused_before_dispatch is one) is refused and the row lost (AC3).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FORWARD = ROOT / "docker/migrations/forward"
NODE_DIR = FORWARD / "nodes/node_projection_delegation"
LEDGER = FORWARD / "_ledger/application-migrations.tsv"

# The migration that adds each column, and its SQL type.
TERMINAL_COPIED_COLUMNS: dict[str, tuple[str, str]] = {
    "terminal_failure_cause": ("0029_delegation_terminal_failure_cause.sql", "TEXT"),
    "operational_outcome": ("0045_terminal_construction_outcome_metrics.sql", "TEXT"),
    "content_verdict": ("0045_terminal_construction_outcome_metrics.sql", "TEXT"),
}
NEW_TERMINAL_COLUMNS: dict[str, tuple[str, str]] = {
    "trace_id": ("0052_delegation_events_trace_and_routing.sql", "TEXT"),
    "routed_model": ("0052_delegation_events_trace_and_routing.sql", "TEXT"),
    "answering_backend": ("0052_delegation_events_trace_and_routing.sql", "TEXT"),
    "finish_reason": ("0053_delegation_events_finish_reason_truncated.sql", "TEXT"),
    "truncated": ("0053_delegation_events_finish_reason_truncated.sql", "BOOLEAN"),
    "requested_model": (
        "0058_delegation_events_requested_model_and_timing.sql",
        "TEXT",
    ),
    "queue_wait_ms": (
        "0058_delegation_events_requested_model_and_timing.sql",
        "INTEGER",
    ),
    "execution_ms": (
        "0058_delegation_events_requested_model_and_timing.sql",
        "INTEGER",
    ),
}


def _ledger_checksums() -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 6:
            rows[fields[0]] = fields[-1]
    return rows


def _added_columns() -> dict[str, list[tuple[str, str]]]:
    """Every delegation_events column a vendored migration adds, with its clause."""
    added: dict[str, list[tuple[str, str]]] = {}
    pattern = re.compile(
        r"ADD COLUMN IF NOT EXISTS\s+(\w+)\s+([^,;\n]+)", flags=re.IGNORECASE
    )
    for path in sorted(NODE_DIR.glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        for statement in re.split(r";", sql):
            if not re.search(r"ALTER TABLE\s+delegation_events\b", statement):
                continue
            for name, clause in pattern.findall(statement):
                added.setdefault(name, []).append((path.name, clause.strip()))
    return added


def _assert_vendored(columns: dict[str, tuple[str, str]]) -> None:
    added = _added_columns()
    ledger = _ledger_checksums()
    for column, (filename, sql_type) in columns.items():
        assert column in added, f"{column}: no vendored migration adds it"
        sources = [name for name, _ in added[column]]
        assert filename in sources, f"{column}: added by {sources}, not {filename}"
        for _, clause in added[column]:
            words = clause.upper().split()
            assert words[0] == sql_type, f"{column}: {clause}"
            assert "DEFAULT" not in words, f"{column} must stay NULL when unsent"
            assert "NOT" not in words, f"{column} must stay nullable"
        relative = f"nodes/node_projection_delegation/{filename}"
        digest = hashlib.sha256((FORWARD / relative).read_bytes()).hexdigest()
        assert ledger.get(relative) == digest, f"{relative}: ledger checksum"


def test_the_columns_the_terminal_is_copied_into_are_vendored_and_nullable() -> None:
    """AC1: cause, operational_outcome and content_verdict exist with no default."""
    _assert_vendored(TERMINAL_COPIED_COLUMNS)


def test_the_new_terminal_columns_arrive_through_vendored_migrations() -> None:
    """AC2: all eight new columns are added by vendored, ledgered migrations."""
    _assert_vendored(NEW_TERMINAL_COLUMNS)


def test_terminal_failure_cause_admits_every_vocabulary_member() -> None:
    """AC3: no vendored migration restricts the cause column to a value list."""
    for path in sorted(NODE_DIR.glob("*.sql")):
        sql = re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))
        for statement in sql.split(";"):
            if "terminal_failure_cause" not in statement:
                continue
            assert not re.search(
                r"CHECK\s*\([^)]*terminal_failure_cause", statement, re.IGNORECASE
            ), f"{path.name} restricts terminal_failure_cause"
