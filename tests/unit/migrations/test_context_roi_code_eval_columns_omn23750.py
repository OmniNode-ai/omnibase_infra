# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The context-ROI code-eval columns migration is vendored and registered (OMN-23750).

omnimarket's node_projection_context_roi adds the per-try columns of a code-eval
trial to ``context_roi_scores`` in its migration 004. The infrastructure copy
must exist before the omnimarket change that writes the columns merges, be
registered in the migration ledger with its own digest, and be classed
expand-only (additive columns with defaults, never a drop or rewrite).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_NODE_DIR = _ROOT / "docker/migrations/forward/nodes/node_projection_context_roi"
_FILE = "004_add_code_eval_trial_columns.sql"
_LEDGER = _ROOT / "docker/migrations/forward/_ledger/application-migrations.tsv"
_CLASSES = _ROOT / "config/migration_classes.yaml"
_COLUMNS = (
    "check_verdict",
    "unscorable_source",
    "row_kind",
    "try_end",
    "suite_digest",
    "size_band",
    "sequence_id",
    "try_ordinal",
    "diff_hash",
    "max_turns",
    "try_timeout_s",
    "generation_s",
    "env_sync_s",
    "check_s",
)


@pytest.mark.unit
def test_the_migration_is_vendored() -> None:
    assert (_NODE_DIR / _FILE).is_file()


@pytest.mark.unit
@pytest.mark.parametrize("column", _COLUMNS)
def test_the_migration_adds_each_column_with_a_guard(column: str) -> None:
    text = (_NODE_DIR / _FILE).read_text(encoding="utf-8")
    assert re.search(rf"ADD COLUMN IF NOT EXISTS {column}\b", text)


@pytest.mark.unit
def test_the_migration_only_adds() -> None:
    text = (_NODE_DIR / _FILE).read_text(encoding="utf-8").upper()
    for word in ("DROP", "DELETE", "UPDATE", "TRUNCATE"):
        assert word not in text


@pytest.mark.unit
def test_the_ledger_row_carries_the_file_digest() -> None:
    digest = hashlib.sha256((_NODE_DIR / _FILE).read_bytes()).hexdigest()
    rows = [
        line.split("\t")
        for line in _LEDGER.read_text(encoding="utf-8").splitlines()
        if f"node_projection_context_roi/{_FILE}" in line
    ]
    assert len(rows) == 1
    assert rows[0][-1] == digest


@pytest.mark.unit
def test_the_class_list_declares_the_file_expand_only() -> None:
    declared = [
        line
        for line in _CLASSES.read_text(encoding="utf-8").splitlines()
        if f"node_projection_context_roi/{_FILE}" in line
    ]
    assert len(declared) == 1
    assert "expand-only" in declared[0]
