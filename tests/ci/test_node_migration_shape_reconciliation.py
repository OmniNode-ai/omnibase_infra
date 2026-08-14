# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Static gate: every guarded CREATE TABLE must reconcile its own shape.

## The class this closes

``CREATE TABLE IF NOT EXISTS t (...)`` SILENTLY NO-OPS when a table named ``t``
already exists with a DIFFERENT shape. The statements that follow it in the same
file are not so forgiving: ``CREATE INDEX IF NOT EXISTS ... ON t (col)`` guards
the index NAME, not the COLUMN, so it raises ``column "col" does not exist`` and
``ON_ERROR_STOP=1`` kills the whole migration Job at that point.

Because the runner halts at the FIRST failure, instances of this class surface
strictly one per deploy cycle. Two of them each cost a full cycle to discover:

* OMN-15376 — ``llm_cost_aggregates.aggregation_key``, deploy-onex-dev run
  30418878385, ``0001_create_llm_cost_aggregates.sql:64``.
* OMN-15302 — ``baselines_comparisons.snapshot_id``,
  ``0001_create_baselines_tables.sql:76``.

## What this asserts

For every vendored node migration, every column declared by a
``CREATE TABLE IF NOT EXISTS`` must ALSO be covered by a guarded
``ALTER TABLE <t> ADD COLUMN IF NOT EXISTS <col> ...`` in the SAME file. Those
adds are no-ops on the fresh-create path and converge a drifted pre-existing
table on the drifted path, so both paths end at the same schema without
dropping or recreating anything.

This is the STATIC half of the gate and runs with no database. The EXECUTION
half — RED against the drifted shape, GREEN after, and schema equality between
the two paths — is
``tests/integration/migrations/test_node_migration_shape_drift_omn15376.py``.

## Fenced ids

The operator-fenced ids (OMN-14974 / OMN-15313 / OMN-15335) are exempt: they are
never applied and MUST NOT be edited to satisfy a gate. The exemption list is
READ FROM ``scripts/run-forward-migrations.sh`` rather than restated, so this
gate cannot drift from the runner it is protecting. Note that the runner in this
repo fences SIX ids while omninode_infra's k8s Job runner fences SEVEN — that
parity gap is a separate finding (see the execution suite's ``_K8S_ONLY_FENCED``
note), not something this gate papers over.

Ticket: OMN-15376
"""

from __future__ import annotations

import re

import pytest

from tests.helpers.util_migration_shape import (
    fenced_migration_ids,
    guarded_create_tables,
    mask_literals,
    node_migration_files,
    reconciled_columns,
)

pytestmark = [pytest.mark.unit]

_BEGIN = "-- ---- BEGIN OMN-15376 shape reconciliation:"
_END = "-- ---- END OMN-15376 shape reconciliation:"

_FENCED = fenced_migration_ids()
_UNFENCED = [
    (migration_id, path)
    for migration_id, path in node_migration_files()
    if migration_id not in _FENCED
]
# Only files that actually declare a guarded CREATE TABLE carry the obligation.
# Parametrising over these (rather than skipping the rest) keeps every emitted
# case load-bearing: a skip is not a pass.
_CASES = [
    (migration_id, path)
    for migration_id, path in _UNFENCED
    if guarded_create_tables(path.read_text(encoding="utf-8"))
]


def test_the_corpus_is_non_empty_and_the_fence_resolved() -> None:
    """Anti-vacuity: an empty corpus or an unread fence would pass everything.

    The fence COUNT is deliberately not pinned here — that is
    ``tests/scripts/test_node_migration_fence_parity.py``'s job, and duplicating
    the number would create a second place to update. This only proves the fence
    was actually parsed out of the runner and looks like node ids.
    """
    assert len(_UNFENCED) >= 60, len(_UNFENCED)
    assert len(_CASES) >= 40, len(_CASES)
    assert _FENCED, "operator fence parsed as empty — the reader is broken"
    assert all(fenced.startswith("node:") for fenced in _FENCED), sorted(_FENCED)


@pytest.mark.parametrize(
    ("migration_id", "path"), _CASES, ids=[case[0] for case in _CASES]
)
def test_guarded_create_table_reconciles_every_declared_column(
    migration_id: str, path: object
) -> None:
    """Each declared column must have a guarded ADD COLUMN in the same file."""
    sql = path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    tables = guarded_create_tables(sql)
    assert tables, migration_id

    missing: dict[str, list[str]] = {}
    for table in tables:
        covered = reconciled_columns(sql, table)
        gaps = [
            column.name.strip('"')
            for column in table.columns
            if column.name.strip('"').lower() not in covered
        ]
        if gaps:
            missing[table.bare_name] = gaps

    assert not missing, (
        f"{migration_id}: CREATE TABLE IF NOT EXISTS no-ops against a drifted "
        f"pre-existing table, and these declared columns are never reconciled, "
        f"so the next column-dependent statement fails the whole deploy "
        f"(OMN-15376 class): {missing}. Add, immediately after the CREATE TABLE, "
        f"one 'ALTER TABLE <t> ADD COLUMN IF NOT EXISTS <col> <type> [DEFAULT ...];' "
        f"per declared column."
    )


@pytest.mark.parametrize(
    ("migration_id", "path"), _UNFENCED, ids=[case[0] for case in _UNFENCED]
)
def test_reconciliation_never_reintroduces_a_drop(
    migration_id: str, path: object
) -> None:
    """The reconciliation idiom must not smuggle in a data-destroying DROP.

    ``0002_realign_child_tables_to_producer_schema.sql`` legitimately carries a
    ``DROP TABLE IF EXISTS`` under an explicit, ticketed zero-rows ruling
    (OMN-14513). What must never appear is a DROP inside an OMN-15376
    reconciliation block, because that block runs on tables whose row count is
    unknown.
    """
    sql = path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    for block in re.findall(re.escape(_BEGIN) + r".*?" + re.escape(_END), sql, re.S):
        # Comments are masked out first: the block's own header says "no DROP,
        # no recreate, no TRUNCATE", and matching that prose would be a
        # self-inflicted false positive.
        upper = mask_literals(block).upper()
        for forbidden in ("DROP TABLE", "DROP COLUMN", "TRUNCATE", "DELETE FROM"):
            assert forbidden not in upper, (
                f"{migration_id}: '{forbidden}' inside an OMN-15376 "
                f"reconciliation block. That block runs against tables whose row "
                f"count is unknown; converge the shape, never destroy the data."
            )
