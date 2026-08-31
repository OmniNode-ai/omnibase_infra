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

## Out of scope: two kinds of un-editable file

Both exemptions exist for the same underlying reason — this gate can only be
satisfied by EDITING a file, so it must not be pointed at a file nobody may
edit — but they are different facts and are read from different manifests
(OMN-17150), so neither can be claimed in place of the other.

1. **Operator-fenced ids** (``fenced-node-migrations.yaml``). Never applied, and
   held for a ruling. Read from the single-sourced manifest rather than
   restated, so this gate cannot drift from the runners it protects.

2. **Frozen bytes** (``shape-reconciliation-exemptions.yaml``). APPLIED on every
   lane, and un-editable for exactly that reason: the file's content sha256 is
   bound both by ``_ledger/application-migrations.tsv`` and by each lane's
   ``platform_catalog.schema_migrations`` row, so a byte change makes the next
   forward-migration run FATAL with ``conflicting migration checksum``.

Category 2 was created by OMN-17150 and is worth understanding, because before
it the two were conflated. ``node_projection_registration/0000`` had always
carried this gate's hazard and had always been un-editable; it was simply
invisible here because it happened to be fenced. When OMN-17150 released it from
the fence — so that a cold lane could create ``node_service_registry`` at all,
which the migration gate REQUIRES — the file did not change but the gate went
red on an id nobody may edit. Re-fencing it to quiet the gate is what
``node_pr_review_bot 001`` did, and is precisely the deadlock OMN-17150 closed.

Ticket: OMN-15376 (gate), OMN-17150 (frozen-bytes exemption)
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
    shape_reconciliation_exempt_ids,
)

pytestmark = [pytest.mark.unit]

_BEGIN = "-- ---- BEGIN OMN-15376 shape reconciliation:"
_END = "-- ---- END OMN-15376 shape reconciliation:"

_FENCED = fenced_migration_ids()
_FROZEN = shape_reconciliation_exempt_ids()
_OUT_OF_SCOPE = _FENCED | _FROZEN
_UNFENCED = [
    (migration_id, path)
    for migration_id, path in node_migration_files()
    if migration_id not in _OUT_OF_SCOPE
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
    # The frozen-bytes record MAY legitimately be empty (see the reader's own
    # docstring), so emptiness is not asserted here — only well-formedness, so
    # a malformed manifest cannot quietly exempt a garbage id.
    assert all(frozen.startswith("node:") for frozen in _FROZEN), sorted(_FROZEN)


# The exact frozen-bytes exemptions, pinned. Same discipline as the fence
# baseline's own pin: this hatch removes files from a real gate, so growing it
# must require editing a test, in a diff a reviewer reads, with the manifest's
# stated `frozen_by` reason next to it. An entry whose bytes are NOT actually
# bound by a live ledger row is an unfixed bug being relabelled.
_EXPECTED_FROZEN = frozenset(
    {"node:node_projection_registration:0000_create_node_service_registry.sql"}
)


def test_shape_exemptions_are_the_known_frozen_set() -> None:
    """The frozen-bytes list may not grow without a reviewer seeing it."""
    assert _FROZEN == _EXPECTED_FROZEN, (
        "docker/migrations/forward/shape-reconciliation-exemptions.yaml changed.\n"
        f"  found:    {sorted(_FROZEN)}\n"
        f"  expected: {sorted(_EXPECTED_FROZEN)}\n"
        "An exemption is only legitimate for a migration whose content sha256 "
        "is ALREADY bound by live ledger rows, so that editing it would make "
        "the next forward-migration run FATAL. A migration that has not applied "
        "anywhere yet must be FIXED, not exempted — its bytes are still free."
    )


def test_no_id_is_both_fenced_and_frozen() -> None:
    """The two manifests answer different questions and must stay disjoint.

    An id in both is a category error: a fenced migration is never applied, so
    its bytes cannot be bound by a ledger row, so 'frozen by the ledger' cannot
    be true of it. Overlap means one of the two records is wrong.
    """
    overlap = sorted(_FENCED & _FROZEN)
    assert not overlap, (
        "these ids claim BOTH an operator fence (never applied) and frozen "
        f"bytes (applied and checksum-bound), which cannot both hold: {overlap}"
    )


def test_every_frozen_id_names_a_real_vendored_migration() -> None:
    """Anti-typo: an exemption for a path that does not exist silences nothing
    visibly and would hide a genuine gap the day that file appears."""
    known = {migration_id for migration_id, _ in node_migration_files()}
    unknown = sorted(_FROZEN - known)
    assert not unknown, (
        f"these frozen-bytes exemptions name no vendored node migration: {unknown}"
    )


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
