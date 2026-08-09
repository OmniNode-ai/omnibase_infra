# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Static gate: a table declared by BOTH a flat and a node migration must not
silently drift between the two declarations.

## The class this closes

OMN-15376 shipped `tests/ci/test_node_migration_shape_reconciliation.py`,
which reconciles a table's fresh-create path against its own
drifted-pre-existing path -- but ONLY within
``docker/migrations/forward/nodes/``. It says nothing about a table declared
a SECOND time by a top-level flat migration
(``docker/migrations/forward/*.sql``, applied against the ``omnibase_infra``
DB) that names the exact same table. Two known instances of exactly that
class each cost a full deploy cycle before anything caught them
(``llm_cost_aggregates`` / OMN-15376, ``baselines_comparisons`` / OMN-15302).
A full overlap computation over dev finds **14** tables declared by both
corpora; this gate is the binding between them that neither producer's own
reconciliation gate can see.

## What this asserts

For every table name this repo's migrations declare via a guarded
``CREATE TABLE IF NOT EXISTS`` in BOTH corpora:

* the column set and normalized column type must match the ledger
  (``docker/migrations/forward/flat-node-shape-parity.yaml``) entry for that
  table -- ``status: identical`` requires the two sides to actually agree
  (columns are re-derived live from the SQL, never trusted from the ledger),
  ``status: accepted_divergence`` requires a non-empty ``reason`` and that
  the two sides are STILL actually divergent (a stale acceptance whose
  shapes have since converged is also a failure -- flip it to ``identical``).
* the ledger's table set matches the LIVE overlap exactly, in both
  directions: a newly-introduced dual-producer table with no ledger entry
  fails closed (AC2 -- a table cannot dodge review by omission), and a
  ledger entry for a table that is no longer dual-produced fails just as
  loudly (a stale record is not free to keep around).

This is the STATIC half only, matching the OMN-15376 sibling gate's own
split: no database is touched, no execution proof is claimed.

Ticket: OMN-15384
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from tests.helpers.util_migration_shape import (
    diff_column_shapes,
    flat_migration_files,
    node_migration_files,
    table_column_shapes,
)

pytestmark = [pytest.mark.unit]

LEDGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "docker"
    / "migrations"
    / "forward"
    / "flat-node-shape-parity.yaml"
)

# The 14-table overlap OMN-15384's own audit found. Pinned here (not just
# derived live) so a corpus edit that shrinks the overlap below what was
# actually audited is visible as a assertion diff, not a quietly-smaller
# passing set.
_AUDITED_OVERLAP = frozenset(
    {
        "agent_routing_decisions",
        "baselines_breakdown",
        "baselines_comparisons",
        "baselines_trend",
        "capability_scores",
        "evidence_correlation_trace_projection",
        "evidence_dashboard_projection",
        "evidence_readiness_aggregate_projection",
        "llm_call_metrics",
        "llm_cost_aggregates",
        "llm_routing_decisions",
        "savings_estimates",
        "session_outcomes",
        "swarm_runs",
    }
)


def _load_ledger() -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(LEDGER_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict) and isinstance(raw.get("tables"), dict), (
        f"{LEDGER_PATH} must be a mapping with a top-level `tables:` mapping"
    )
    return cast("dict[str, dict[str, Any]]", raw["tables"])


def _live_overlap() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    flat_shapes = table_column_shapes(flat_migration_files())
    node_shapes = table_column_shapes(node_migration_files())
    return flat_shapes, node_shapes


def test_the_corpus_is_non_empty_and_the_ledger_matches_the_live_overlap() -> None:
    """Anti-vacuity + AC2: the ledger cannot silently drift from the corpus.

    A table can't dodge this gate by never getting a ledger entry (missing ==
    fail), and a ledger entry can't survive after its table stops being
    dual-produced (stale == fail) -- so deleting one side's declaration to
    "resolve" a divergence is not a way to make this test quietly pass; it
    surfaces as a stale-entry failure demanding the ledger be updated too.
    """
    flat_shapes, node_shapes = _live_overlap()
    live_overlap = set(flat_shapes) & set(node_shapes)
    ledger = _load_ledger()
    ledger_tables = set(ledger)

    assert len(live_overlap) >= 14, sorted(live_overlap)
    assert live_overlap >= _AUDITED_OVERLAP, sorted(_AUDITED_OVERLAP - live_overlap)

    missing_from_ledger = live_overlap - ledger_tables
    assert not missing_from_ledger, (
        f"{sorted(missing_from_ledger)} are declared by BOTH a flat and a node "
        f"migration but have no entry in {LEDGER_PATH}. Add one with "
        f"status: identical (if the shapes agree) or status: accepted_divergence "
        f"plus a reason (if they legitimately don't) -- see the module docstring."
    )
    stale_ledger_entries = ledger_tables - live_overlap
    assert not stale_ledger_entries, (
        f"{sorted(stale_ledger_entries)} have a {LEDGER_PATH} entry but are no "
        f"longer declared by both corpora (one side's CREATE TABLE was removed "
        f"or renamed). Remove the stale entry."
    )


@pytest.mark.parametrize("table", sorted(_AUDITED_OVERLAP))
def test_dual_producer_table_shape_matches_the_ledger_disposition(table: str) -> None:
    """Per-table enforcement: identical stays identical, divergence stays named."""
    flat_shapes, node_shapes = _live_overlap()
    ledger = _load_ledger()

    entry = ledger.get(table)
    assert entry is not None, f"{table}: no ledger entry (see the overlap test)"
    status = entry.get("status")
    assert status in ("identical", "accepted_divergence"), (
        f"{table}: unknown ledger status {status!r}"
    )

    diff = diff_column_shapes(flat_shapes[table], node_shapes[table])

    if status == "identical":
        assert not diff, (
            f"{table} is recorded `status: identical` in {LEDGER_PATH} but its "
            f"flat and node declarations have drifted apart (OMN-15384 class):\n"
            f"{diff.describe(table=table)}\n"
            f"Either converge the SQL, or change the ledger entry to "
            f"accepted_divergence with a reason."
        )
    else:
        reason = entry.get("reason", "")
        assert isinstance(reason, str) and reason.strip(), (
            f"{table}: status: accepted_divergence requires a non-empty `reason` "
            f"in {LEDGER_PATH}"
        )
        assert diff, (
            f"{table} is recorded `status: accepted_divergence` in {LEDGER_PATH} "
            f"but its flat and node declarations now agree -- the entry is "
            f"stale. Flip it to `status: identical` (and drop `reason`)."
        )


def test_diff_column_shapes_detects_every_drift_class() -> None:
    """The pure comparator's branches are all exercised, not just imported.

    Without this, `diff_column_shapes` could be broken (e.g. always return an
    empty diff) and every case above would still pass -- a checker whose
    branches never fire is the same theater the OMN-15639 sentinel's own
    synthetic-input test exists to rule out.
    """
    identical = diff_column_shapes(
        {"a": "UUID", "b": "TEXT"}, {"a": "UUID", "b": "TEXT"}
    )
    assert not identical, identical

    only_flat = diff_column_shapes({"a": "UUID", "b": "TEXT"}, {"a": "UUID"})
    assert only_flat.only_flat == ("b",)
    assert not only_flat.only_node
    assert not only_flat.type_diff

    only_node = diff_column_shapes({"a": "UUID"}, {"a": "UUID", "b": "TEXT"})
    assert only_node.only_node == ("b",)
    assert not only_node.only_flat
    assert not only_node.type_diff

    type_diff = diff_column_shapes({"a": "REAL"}, {"a": "DOUBLEPRECISION"})
    assert type_diff.type_diff == (("a", "REAL", "DOUBLEPRECISION"),)
    assert not type_diff.only_flat
    assert not type_diff.only_node

    described = type_diff.describe(table="t")
    assert "t:" in described
    assert "type diff a" in described
