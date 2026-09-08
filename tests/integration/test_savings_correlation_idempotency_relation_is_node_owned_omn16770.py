# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The savings-correlation write set must be relations this node owns [OMN-16770].

OMN-16770's whole history is one handler reading a relation nothing in its own
migration tree creates. Twice:

* First it read ``omninode_internal.savings_injection_signals`` from the wrong
  DATABASE — ``UndefinedTableError`` on every 60s tick from 2026-08-23.
* Then, re-bound to the right database, it read ``savings_estimates`` over the
  wrong BINDING — a TENANT relation under ``ENABLE`` + ``FORCE ROW LEVEL
  SECURITY`` (``node_projection_savings/081``) that this node neither owns nor
  writes, over a pool whose principal ``omninode_runtime`` is NOSUPERUSER /
  NOBYPASSRLS / non-owner (OMN-16843). Row-level security fails OPEN from the
  caller's side, so the honest outcomes were an ``InsufficientPrivilegeError``
  once a minute or — with the "obvious" ``GRANT SELECT`` — an anti-join
  silently inverted to universally true. The OMN-16770 seam refuses instead,
  and refused 480 times in four hours on the ``.201`` dev lane.

Both are the same defect: ``HandlerSavingsCorrelation`` takes a RAW injected
``asyncpg`` pool rather than a contract-declared table operation, so its reads
never pass through ``ProjectionTableOperation._assert_read_declared`` — the
seam that fails closed on an undeclared relation. Nothing could refuse them at
declaration time, so both were only ever going to surface at runtime, on a
lane, once a minute.

This is OMN-16770 AC4 ("the disagreement cannot silently recur") for the write
side, asserted statically so it fails in CI instead of on a lane. The rule is
deliberately narrow enough to need no allowlist:

* Every relation this handler WRITES must be created by a migration in this
  node's OWN tree. A node that writes a relation it does not own is the defect.
* The relation the OMN-16770 seam probes for idempotency — the one the
  readiness anti-join reads — must likewise be node-owned. That is the specific
  relation whose cross-domain read produced this ticket.
* READS are not constrained. ``llm_call_metrics`` and ``session_outcomes`` are
  legitimately cross-repo, INTERNAL-domain, granted to this pool's principal,
  and carry no row-level security — the same read-only pattern
  ``HandlerBaselinesBatchCompute`` uses. Constraining them would be a rule
  nobody could satisfy, and a rule nobody can satisfy acquires an allowlist.

Nothing here is hardcoded to a relation name: the write set is parsed from the
handler's own SQL and the owned set from the node's own migration tree, so a
handler that starts writing somewhere new fails this test without anyone
remembering to update it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_savings_estimation_compute.handlers.handler_savings_correlation import (
    IDEMPOTENCY_RELATION,
    TENANT_GUC,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
HANDLER = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_savings_estimation_compute"
    / "handlers"
    / "handler_savings_correlation.py"
)
NODE_MIGRATIONS = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_savings_estimation_compute"
)
RLS_MIGRATION = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_savings"
    / "081_savings_estimates_rls_tenant_isolation.sql"
)

#: The TENANT relation whose cross-domain read produced this ticket. Named so a
#: reintroduction fails by name rather than by symptom on a lane.
TENANT_RELATION = "savings_estimates"

_INSERT_TARGET_RE = re.compile(r"INSERT\s+INTO\s+([A-Za-z_][A-Za-z0-9_.]*)", re.I)
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([A-Za-z_][A-Za-z0-9_.]*)", re.I
)


def _handler_write_targets() -> set[str]:
    """Every relation the handler's own SQL writes to."""
    return {m.group(1) for m in _INSERT_TARGET_RE.finditer(HANDLER.read_text())}


def _executable_sql(path: Path) -> str:
    """A migration's statements with its ``--`` commentary removed.

    These headers quote the very statements they explain, so scanning the raw
    file admits comment prose into the owned set. That direction is the unsafe
    one: a spuriously LARGER owned set silently satisfies the write assertion
    below for a relation no migration actually creates.
    """
    return "\n".join(line.split("--", 1)[0] for line in path.read_text().splitlines())


def _node_owned_relations() -> set[str]:
    """Every relation this node's own migration tree creates."""
    owned: set[str] = set()
    for sql_file in sorted(NODE_MIGRATIONS.glob("*.sql")):
        owned |= {
            m.group(1) for m in _CREATE_TABLE_RE.finditer(_executable_sql(sql_file))
        }
    return owned


def test_the_handler_writes_only_relations_this_node_creates() -> None:
    """A node writing a relation it does not own is the OMN-16770 defect."""
    writes = _handler_write_targets()
    assert writes, "parsed no write targets — the parser drifted, not the handler"
    unowned = writes - _node_owned_relations()
    assert not unowned, (
        f"{sorted(unowned)} are written by HandlerSavingsCorrelation but created by "
        f"no migration under {NODE_MIGRATIONS.relative_to(REPO_ROOT)}. This handler "
        f"takes a raw asyncpg pool, so nothing refuses an undeclared relation at "
        f"wiring time — it surfaces on a lane, once a minute, forever (OMN-16770)."
    )


def test_the_idempotency_relation_is_created_by_this_nodes_own_migration() -> None:
    """The readiness anti-join reads something this node can answer for.

    Before the durable close this was ``savings_estimates``, created by
    omnimarket's ``node_projection_savings`` and owned by nobody this node can
    speak for. Now it is created here, in this node's own tree.
    """
    assert IDEMPOTENCY_RELATION in _node_owned_relations()


def test_the_tenant_relation_is_not_written_or_probed_by_this_node() -> None:
    """The relation whose cross-domain read produced this ticket, by name."""
    assert TENANT_RELATION not in {
        target.split(".")[-1] for target in _handler_write_targets()
    }
    assert IDEMPOTENCY_RELATION.split(".")[-1] != TENANT_RELATION


def test_the_tenant_relation_really_is_tenant_domain() -> None:
    """Derived from 081, not recalled — this is WHY the read had to move.

    If 081 ever stops forcing row-level security or changes its predicate, the
    reasoning that made ``savings_estimates`` unreadable under this node's
    binding has changed, and the close must be re-derived rather than trusted.
    """
    sql = RLS_MIGRATION.read_text()
    assert re.search(
        rf"ALTER TABLE {TENANT_RELATION}\s+FORCE ROW LEVEL SECURITY", sql
    ), "081 no longer FORCEs RLS — re-derive OMN-16770 rather than trusting it"
    assert f"current_setting('{TENANT_GUC}', true)" in sql


def test_the_node_owned_relations_carry_no_row_level_security() -> None:
    """This node has no tenant scope to bind, so its own tree must not need one.

    A GUC-predicated policy on any relation this node reads under its own
    binding puts the batch back into the permanent OMN-16770 refusal — the
    handler would have to invent a tenant scope, which the OMN-16831 ruling
    forbids.
    """
    for sql_file in sorted(NODE_MIGRATIONS.glob("*.sql")):
        statements = _executable_sql(sql_file).upper()
        assert "ROW LEVEL SECURITY" not in statements, (
            f"{sql_file.name} puts row-level security on a relation this node "
            f"reads under a binding that can never satisfy it (OMN-16770)"
        )
        assert "CREATE POLICY" not in statements
