#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Every TABLE grant the topology declares must be DELIVERED by a migration.

OMN-17374.

THE DEFECT CLASS THIS EXISTS TO CLOSE
-------------------------------------
``src/omnibase_infra/topology/instances/*.yaml`` declares, per principal, the
exact set of relations that principal may read and write. That file is not
hand-authored: ``scripts/generate_application_database_table_grants.py --write``
derives it from node contract ``db_io.db_tables`` declarations. It is the
platform's statement of intent, and it is checked by
``application_database_domain_enforcement`` against a *declared* state.

Nothing has ever checked that a migration actually ISSUES those grants against a
real database. The two halves drifted apart silently, and the drift is only ever
discovered as a live outage on whichever relation happens to take traffic next:

  * OMN-15701 -- a pin regeneration silently reverted ``tenant_projection_writer``
    TABLE grants for eight house-tenant relations.
  * OMN-16436 -- grant drift on ``delegation_routing_tenant_overlay``.
  * OMN-16993 -- ``session_replay_snapshots``: the topology declared the grant,
    no migration issued it, and every write failed ``InsufficientPrivilege``
    while the runtime reported healthy and committed offsets. That file's own
    header names the residual: "The same gap exists for the other 38 relations
    in that same topology grant list."
  * OMN-17290 -- contract-derived table grant drift after omnimarket dev moved.
  * OMN-17374 -- ``tenant_registry_mirror``: same shape again. It blocked the
    delegation write path's identity resolution AND kept the mirror itself at
    zero rows, because the read and the write both ride the one absent grant.

Five instances of one defect class in a fortnight. Detection that is not a gate
will not hold (Operating Rule 5), so this is wired as a gate.

WHAT IT CHECKS, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------
For every ``(principal, schema, table)`` the topology declares under
``object_type: TABLE``, it looks for at least one ``GRANT ... ON <table> TO
<principal>`` statement somewhere in the vendored forward-migration corpus. It
does NOT check privilege sets, ordering, or whether the migration has been
applied on any particular lane -- a live-lane assertion belongs in the migration
itself, and every grant migration in this corpus already carries one through
``has_table_privilege`` or the older
``information_schema.role_table_grants``-based assertion form.

WHY A RATCHET AND NOT A CLEAN ZERO
----------------------------------
Measured on 2026-09-01 the corpus delivered 24 of the 65 declared grants. The
other 41 were real, and the live ``.201`` dev lane carried 38 of them ONLY
because they were granted out of band by hand -- which is precisely why a fresh
lane (staging, onex-dev, prod) does not have them and nobody notices until a
projection starts refusing writes.

Closing all 41 in one change is not possible and should not be attempted: each
grant belongs in its own node's migration lineage, next to the file that creates
the relation, which is the convention ``node_projection_session_replay/0002``,
``node_log_persistence_effect/0000``, ``node_projection_registration/0005`` and
``node_savings_estimation_compute/0001`` already follow. So this gate is a
ratchet on the undelivered count, in the same shape and for the same reason as
the ``test_application_migration_manifest.py`` exactness ratchet: the number may
go DOWN in any change and may never go UP. That is not an allowlist -- no name
is exempted, every undelivered pair is printed on every run, and a new
undelivered grant fails the gate immediately.

DELIVERY PROGRESS
-----------------
  * OMN-17374 delivered ``tenant_registry_mirror`` and landed this gate at 40.
  * OMN-17440 delivered 13 more across 8 node lineages, taking the bound to 27.
    The tranche was chosen mechanically rather than by taste: the nodes
    OMN-17447 derived as sequence-backed, i.e. where a delivered TABLE grant is
    STILL not sufficient to write because the INSERT fails at the sequence
    behind the key first. Three of them (``merge_state_transitions``,
    ``pr_lifecycle_ledger_entries``, ``receipt_gate_rows``) were measured at
    zero rows by OMN-17377 and proven refusing on the real wired path by
    OMN-17379. The ``node_projection_baselines`` relations in that tranche
    turned out NOT to be sequence-backed (see THE SEQUENCE HALF below); their
    TABLE grants were just as undelivered, so they stayed.
  * OMN-17440 tranche 2 delivered the remaining 26 across 21 node lineages,
    taking the bound to 1 -- the undeliverable residual below. The tranche is
    the whole rest of the set rather than another sub-selection, because the
    live measurement made further staging pointless: of the 63 declared grants
    the ``.201`` dev lane actually holds (read-only probe of
    ``information_schema.role_table_grants`` in ``omnidash_analytics``,
    2026-09-03), exactly 38 were issued by a migration. The other 26 were
    granted OUT OF BAND BY HAND, so every one of them is already a live
    dependency on a lane nobody can rebuild. A fresh staging namespace, a
    rebuilt onex-dev or prod gets 38 of 65 and refuses the rest.

WHAT THE REMAINING 1 IS
-----------------------
``nightly_loop_configs`` is NOT deliverable by any migration: nothing in the
corpus issues a CREATE TABLE for it, so there is no lineage to put a grant in
and no relation for a grant to bite on. It is the reason the floor is 1 rather
than 0, and ``test_residual_relation_has_no_creating_migration`` asserts that
reason still holds instead of trusting this prose. Closing it to 0 is not a
grant-delivery change: either ``node_nightly_loop_controller`` grows a CREATE
TABLE for the relation its contract declares under ``db_io.db_tables``, or that
declaration is removed. Both are OMN-17440 AC2/AC4 decisions, and neither is
made by adding a GRANT for a relation that does not exist.

THE SEQUENCE HALF (OMN-17447)
-----------------------------
A delivered TABLE grant is not sufficient to write a relation whose primary key
is ``SERIAL``/``BIGSERIAL``. PostgreSQL rewrites such a column into a plain
``nextval()`` DEFAULT over a STANDALONE sequence and checks that sequence's OWN
acl on every INSERT -- a privilege ``GRANT INSERT ON TABLE`` does not reach. So
a relation can pass every table check here and still refuse every write. That is
not hypothetical: ``pr_merged_events`` sat 24 days behind its topic at consumer
LAG 0 for exactly this reason, and OMN-17377 independently found three more of
these relations sitting at zero rows.

Before OMN-17447 this gate could not see that half AT ALL. ``_GRANT_RE`` reads a
relation-and-role pair, and ``GRANT USAGE ON SEQUENCE <name> TO <role>`` does not
parse as one -- so sequence grants were INVISIBLE to it rather than merely
filtered out of it, and no ratchet movement could ever have revealed them.

The requirement is now DERIVED rather than hand-listed, because a hand list is
the thing that keeps failing:

  * the topology models no ``object_type: SEQUENCE`` grant anywhere, so there is
    nothing to read directly. Instead, every declared TABLE grant carrying
    INSERT is resolved against the corpus's own column shapes: if the relation
    has a SERIAL/BIGSERIAL column, a sequence grant is required for that
    principal.
  * ``GENERATED ... AS IDENTITY`` columns are excluded. Their sequence is owned
    by the column and rides the table's own INSERT privilege, so demanding a
    separate USAGE grant for one would be wrong.
  * column shapes are read from the APPLIED END STATE, replaying the corpus in
    apply order so a later ``DROP TABLE`` + recreate wins over the original
    declaration. This is load-bearing, not fastidious: OMN-17447's filed list
    named the three ``baselines_*`` child tables as BIGSERIAL, having read the
    CREATE in ``node_projection_baselines/0001``. ``0002`` recreates all three
    with ``id TEXT PRIMARY KEY``. Grant migrations for them would have hit the
    delivering file's own fail-loud NULL guard and broken every lane's deploy.

Deriving it also found three gaps the ticket's ``omninode_runtime``-only scope
never looked at -- ``tenant_projection_writer`` on ``capability_scores``,
``delegation_routing_tenant_overlay`` and ``dep_health_findings``.

A NOTE ON SCOPE: savings_estimates
----------------------------------
``savings_estimates`` is deliberately absent from everything above. It is
declared for ``tenant_projection_writer`` only (already delivered) and for
``omninode_runtime`` NOT AT ALL, so it can never appear in this gate's output.
That is correct and must stay that way: granting the runtime principal a SELECT
there would be actively wrong, because FORCE RLS with an unset GUC makes a
granted SELECT return zero rows, inverting the ``NOT EXISTS`` anti-join into
re-finalizing every session forever (OMN-16770).

The full residual is tracked by OMN-17440.

THE REVERSE ARM: DELIVERED BUT UNDECLARED (OMN-18768)
-----------------------------------------------------
Everything above reads in one direction -- the topology declares, the corpus
must deliver. The opposite direction is the one that took the runtime down.

On 2026-09-19 the main ONEX runtime on the ``.201`` dev lane crash-looped on
every boot, 16 restarts, on::

    Auto-wiring contract 'projection_runner_fleet' failed:
    handler=FleetLivenessProjectionWriter: ValueError: Projection binding
    'omninode_runtime_service' principal 'omninode_runtime' lacks declared read

The database was not the problem. ``omninode_runtime`` HELD every privilege it
needed: this corpus's own
``node_projection_runner_fleet/0001_grant_omninode_runtime_runner_fleet_liveness.sql``
had issued them, and it landed here at 17:33Z. What was missing was the
DECLARATION -- the ``object_type: TABLE`` entry in the topology instances that
``_require_projection_binding_privileges`` reads. The runtime is fail-closed on
the declaration, so an undeclared relation refuses the whole process, not just
that one handler.

The declaration was missing because it is DERIVED from omnimarket node
contracts at a PINNED ref (``.github/omnimarket-contract-pin.yaml``), and the
window between a contract merging upstream and the pin advancing is real: the
refresh job snapshotted omnimarket ``dev`` at 18:30Z and the contract declaring
``runner_fleet_liveness`` merged at 18:41Z, eleven minutes later. Until the next
refresh the derivation could not see it, so
``application_database_domain_enforcement`` was GREEN by construction while the
lane -- which builds omnimarket from source, not from the pin -- was down.

This arm closes that window from inside the repository, with no network and no
pin involved. A relation this corpus GRANTS to a principal the topology models
is a relation this repository already knows about; if the topology does not
declare it, the runtime will refuse it at boot. The grant migration and the
declaration therefore land together or the gate fails -- and PR #3819, which
vendored the migration four hours before the crash, is exactly where it would
have fired.

Same ratchet discipline as both arms above: the count may go DOWN in any change
and may never go UP, and a reading below the bound fails too so the bound is
tightened in the change that earns it. The residual of 4 is NOT an allowlist --
every pair is printed on every run:

  * ``omninode_runtime`` on ``savings_injection_signals``,
    ``savings_correlation_finalizations`` and ``savings_validator_catch_signals``.
    A different class, not this one: ``node_savings_estimation_compute`` lives
    in omnibase_infra and declares NO ``db_io`` block at all, so no contract
    anywhere declares these relations and the contract-driven derivation has
    nothing to derive from. They cannot crash the runtime the way
    ``runner_fleet_liveness`` did -- nothing resolves a projection binding for
    them -- but the corpus granting a relation no contract declares is real
    drift and is left visible rather than filtered away.
  * ``tenant_projection_writer`` on ``public.projection_delegation_savings``,
    granted by ``089_savings_aggregate_views_per_tenant.sql``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

# ``GRANT <privs> ON [TABLE] [schema.]table TO <role>``. Deliberately tolerant of
# the spellings actually used in this corpus (quoted identifiers, an optional
# TABLE keyword, an optional schema qualifier) and deliberately NOT a general SQL
# parser: a grant this pattern cannot read is reported as undelivered, which is
# the direction that fails closed.
_GRANT_RE = re.compile(
    r"GRANT\s+(?P<privs>[A-Za-z][A-Za-z ,\n\r\t]*?)\s+ON\s+(?:TABLE\s+)?"
    r"(?P<relation>[A-Za-z0-9_.\"]+)\s+TO\s+(?P<role>[A-Za-z0-9_\"]+)",
    re.IGNORECASE,
)

# The count measured on 2026-09-03 after OMN-17440's second delivery tranche
# (the remaining 26 grants across 21 node lineages) landed on top of the first
# (13 across 8) and OMN-17374's own. What is left is the single undeliverable
# residual named in WHAT THE REMAINING 1 IS above, so this is now the floor and
# not merely the current reading.
# It may only ever go down. Moving it UP is the one edit this file rejects on
# sight: it converts the gate into a record of the drift instead of a bound on
# it.
MAX_UNDELIVERED = 1

# OMN-17447: the SEQUENCE half of the same defect class, measured after this
# change lands its own seven deliveries. Same ratchet discipline as above.
MAX_UNDELIVERED_SEQUENCES = 0

# OMN-18768: the REVERSE arm -- relations this corpus grants that the topology
# does not declare. Measured on 2026-09-19 after this change declared
# `runner_fleet_liveness`. Same ratchet discipline as both arms above; see THE
# REVERSE ARM in the module docstring for what the 4 are and why each is a
# different class from the crash this arm exists to prevent.
MAX_UNDECLARED = 4

TOPOLOGY_RELPATH = "src/omnibase_infra/topology/instances/local.yaml"
CORPUS_RELPATH = "docker/migrations/forward"

# ``CREATE TABLE [IF NOT EXISTS] [schema.]name (`` -- the opening paren is
# required so the column body can be paren-matched from it.
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?P<if_not_exists>IF\s+NOT\s+EXISTS\s+)?"
    r"(?P<relation>[A-Za-z0-9_.\"]+)\s*\(",
    re.IGNORECASE,
)

_DROP_TABLE_RE = re.compile(
    r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?P<relation>[A-Za-z0-9_.\"]+)",
    re.IGNORECASE,
)

# A ``SERIAL``/``BIGSERIAL``/``SMALLSERIAL`` column declaration. PostgreSQL
# rewrites these into a plain ``nextval()`` DEFAULT over a STANDALONE sequence
# whose own ACL it checks on every INSERT -- which is exactly the privilege
# ``GRANT INSERT ON TABLE`` does not reach.
_IDENTIFIER_RE = r'(?:[A-Za-z_][A-Za-z0-9_]*|"[^"]+")'

_SERIAL_COLUMN_RE = re.compile(
    rf"^\s*(?P<column>{_IDENTIFIER_RE})\s+"
    r"(?:BIGSERIAL|SMALLSERIAL|SERIAL[248]?|SERIAL)\b",
    re.IGNORECASE | re.MULTILINE,
)

# OMN-18353: the SAME column shape, added to an existing relation instead of
# declared in a CREATE body:
#
#     ALTER TABLE omninode_internal.consumer_flow_windows
#         ADD COLUMN IF NOT EXISTS projection_cursor BIGSERIAL;
#
# PostgreSQL creates exactly the same standalone sequence either way, so the
# grant requirement is identical -- but the derivation below read CREATE bodies
# ONLY, and an ALTER-added SERIAL was therefore invisible to this gate.
#
# It stayed invisible because every ALTER of this shape already in the corpus
# sits in the same file as a CREATE declaring the same column (the idempotent
# reconcile pattern: create-if-absent, then add-column-if-absent for a lane that
# predates it). The CREATE covered the ALTER by accident in every case that
# existed -- until OMN-18043 landed a file carrying the ALTER and nothing else.
#
# The statement is matched to its terminating semicolon rather than clause by
# clause, so a multi-clause ``ALTER TABLE t ADD COLUMN a INT, ADD COLUMN b
# BIGSERIAL;`` yields both columns. Deliberately NOT handled, and named rather
# than left silent: ``ALTER COLUMN ... TYPE``, which cannot introduce a serial
# in PostgreSQL (there is no ``SET DATA TYPE SERIAL``), and a ``DEFAULT
# nextval(...)`` attached by hand to an already-created column, which no file in
# this corpus does and which would need the sequence to be CREATEd separately
# and would therefore be visible as its own statement.
_ALTER_TABLE_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?"
    r"(?P<relation>[A-Za-z0-9_.\"]+)(?P<body>[^;]*);",
    re.IGNORECASE,
)

_ADD_COLUMN_RE = re.compile(
    rf"ADD\s+(?:COLUMN\s+)?(?:IF\s+NOT\s+EXISTS\s+)?"
    rf"(?P<column>{_IDENTIFIER_RE})\s+(?P<definition>[^,;]+)",
    re.IGNORECASE,
)

_SERIAL_TYPE_RE = re.compile(
    r"^\s*(?:BIGSERIAL|SMALLSERIAL|SERIAL[248]?|SERIAL)\b", re.IGNORECASE
)

_IDENTITY_DEFINITION_RE = re.compile(
    r"GENERATED\s+(?:ALWAYS|BY\s+DEFAULT)\s+AS\s+IDENTITY", re.IGNORECASE
)

# ``GENERATED ... AS IDENTITY`` is deliberately NOT matched above. An identity
# column's sequence is OWNED by the column and is reachable through the table's
# own INSERT privilege, so demanding a separate USAGE grant for one would be
# wrong. The distinction is the one infra#3094's companion migration documents.
_IDENTITY_COLUMN_RE = re.compile(
    r"^\s*(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s+[A-Za-z0-9_ ]*?"
    r"GENERATED\s+(?:ALWAYS|BY\s+DEFAULT)\s+AS\s+IDENTITY",
    re.IGNORECASE | re.MULTILINE,
)

# ``GRANT USAGE ON SEQUENCE <name> TO <role>`` -- the statement shape the
# existing _GRANT_RE cannot parse AT ALL (it reads a relation-and-role pair, and
# this is neither), which is why sequence grants were invisible to this gate
# rather than merely filtered out of it.
_SEQUENCE_GRANT_RE = re.compile(
    r"GRANT\s+[A-Za-z][A-Za-z ,\n\r\t]*?\s+ON\s+SEQUENCE\s+"
    r"(?P<sequence>[^\s;]+)\s+TO\s+(?P<role>[A-Za-z0-9_\"]+)",
    re.IGNORECASE,
)

# The dynamic form the proven fix uses (omnimarket#2256 / OMN-17379): the
# sequence is resolved at apply time via ``pg_get_serial_sequence`` rather than
# by spelling a name that a restore or rename could invalidate. The grant is
# issued from inside a DO block by ``EXECUTE format(...)``, so no literal
# sequence name appears anywhere in the file and a name-matching scanner sees
# nothing. This captures the (table, column) pair it resolves instead.
_PG_GET_SERIAL_SEQUENCE_RE = re.compile(
    r"pg_get_serial_sequence\(\s*'(?P<relation>[^']+)'\s*,\s*'(?P<column>[^']+)'\s*\)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class GrantKey:
    """One relation-level authorization, as the topology names it."""

    principal: str
    schema: str
    table: str

    def __str__(self) -> str:
        return f"{self.principal} -> {self.schema}.{self.table}"


@dataclass(frozen=True, slots=True)
class SequenceKey:
    """One sequence-backed column a declared TABLE grant implies.

    Identified by ``(principal, schema, table, column)`` rather than by the
    sequence's NAME. The name is a PostgreSQL implementation detail
    (``<table>_<column>_seq`` by default, but a restore, a rename or an
    out-of-band apply can produce another), and the delivering migrations
    resolve it through ``pg_get_serial_sequence`` for exactly that reason. A
    gate keyed on the name would disagree with the fix it is gating.
    """

    principal: str
    schema: str
    table: str
    column: str

    def __str__(self) -> str:
        return f"{self.principal} -> {self.schema}.{self.table}.{self.column}"


def _unquote(identifier: str) -> str:
    return identifier.replace('"', "")


def _postgres_identifier(identifier: str) -> str:
    if identifier.startswith('"') and identifier.endswith('"'):
        return _unquote(identifier)
    return identifier.lower()


def _corpus_files_in_apply_order(corpus_root: Path) -> list[Path]:
    """Corpus files in the order a lane applies them.

    Order is load-bearing here, and getting it wrong is not a cosmetic problem:
    a table can be CREATEd with a ``BIGSERIAL`` key in one migration and then
    DROPped and recreated with a ``TEXT`` key in a later one within the same
    node lineage. Reading only the first CREATE reports a sequence that does
    not exist in the applied end state, and the delivering migration would then
    ``RAISE EXCEPTION`` on a NULL ``pg_get_serial_sequence`` -- turning this
    gate into a broken build rather than a correct one.

    That is not hypothetical: ``node_projection_baselines/0002`` does exactly
    this to ``baselines_breakdown``, ``baselines_comparisons`` and
    ``baselines_trend``, all three of which ``0001`` declares ``BIGSERIAL``.
    OMN-17447's filed derivation read only the CREATE statements and listed all
    three as gapped sequences; they are not.

    Flat files sort numerically by their leading ordinal, then node-owned files
    by node directory and ordinal, which is the order
    ``run-forward-migrations.sh`` applies them in.
    """

    def sort_key(path: Path) -> tuple[int, str, str]:
        relative = path.relative_to(corpus_root)
        if len(relative.parts) == 1:
            return (0, "", relative.name)
        return (1, "/".join(relative.parts[:-1]), relative.name)

    return sorted(corpus_root.rglob("*.sql"), key=sort_key)


def _column_body(text: str, open_paren_index: int) -> str:
    """The parenthesised column list starting at ``open_paren_index``.

    Paren-matched rather than regex-terminated so that a ``NUMERIC(10,2)`` or a
    table-level ``CHECK (...)`` inside the body cannot end the match early.
    """
    depth = 0
    for index in range(open_paren_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_index : index + 1]
    return text[open_paren_index:]


def sequence_backed_columns(corpus_root: Path) -> dict[tuple[str, str], set[str]]:
    """``(schema, table) -> {sequence-backed column}`` in the APPLIED end state.

    Replays the corpus in apply order, letting a later ``CREATE TABLE`` replace
    an earlier definition and a ``DROP TABLE`` remove it, so the answer reflects
    what a lane actually ends up with rather than what any single file says.

    ``IDENTITY`` columns are excluded by design -- their sequence is owned by
    the column and rides the table's own INSERT privilege.
    """
    definitions: dict[tuple[str, str], set[str]] = {}
    for sql_path in _corpus_files_in_apply_order(corpus_root):
        text = sql_path.read_text(encoding="utf-8", errors="replace")

        events: list[tuple[int, str, str, str, bool]] = []
        for match in _DROP_TABLE_RE.finditer(text):
            events.append((match.start(), "drop", match.group("relation"), "", False))
        for match in _CREATE_TABLE_RE.finditer(text):
            events.append(
                (
                    match.start(),
                    "create",
                    match.group("relation"),
                    "",
                    bool(match.group("if_not_exists")),
                )
            )
        for match in _ALTER_TABLE_RE.finditer(text):
            events.append(
                (
                    match.start(),
                    "alter",
                    match.group("relation"),
                    match.group("body"),
                    False,
                )
            )

        for position, kind, relation, body, if_not_exists in sorted(events):
            key = _split_relation(relation)
            if kind == "drop":
                definitions.pop(key, None)
                continue
            if kind == "alter":
                added = _added_serial_columns(body)
                if added:
                    definitions.setdefault(key, set()).update(added)
                continue
            open_paren = text.index("(", position)
            column_body = _column_body(text, open_paren)
            serial = {
                _postgres_identifier(m.group("column"))
                for m in _SERIAL_COLUMN_RE.finditer(column_body)
            }
            identity = {
                _postgres_identifier(m.group("column"))
                for m in _IDENTITY_COLUMN_RE.finditer(column_body)
            }
            if if_not_exists:
                definitions.setdefault(key, set()).update(serial - identity)
            else:
                definitions[key] = serial - identity
    return definitions


def _added_serial_columns(alter_body: str) -> set[str]:
    """Sequence-backed columns an ``ALTER TABLE`` body ADDs (OMN-18353).

    ``ALTER`` accumulates onto whatever the relation already holds, so unlike a
    ``CREATE`` it never REPLACES the derived column set -- an ALTER-added
    ``BIGSERIAL`` on a table whose CREATE declared none must add to it, and an
    ALTER that adds no serial column must leave it untouched rather than clear
    it.

    ``GENERATED ... AS IDENTITY`` is excluded here for the same reason it is in
    a CREATE body: that sequence is owned by the column and rides the table's
    own INSERT privilege, so requiring a separate USAGE grant would make the
    delivering migration RAISE on a NULL ``pg_get_serial_sequence``.
    """
    added: set[str] = set()
    for match in _ADD_COLUMN_RE.finditer(alter_body):
        definition = match.group("definition")
        if _IDENTITY_DEFINITION_RE.search(definition):
            continue
        if _SERIAL_TYPE_RE.match(definition):
            added.add(_postgres_identifier(match.group("column")))
    return added


def _split_relation(relation: str) -> tuple[str, str]:
    """Split ``[schema.]table`` into ``(schema, table)``.

    An unqualified relation resolves to ``public``: that is what PostgreSQL does
    under this corpus's search_path, and it is what every unqualified GRANT in
    the corpus means.
    """
    bare = _unquote(relation)
    if "." in bare:
        schema, table = bare.split(".", 1)
        return schema, table
    return "public", bare


def declared_grants(topology_path: Path) -> set[GrantKey]:
    """The TABLE grants the topology declares, per principal.

    Read from ``local.yaml`` alone by design. The three instance files carry the
    same generated principal grant blocks, and reading one keeps the gate's
    subject unambiguous; an instance that drifts from the others is a different
    defect with its own check.
    """
    document = yaml.safe_load(topology_path.read_text(encoding="utf-8"))
    principals = document["databases"]["application"]["principals"]
    declared: set[GrantKey] = set()
    for principal, spec in principals.items():
        for grant in spec.get("grants") or ():
            if grant.get("object_type") != "TABLE":
                continue
            schema = grant.get("schema")
            if schema is None:
                raise ValueError(
                    f"topology principal {principal!r} declares a TABLE grant "
                    "with no schema; the generated file is malformed"
                )
            for table in grant.get("objects") or ():
                declared.add(GrantKey(principal, schema, table))
    return declared


def delivered_grants(corpus_root: Path) -> dict[GrantKey, list[str]]:
    """Every relation-level GRANT the vendored forward corpus issues.

    Values are the file names that issue it, so a failure can name where a
    sibling grant was landed and the missing one was not.
    """
    delivered: dict[GrantKey, list[str]] = {}
    for sql_path in sorted(corpus_root.rglob("*.sql")):
        text = sql_path.read_text(encoding="utf-8", errors="replace")
        for match in _GRANT_RE.finditer(text):
            schema, table = _split_relation(match.group("relation"))
            # `GRANT ... ON ALL TABLES IN SCHEMA x` parses here with a relation
            # of `ALL`; it grants no NAMED relation and must not be read as
            # delivering one. `TABLES` is the same class one statement over:
            # `ALTER DEFAULT PRIVILEGES ... GRANT ... ON TABLES TO <role>`
            # (099_create_omninode_internal_live_events.sql) names a future
            # default, not a relation, and reading it as `public.TABLES` put a
            # phantom pair in the delivered set (OMN-18768).
            if table.upper() in {"ALL", "SCHEMA", "DATABASE", "SEQUENCE", "TABLES"}:
                continue
            key = GrantKey(_unquote(match.group("role")), schema, table)
            delivered.setdefault(key, []).append(sql_path.name)
    return delivered


def declared_sequences(repo_root: Path) -> set[SequenceKey]:
    """Sequence grants IMPLIED by the topology's declared TABLE grants.

    The topology models no ``object_type: SEQUENCE`` grant anywhere -- sequences
    are not grantable objects in it at all. That is not treated as "nothing to
    check": a declared INSERT on a ``BIGSERIAL``-keyed relation is a statement
    that the principal must be able to write that relation, and it CANNOT
    without USAGE on the sequence behind the key. So the requirement is DERIVED
    from the table declaration plus the corpus's own column shapes, which is
    the only place the truth exists.

    Only write-implying declarations are considered. A principal declared
    SELECT-only has no INSERT to fail and owes no sequence privilege.
    """
    document = yaml.safe_load(
        (repo_root / TOPOLOGY_RELPATH).read_text(encoding="utf-8")
    )
    principals = document["databases"]["application"]["principals"]
    columns = sequence_backed_columns(repo_root / CORPUS_RELPATH)

    implied: set[SequenceKey] = set()
    for principal, spec in principals.items():
        for grant in spec.get("grants") or ():
            if grant.get("object_type") != "TABLE":
                continue
            privileges = {p.upper() for p in grant.get("privileges") or ()}
            if "INSERT" not in privileges:
                continue
            schema = grant["schema"]
            for table in grant.get("objects") or ():
                for column in sorted(columns.get((schema, table), set())):
                    implied.add(SequenceKey(principal, schema, table, column))
    return implied


def delivered_sequences(corpus_root: Path) -> dict[SequenceKey, list[str]]:
    """Every sequence USAGE the corpus issues, by ``(table, column)`` identity.

    Two shapes are read, because both are in the corpus:

      * the dynamic ``pg_get_serial_sequence('public.t', 'c')`` form the proven
        fix uses, issued from a DO block via ``EXECUTE format(...)``;
      * a literal ``GRANT USAGE ON SEQUENCE <name> TO <role>``, resolved back to
        a ``(table, column)`` pair by the default ``<table>_<column>_seq``
        naming convention.

    A grant neither shape can read is reported as undelivered, which is the
    direction that fails closed.
    """
    delivered: dict[SequenceKey, list[str]] = {}
    columns = sequence_backed_columns(corpus_root)
    for sql_path in sorted(corpus_root.rglob("*.sql")):
        text = sql_path.read_text(encoding="utf-8", errors="replace")

        for match in _SEQUENCE_GRANT_RE.finditer(text):
            role = _unquote(match.group("role"))
            target = match.group("sequence")
            resolved = _PG_GET_SERIAL_SEQUENCE_RE.search(target)
            if resolved is not None:
                schema, table = _split_relation(resolved.group("relation"))
                column = resolved.group("column").lower()
            else:
                schema, sequence = _split_relation(target)
                bare = _unquote(sequence)
                if not bare.endswith("_seq"):
                    continue
                candidates = [
                    (table, column)
                    for candidate_schema, table in columns
                    if candidate_schema == schema
                    for column in columns[(candidate_schema, table)]
                    if bare == f"{table}_{column}_seq"
                ]
                if len(candidates) == 1:
                    table, column = candidates[0]
                else:
                    table, _, column = bare[: -len("_seq")].rpartition("_")
                if not table or not column:
                    continue
            delivered.setdefault(
                SequenceKey(role, schema, table, column.lower()), []
            ).append(sql_path.name)

        # The DO-block form: `EXECUTE format('GRANT USAGE ON SEQUENCE %s TO
        # <role>', v_seq)` where v_seq came from pg_get_serial_sequence. The
        # literal pattern above cannot see it, because the sequence name is a
        # format placeholder rather than an identifier.
        for match in re.finditer(
            r"GRANT\s+[A-Za-z][A-Za-z ,]*?\s+ON\s+SEQUENCE\s+%s\s+TO\s+"
            r"(?P<role>[A-Za-z0-9_]+)",
            text,
            re.IGNORECASE,
        ):
            role = match.group("role")
            for resolved in _PG_GET_SERIAL_SEQUENCE_RE.finditer(text):
                schema, table = _split_relation(resolved.group("relation"))
                delivered.setdefault(
                    SequenceKey(role, schema, table, resolved.group("column").lower()),
                    [],
                ).append(sql_path.name)
    return delivered


def undelivered_sequences(repo_root: Path) -> list[SequenceKey]:
    """Implied sequence grants with no delivering statement in the corpus."""
    declared = declared_sequences(repo_root)
    delivered = delivered_sequences(repo_root / CORPUS_RELPATH)
    return sorted(
        (key for key in declared if key not in delivered),
        key=lambda key: (key.principal, key.schema, key.table, key.column),
    )


def undelivered(repo_root: Path) -> list[GrantKey]:
    """Declared grants with no delivering statement anywhere in the corpus."""
    declared = declared_grants(repo_root / TOPOLOGY_RELPATH)
    delivered = delivered_grants(repo_root / CORPUS_RELPATH)
    return sorted(
        (key for key in declared if key not in delivered),
        key=lambda key: (key.principal, key.schema, key.table),
    )


def undeclared(repo_root: Path) -> list[GrantKey]:
    """Grants the corpus DELIVERS that the topology does not DECLARE.

    The inverse of :func:`undelivered`, and the direction that crash-loops the
    runtime rather than silently refusing writes: ``handler_wiring``'s
    ``_require_projection_binding_privileges`` resolves a projection binding
    against the topology's declared grants alone, so a relation this corpus has
    already granted in the database is still refused at boot if no declaration
    names it. See THE REVERSE ARM in the module docstring for the OMN-18768
    incident this derives from.

    Scoped to principals the topology actually models. A grant to a role the
    topology has never heard of (a migration-internal ``role_*`` owner, say) is
    not a declaration gap and is not this gate's subject.
    """
    declared = declared_grants(repo_root / TOPOLOGY_RELPATH)
    delivered = delivered_grants(repo_root / CORPUS_RELPATH)
    modelled = {key.principal for key in declared}
    return sorted(
        (key for key in delivered if key.principal in modelled and key not in declared),
        key=lambda key: (key.principal, key.schema, key.table),
    )


def _render(missing: Iterable[GrantKey]) -> str:
    return "\n".join(f"  UNDELIVERED  {key}" for key in missing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: resolved from this file's location)",
    )
    parser.add_argument(
        "--max-undelivered",
        type=int,
        default=MAX_UNDELIVERED,
        help="ratchet bound; the run fails when the count exceeds it",
    )
    parser.add_argument(
        "--max-undelivered-sequences",
        type=int,
        default=MAX_UNDELIVERED_SEQUENCES,
        help="ratchet bound for implied SEQUENCE grants (OMN-17447)",
    )
    parser.add_argument(
        "--max-undeclared",
        type=int,
        default=MAX_UNDECLARED,
        help="ratchet bound for delivered-but-undeclared grants (OMN-18768)",
    )
    args = parser.parse_args(argv)

    missing_sequences = undelivered_sequences(args.repo_root)
    print(
        f"topology sequence grant delivery: {len(missing_sequences)} undelivered "
        f"(bound {args.max_undelivered_sequences})"
    )
    if missing_sequences:
        print("\n".join(f"  UNDELIVERED-SEQ  {key}" for key in missing_sequences))
    sequence_status = 0
    if len(missing_sequences) > args.max_undelivered_sequences:
        print(
            f"\nFAIL: {len(missing_sequences)} sequence-backed columns are behind "
            "a declared INSERT grant with no delivering GRANT USAGE ON SEQUENCE, "
            f"above the bound of {args.max_undelivered_sequences}.\n"
            "A table grant alone does NOT make such a relation writable: a "
            "SERIAL/BIGSERIAL key is a nextval() DEFAULT over a standalone "
            "sequence whose own ACL PostgreSQL checks on every INSERT. That is "
            "the OMN-17379 outage -- pr_merged_events sat 24 days behind its "
            "topic at LAG 0 while every write failed on the sequence. Land the "
            "grant in the owning node's lineage using pg_get_serial_sequence, "
            "and lower the bound in the same change.",
            file=sys.stderr,
        )
        sequence_status = 1
    elif len(missing_sequences) < args.max_undelivered_sequences:
        print(
            f"\nFAIL: {len(missing_sequences)} undelivered sequences is BELOW the "
            f"bound of {args.max_undelivered_sequences}. Lower "
            "MAX_UNDELIVERED_SEQUENCES to "
            f"{len(missing_sequences)} in this same change, so the ratchet keeps "
            "biting.",
            file=sys.stderr,
        )
        sequence_status = 1

    # OMN-18768. Runs BEFORE the undelivered arm below so its findings are
    # printed even on a run the other arm already fails: a lane that has to fix
    # two arms should see both in one run, not discover the second after fixing
    # the first.
    extra = undeclared(args.repo_root)
    print(
        f"topology grant declaration: {len(extra)} delivered-but-undeclared "
        f"(bound {args.max_undeclared})"
    )
    if extra:
        print("\n".join(f"  UNDECLARED  {key}" for key in extra))
    undeclared_status = 0
    if len(extra) > args.max_undeclared:
        print(
            f"\nFAIL: {len(extra)} relations are GRANTED by this corpus and "
            "declared by no topology principal, above the bound of "
            f"{args.max_undeclared}.\n"
            "This is the OMN-18768 crash, not a cosmetic gap: the runtime "
            "resolves a projection binding against the DECLARATION, so a "
            "relation the database already grants is still refused at boot -- "
            "'principal ... lacks declared read' -- and auto-wiring failing "
            "one contract takes the whole runtime process down, not just that "
            "handler. Declare it, then lower the bound in the same change. If "
            "the relation is declared by an omnimarket node contract, do NOT "
            "hand-edit the instances: advance "
            ".github/omnimarket-contract-pin.yaml and regenerate with "
            "scripts/generate_application_database_table_grants.py --write, "
            "which is the only sanctioned writer of that block.",
            file=sys.stderr,
        )
        undeclared_status = 1
    elif len(extra) < args.max_undeclared:
        print(
            f"\nFAIL: {len(extra)} delivered-but-undeclared is BELOW the bound "
            f"of {args.max_undeclared}. Lower MAX_UNDECLARED in "
            "scripts/validation/check_topology_grant_delivery.py to "
            f"{len(extra)} in this same change, so the ratchet keeps biting.",
            file=sys.stderr,
        )
        undeclared_status = 1

    missing = undelivered(args.repo_root)
    print(
        f"topology grant delivery: {len(missing)} undelivered (bound {args.max_undelivered})"
    )
    if missing:
        print(_render(missing))
    if len(missing) > args.max_undelivered:
        print(
            f"\nFAIL: {len(missing)} declared TABLE grants have no delivering "
            f"migration, above the ratchet bound of {args.max_undelivered}.\n"
            "A topology grant with no migration is an outage waiting for the "
            "relation to take traffic (OMN-16993, OMN-17374). Land the GRANT in "
            "the owning node's own migration lineage, next to the file that "
            "creates the relation, and lower the bound in the same change.",
            file=sys.stderr,
        )
        return 1
    if len(missing) < args.max_undelivered:
        print(
            f"\nFAIL: {len(missing)} undelivered is BELOW the bound of "
            f"{args.max_undelivered}. Lower MAX_UNDELIVERED in "
            "scripts/validation/check_topology_grant_delivery.py to "
            f"{len(missing)} in this same change, so the ratchet keeps biting.",
            file=sys.stderr,
        )
        return 1
    return sequence_status or undeclared_status


if __name__ == "__main__":
    raise SystemExit(main())
