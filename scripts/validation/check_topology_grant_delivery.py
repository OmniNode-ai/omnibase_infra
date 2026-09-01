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
itself, and every grant migration in this corpus already carries one
(``SELECT 1 / count(*) ... FROM information_schema.role_table_grants``).

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
    The tranche was chosen mechanically rather than by taste: the owning nodes
    with at least one BIGSERIAL/SERIAL-keyed relation, i.e. exactly the set
    where a delivered TABLE grant is STILL not sufficient to write, because the
    INSERT fails at the sequence behind the key first. Three of them
    (``merge_state_transitions``, ``pr_lifecycle_ledger_entries``,
    ``receipt_gate_rows``) were measured at zero rows by OMN-17377 and proven
    refusing on the real wired path by OMN-17379.

WHAT THE REMAINING 27 ARE
-------------------------
26 are deliverable and simply not yet done -- each has an owning node with a
creating migration to land the grant beside. The 27th, ``nightly_loop_configs``,
is NOT deliverable by any migration: nothing in the corpus issues a CREATE TABLE
for it, so there is no lineage to put a grant in and no relation for a grant to
bite on. It is named here so the floor is understood as 1 rather than 0, and
``test_residual_relation_has_no_creating_migration`` asserts that reason still
holds instead of trusting this prose.

A NOTE ON SCOPE: THIS GATE SEES TABLES ONLY
-------------------------------------------
``savings_estimates`` is deliberately absent from everything above. It is
declared for ``tenant_projection_writer`` only (already delivered) and for
``omninode_runtime`` NOT AT ALL, so it can never appear in this gate's output.
That is correct and must stay that way: granting the runtime principal a SELECT
there would be actively wrong, because FORCE RLS with an unset GUC makes a
granted SELECT return zero rows, inverting the ``NOT EXISTS`` anti-join into
re-finalizing every session forever (OMN-16770).

This gate also does not see SEQUENCES, and cannot: ``_GRANT_RE`` below matches a
relation-and-role pair, and ``GRANT USAGE ON SEQUENCE <name> TO <role>`` does
not parse as one at all -- so sequence grants are invisible to it rather than
merely filtered out. A table can therefore pass this gate and still fail every
write, which is exactly what happened to ``pr_merged_events``. That half is
OMN-17447.

The full residual is tracked by OMN-17440.
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

# The count measured on 2026-09-01 after OMN-17440 landed the first delivery
# tranche (13 grants across 8 node lineages) on top of OMN-17374's own.
# It may only ever go down. Moving it UP is the one edit this file rejects on
# sight: it converts the gate into a record of the drift instead of a bound on
# it.
MAX_UNDELIVERED = 27

TOPOLOGY_RELPATH = "src/omnibase_infra/topology/instances/local.yaml"
CORPUS_RELPATH = "docker/migrations/forward"


@dataclass(frozen=True, slots=True)
class GrantKey:
    """One relation-level authorization, as the topology names it."""

    principal: str
    schema: str
    table: str

    def __str__(self) -> str:
        return f"{self.principal} -> {self.schema}.{self.table}"


def _unquote(identifier: str) -> str:
    return identifier.replace('"', "")


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
            # delivering one.
            if table.upper() in {"ALL", "SCHEMA", "DATABASE", "SEQUENCE"}:
                continue
            key = GrantKey(_unquote(match.group("role")), schema, table)
            delivered.setdefault(key, []).append(sql_path.name)
    return delivered


def undelivered(repo_root: Path) -> list[GrantKey]:
    """Declared grants with no delivering statement anywhere in the corpus."""
    declared = declared_grants(repo_root / TOPOLOGY_RELPATH)
    delivered = delivered_grants(repo_root / CORPUS_RELPATH)
    return sorted(
        (key for key in declared if key not in delivered),
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
    args = parser.parse_args(argv)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
