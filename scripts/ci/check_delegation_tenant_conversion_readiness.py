#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Read-only readiness check for the delegation_events tenant_id conversion.

OMN-15683. The operator-safe form of a recipe that, as a bare SQL query,
returns the exact string that means PASS on a table it cannot see.

THE DEFECT THIS REPLACES
------------------------
Migration ``0034_delegation_events_uuid_via_registry_role_set_guard.sql`` tells
the operator, in its own deployment-ordering header, to verify readiness "with
this file's own pre-guard query, not pod status". Measured against onex-dev on
2026-09-08 (ROLLING_WORK_LEDGER.md, ``fence-apply-onexdev``), that instruction
inverts its own answer:

* the pre-guard ``LEFT JOIN`` returned ZERO unresolvable values -- PASS -- while
  ``pg_stat_user_tables.n_live_tup`` for the same relation read **229**;
* ``delegation_events`` carries ``relforcerowsecurity = t``, and its policy is
  ``USING (tenant_id = current_setting('app.tenant_id', true))``. With the GUC
  unset that predicate is NULL for every row, so the probe saw an EMPTY TABLE;
* FORCE means even the owner is not exempt -- ``SET ROLE`` to the owner still
  returned 0, and ``SET row_security = off`` returned ``ERROR: query would be
  affected by row-level security policy`` rather than a count.

The migration itself is not blind: it reaches its guard after ``SET ROLE`` to the
owner and after ``ALTER TABLE ... NO FORCE ROW LEVEL SECURITY``. A READ-ONLY
probe cannot take that DDL step, which is exactly why quoting the migration's SQL
as a standalone readiness check is unsafe.

WHAT THIS DOES INSTEAD
----------------------
1. Reads the visibility facts -- ``relrowsecurity``, ``relforcerowsecurity``,
   ``row_security_active()``, and ``pg_stat_user_tables.n_live_tup``.
2. Enumerates ``tenant_id`` -> row count. When the session can see the table it
   groups directly (DIRECT mode). When it is policy-blinded it reconstructs
   visibility per identity -- ``set_config('app.tenant_id', <candidate>, true)``
   once per candidate, counting under each (RECONSTRUCTED mode). Candidates come
   from ``tenant_registry_mirror`` (both ``tenant_slug`` and ``tenant_uuid``)
   plus ``--candidate``.
3. **RECONCILES the enumerated total against n_live_tup, and REFUSES to print
   PASS unless it holds.** In RECONSTRUCTED mode a shortfall means rows exist
   under an identity that was never guessed, and an unguessed identity is
   precisely the one that would abort the conversion. A zero that has not been
   reconciled is never reported as a clean bill of health.
4. Classifies every surviving value against BOTH resolution forms the successor
   migration uses -- ``tenant_slug = <value>`` OR ``tenant_uuid::text = <value>``
   -- and names, per unresolvable value, which of the two failed.

n_live_tup is an ESTIMATE. Stated plainly rather than implied: the reconciliation
is a **blindness detector**, not a row-count audit. In DIRECT mode the count is
authoritative and n_live_tup is reported for corroboration only; the run is
refused only on the signature no amount of statistics drift produces -- a table
that reads empty while the statistics say it holds rows. In RECONSTRUCTED mode
the sum must reach n_live_tup, because there the shortfall IS the finding, and a
stale estimate makes that check stricter rather than laxer.

EXIT STATUS
-----------
``0``  PASS -- reconciled, and every surviving value resolves.
``1``  REFUSED -- unresolvable values survive the debris delete.
``2``  INDETERMINATE -- the enumeration could not be reconciled, or the relation
       is missing. Never reported as PASS; an unanswered question is not a
       negative answer.
``3``  The probe could not be run at all.

USAGE
-----
::

    python3 scripts/ci/check_delegation_tenant_conversion_readiness.py \\
        --dsn "postgresql://user@host:5432/omnidash_analytics"

    # or, for a lane reachable only through a container/ssh hop:
    python3 scripts/ci/check_delegation_tenant_conversion_readiness.py \\
        --database omn15683_scratch \\
        --psql-exec '["ssh","host","docker","exec","-i","pg","psql","-U","postgres"]'

Ticket: OMN-15683 (this tool, and 0036), OMN-16930 / OMN-17316 (the chain),
OMN-16804 (write-time stamping, which produced the mixed column).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass

RELATION = "delegation_events"
MIRROR = "tenant_registry_mirror"

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# The six pre-tenancy fixture rows migration 0034/0036 delete by EXACT
# correlation_id. Named here so this probe's arithmetic matches the migration's:
# a row that will be deleted must not be reported as a blocker.
DEBRIS_CORRELATION_IDS = (
    "SEED-A-1",
    "SEED-A-2",
    "SEED-A-3",
    "SEED-B-1",
    "SEED-B-2",
    "SEED-B-3",
)

# Two literal identities that the fixture rows above carry and that
# tenant_registry_mirror cannot contain (they are not registry tenants). They are
# defaults ONLY as reconstruction CANDIDATES -- values to try when the table is
# policy-blinded and the mirror cannot name them. They are never treated as
# resolvable and never map anything.
DEFAULT_EXTRA_CANDIDATES = (
    "11111111-1111-1111-1111-111111111111",
    "22222222-2222-2222-2222-222222222222",
)

FIELD_SEP = "\x1f"

MODE_DIRECT = "DIRECT"
MODE_RECONSTRUCTED = "RECONSTRUCTED"

VERDICT_PASS = "PASS"
VERDICT_REFUSED = "REFUSED"
VERDICT_INDETERMINATE = "INDETERMINATE"

EXIT_PASS = 0
EXIT_REFUSED = 1
EXIT_INDETERMINATE = 2
EXIT_PROBE_ERROR = 3


class ProbeError(RuntimeError):
    """The probe could not be run at all."""


@dataclass(frozen=True)
class PsqlClient:
    """A psql invocation prefix. SQL travels on stdin, never in argv.

    Same shape as scripts/migrations/check_migration_applied_on_lane.py, for the
    same reason: a lane is often reachable only through an ssh + docker exec hop,
    and building that prefix into the tool beats hand-typing psql under pressure.
    """

    argv: tuple[str, ...]
    database: str | None

    def rows(self, sql: str, *variables: str) -> list[list[str]]:
        command = [
            *self.argv,
            "-X",
            "-q",
            "-v",
            "ON_ERROR_STOP=1",
            "-At",
            "-F",
            FIELD_SEP,
            *[item for variable in variables for item in ("-v", variable)],
        ]
        if self.database is not None:
            if not IDENTIFIER_RE.match(self.database):
                raise ProbeError(f"unsafe database identifier: {self.database!r}")
            command += ["-d", self.database]
        command += ["-f", "-"]
        completed = subprocess.run(
            command, input=sql, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise ProbeError(
                f"psql failed (exit {completed.returncode}): "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return [
            line.split(FIELD_SEP)
            for line in completed.stdout.splitlines()
            if line.strip() != ""
        ]

    def scalar(self, sql: str, *variables: str) -> str | None:
        rows = self.rows(sql, *variables)
        if not rows or not rows[0]:
            return None
        return rows[0][0]


@dataclass(frozen=True)
class Visibility:
    row_security_enabled: bool
    row_security_forced: bool
    row_security_active: bool
    n_live_tup: int


@dataclass(frozen=True)
class ValueCensus:
    """One distinct tenant_id value, as this probe measured it."""

    value: str
    rows: int
    debris_rows: int
    resolves_by_slug: bool
    resolves_by_uuid: bool
    distinct_registry_tenants: int

    @property
    def surviving_rows(self) -> int:
        return self.rows - self.debris_rows

    @property
    def resolves(self) -> bool:
        return self.resolves_by_slug or self.resolves_by_uuid

    @property
    def form(self) -> str:
        if self.resolves_by_slug and self.resolves_by_uuid:
            return "tenant_slug AND tenant_uuid (ambiguous)"
        if self.resolves_by_slug:
            return "tenant_slug"
        if self.resolves_by_uuid:
            return "tenant_uuid"
        if UUID_RE.match(self.value):
            return (
                "UNRESOLVABLE -- well-formed UUID, but no "
                f"{MIRROR} row has tenant_uuid = this value"
            )
        return (
            "UNRESOLVABLE -- not a UUID, and no "
            f"{MIRROR} row has tenant_slug = this value"
        )


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _in_list(values: Sequence[str]) -> str:
    return ", ".join(_quote_literal(value) for value in values)


def read_visibility(client: PsqlClient) -> Visibility:
    row = client.rows(
        f"""
        SELECT c.relrowsecurity,
               c.relforcerowsecurity,
               row_security_active('{RELATION}'),
               COALESCE(s.n_live_tup, -1)
        FROM pg_catalog.pg_class c
        LEFT JOIN pg_catalog.pg_stat_user_tables s ON s.relid = c.oid
        WHERE c.oid = '{RELATION}'::regclass;
        """  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
    )
    if not row:
        raise ProbeError(f"{RELATION} does not exist in this database")
    enabled, forced, active, live = row[0]
    return Visibility(
        row_security_enabled=enabled == "t",
        row_security_forced=forced == "t",
        row_security_active=active == "t",
        n_live_tup=int(live),
    )


def read_mirror_candidates(client: PsqlClient) -> list[str]:
    rows = client.rows(
        f"""
        SELECT tenant_slug FROM {MIRROR}
        UNION
        SELECT tenant_uuid::text FROM {MIRROR};
        """  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
    )
    return sorted({row[0] for row in rows})


def census_direct(client: PsqlClient) -> list[tuple[str, int, int]]:
    """(value, rows, debris_rows) read straight off the table."""
    rows = client.rows(
        f"""
        SELECT d.tenant_id,
               count(*),
               count(*) FILTER (
                   WHERE d.correlation_id IN ({_in_list(DEBRIS_CORRELATION_IDS)}))
        FROM {RELATION} d
        GROUP BY d.tenant_id
        ORDER BY 2 DESC, 1;
        """  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
    )
    return [(row[0], int(row[1]), int(row[2])) for row in rows]


def census_reconstructed(
    client: PsqlClient, candidates: Sequence[str]
) -> list[tuple[str, int, int]]:
    """Per-identity counts, reconstructed one ``app.tenant_id`` at a time.

    ``set_config(..., is_local => true)`` is session-local and this probe opens no
    transaction that writes anything; the GUC dies with the connection.
    """
    measured: list[tuple[str, int, int]] = []
    for candidate in candidates:
        row = client.rows(
            f"""
            SELECT set_config('app.tenant_id', {_quote_literal(candidate)}, false);
            SELECT count(*),
                   count(*) FILTER (
                       WHERE correlation_id IN ({_in_list(DEBRIS_CORRELATION_IDS)}))
            FROM {RELATION};
            """  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
        )
        if not row:
            continue
        total, debris = int(row[-1][0]), int(row[-1][1])
        if total > 0:
            measured.append((candidate, total, debris))
    measured.sort(key=lambda item: (-item[1], item[0]))
    return measured


def classify(
    client: PsqlClient, values: Sequence[str]
) -> dict[str, tuple[bool, bool, int]]:
    if not values:
        return {}
    rows = client.rows(
        f"""
        WITH candidate(value) AS (VALUES {", ".join("(" + _quote_literal(v) + ")" for v in values)})
        SELECT c.value,
               bool_or(m.tenant_slug IS NOT NULL AND m.tenant_slug = c.value),
               bool_or(m.tenant_uuid::text = c.value),
               count(DISTINCT m.tenant_uuid)
        FROM candidate c
        LEFT JOIN {MIRROR} m
          ON m.tenant_slug = c.value
          OR m.tenant_uuid::text = c.value
        GROUP BY c.value;
        """  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
    )
    return {row[0]: (row[1] == "t", row[2] == "t", int(row[3])) for row in rows}


def evaluate(
    visibility: Visibility,
    mode: str,
    census: list[ValueCensus],
) -> tuple[str, list[str]]:
    """The verdict, and the reasons for it. PASS is the hardest to reach."""
    reasons: list[str] = []
    enumerated = sum(item.rows for item in census)

    if mode == MODE_RECONSTRUCTED:
        if visibility.n_live_tup <= 0:
            reasons.append(
                "the table is policy-blinded to this session AND "
                f"pg_stat_user_tables.n_live_tup is {visibility.n_live_tup} -- "
                "there is no independent number to reconcile the reconstruction "
                "against, so emptiness cannot be proven. Run ANALYZE "
                f"{RELATION} (as a role permitted to) or re-run as an identity "
                "that can see the table."
            )
            return VERDICT_INDETERMINATE, reasons
        if enumerated != visibility.n_live_tup:
            reasons.append(
                f"reconstruction reached {enumerated} row(s) but "
                f"pg_stat_user_tables.n_live_tup is {visibility.n_live_tup}: "
                f"{visibility.n_live_tup - enumerated} row(s) sit under a tenant "
                "identity that was never guessed. An unguessed identity is "
                "exactly the one that would abort the conversion, so this is "
                "NOT a pass with a rounding error. Supply the missing "
                "identities with --candidate, or re-run as an identity that "
                "can see the table."
            )
            return VERDICT_INDETERMINATE, reasons
    elif enumerated == 0 and visibility.n_live_tup > 0:
        reasons.append(
            f"{RELATION} reads as EMPTY to this session while "
            f"pg_stat_user_tables.n_live_tup is {visibility.n_live_tup}. "
            "n_live_tup is an estimate and is not asserted exactly, but no "
            "amount of statistics drift turns a populated table into a zero "
            "count: this is the row-level-security blindness signature that "
            "made the predecessor migration's own pre-guard report PASS "
            "against 229 unread rows on onex-dev."
        )
        return VERDICT_INDETERMINATE, reasons

    ambiguous = [item for item in census if item.distinct_registry_tenants > 1]
    if ambiguous:
        for item in ambiguous:
            reasons.append(
                f"{item.value!r} resolves to {item.distinct_registry_tenants} "
                "DIFFERENT registry tenants -- the conversion would pick one "
                "arbitrarily. Refusing."
            )
        return VERDICT_REFUSED, reasons

    blockers = [
        item for item in census if item.surviving_rows > 0 and not item.resolves
    ]
    if blockers:
        for item in blockers:
            reasons.append(
                f"{item.value!r} -- {item.surviving_rows} row(s) survive the "
                f"debris delete -- {item.form}"
            )
        return VERDICT_REFUSED, reasons

    return VERDICT_PASS, reasons


def render(
    visibility: Visibility,
    mode: str,
    census: list[ValueCensus],
    verdict: str,
    reasons: list[str],
) -> str:
    lines: list[str] = []
    lines.append(f"relation                 {RELATION}")
    lines.append(f"relrowsecurity           {visibility.row_security_enabled}")
    lines.append(f"relforcerowsecurity      {visibility.row_security_forced}")
    lines.append(f"row_security_active      {visibility.row_security_active}")
    lines.append(f"n_live_tup (estimate)    {visibility.n_live_tup}")
    lines.append(f"enumeration mode         {mode}")
    enumerated = sum(item.rows for item in census)
    debris = sum(item.debris_rows for item in census)
    lines.append(f"enumerated total         {enumerated}")
    lines.append(f"  of which debris        {debris} (deleted by exact correlation_id)")
    lines.append(f"  surviving the delete   {enumerated - debris}")
    lines.append("")
    lines.append("tenant_id enumeration:")
    for item in census:
        lines.append(
            f"  {item.value:<40} {item.rows:>5} row(s)"
            + (f" ({item.debris_rows} debris)" if item.debris_rows else "")
            + f"  -> {item.form}"
        )
    lines.append("")
    if verdict == VERDICT_PASS:
        lines.append(
            f"RECONCILED: enumerated total {enumerated} == n_live_tup "
            f"{visibility.n_live_tup}"
        )
        lines.append("UNRESOLVABLE SET: EMPTY")
        lines.append("VERDICT: PASS")
    else:
        lines.append(f"VERDICT: {verdict}")
        for reason in reasons:
            lines.append(f"  - {reason}")
        lines.append(
            "  PASS is deliberately NOT printed. A readiness answer that has "
            "not been reconciled is not a clean bill of health."
        )
    return "\n".join(lines)


def build_client(args: argparse.Namespace) -> PsqlClient:
    if args.psql_exec:
        try:
            argv = tuple(json.loads(args.psql_exec))
        except json.JSONDecodeError as exc:
            raise ProbeError(f"--psql-exec is not valid JSON: {exc}") from exc
        if not argv or not all(isinstance(item, str) for item in argv):
            raise ProbeError("--psql-exec must be a non-empty JSON array of strings")
        return PsqlClient(argv=argv, database=args.database)
    if not args.dsn:
        raise ProbeError("one of --dsn or --psql-exec is required")
    return PsqlClient(argv=("psql", args.dsn), database=None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only readiness check for the delegation_events tenant_id "
            "TEXT->UUID conversion (OMN-15683). Refuses to print PASS unless "
            "its enumeration reconciles."
        )
    )
    parser.add_argument("--dsn", help="libpq DSN or URI for the target database")
    parser.add_argument(
        "--psql-exec",
        help='JSON array psql prefix, e.g. \'["ssh","host","docker","exec","-i","pg","psql","-U","postgres"]\'',
    )
    parser.add_argument("--database", help="database name (with --psql-exec)")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help=(
            "extra tenant identity to try when the table is policy-blinded; "
            "repeatable. Candidates are reconstruction guesses only -- one is "
            "never treated as resolvable."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    args = parser.parse_args(argv)

    try:
        client = build_client(args)
        visibility = read_visibility(client)
        if visibility.row_security_active:
            mode = MODE_RECONSTRUCTED
            candidates = sorted(
                set(read_mirror_candidates(client))
                | set(DEFAULT_EXTRA_CANDIDATES)
                | set(args.candidate)
            )
            raw = census_reconstructed(client, candidates)
        else:
            mode = MODE_DIRECT
            raw = census_direct(client)
        classified = classify(client, [value for value, _, _ in raw])
    except ProbeError as exc:
        print(f"PROBE ERROR: {exc}", file=sys.stderr)
        return EXIT_PROBE_ERROR

    census = [
        ValueCensus(
            value=value,
            rows=rows,
            debris_rows=debris,
            resolves_by_slug=classified.get(value, (False, False, 0))[0],
            resolves_by_uuid=classified.get(value, (False, False, 0))[1],
            distinct_registry_tenants=classified.get(value, (False, False, 0))[2],
        )
        for value, rows, debris in raw
    ]
    verdict, reasons = evaluate(visibility, mode, census)

    if args.json:
        print(
            json.dumps(
                {
                    "relation": RELATION,
                    "mode": mode,
                    "verdict": verdict,
                    "reasons": reasons,
                    "n_live_tup": visibility.n_live_tup,
                    "row_security_active": visibility.row_security_active,
                    "enumerated_total": sum(item.rows for item in census),
                    "census": [
                        {
                            "value": item.value,
                            "rows": item.rows,
                            "debris_rows": item.debris_rows,
                            "surviving_rows": item.surviving_rows,
                            "resolution": item.form,
                        }
                        for item in census
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render(visibility, mode, census, verdict, reasons))

    if verdict == VERDICT_PASS:
        return EXIT_PASS
    if verdict == VERDICT_REFUSED:
        return EXIT_REFUSED
    return EXIT_INDETERMINATE


if __name__ == "__main__":
    raise SystemExit(main())
