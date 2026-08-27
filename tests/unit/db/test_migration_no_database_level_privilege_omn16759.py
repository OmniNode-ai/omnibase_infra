# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""No deployable migration may demand a DATABASE-level privilege (OMN-16759).

## The class this closes

``CREATE SCHEMA`` requires the ``CREATE`` privilege **on the database**. Neither
migration role holds it on the managed (RDS) lane: those databases were created
by the instance's master user, not by ``omninode_infra``'s ``init-databases`` Job
(which only ever targets the in-cluster Postgres). ``IF NOT EXISTS`` does not
rescue the statement either -- Postgres checks the privilege BEFORE it checks
existence, so the statement fails even where the schema is already present.

That has now cost two full deploy stalls on two different files:

* **OMN-16249** -- ``nodes/node_projection_registration/0005_create_projection_watermarks.sql``
  failed with ``permission denied for database omnidash_analytics`` and stalled
  deploy run ``32301533344``. Fixed by re-vendoring the file to ASSERT the schema
  via a ``pg_catalog.pg_namespace`` probe instead of creating it.
* **OMN-16759** -- ``100_create_gateway_link_health.sql`` (landed by OMN-15570)
  reintroduced the identical statement and failed with ``permission denied for
  database omnibase_infra``, aborting deploy run ``33080116991``. The migrate Job
  is migration-order 1 of 6 and runs BEFORE the overlay apply and the runtime
  digest pin, so this blocked EVERY staging deploy, not only the one that
  surfaced it.

Two occurrences of one class, ~8 days apart, both found in production rather
than pre-merge. This module is the pre-merge gate: a third occurrence fails here
instead of at deploy time.

## What this asserts

No deployable forward migration -- flat (``docker/migrations/forward/*.sql``) or
node-owned (``docker/migrations/forward/nodes/<node>/*.sql``) -- may contain a
``CREATE SCHEMA``, ``CREATE DATABASE``, or ``ALTER DATABASE`` statement.

``CREATE EXTENSION`` is deliberately NOT in scope: it is used by ~15 migrations
that demonstrably apply on the managed lane, so it is not a privilege the
migration roles lack.

## The one exemption, and why it is not an allowlist

``098_create_omninode_internal_schema.sql`` still carries a ``CREATE SCHEMA``. It
is exempt only because it is declared in
``docker/migrations/forward/cross-database-flat-migrations.yaml`` -- the OMN-15819
ledger of flat migrations whose ``\\connect`` names a database the runner never
connects to. Those files have no execution path at all (the runner prints
``UNDELIVERABLE`` and moves on), so their SQL cannot fail a deploy. The exemption
is therefore derived from an existing, separately-gated ledger rather than from a
list maintained here: a file stops being exempt the moment it stops being
ledgered as undeliverable, with no edit to this test.

``_ledger/bootstrap.sql`` is out of scope for the same reason the application
database SQL gate excludes it (``_NON_DEPLOYABLE_SQL_EXACT_PATHS`` in
``scripts/ci/check_application_database_sql.py``): it is the ledger bootstrap the
runner executes on its own terms, not a migration in the applied corpus.

Ticket: OMN-16759
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
CROSS_DB_LEDGER = FORWARD_DIR / "cross-database-flat-migrations.yaml"

# Statements that require a privilege ON THE DATABASE. Anchored at line start so
# a prose mention inside a `--` comment (this repo's migrations carry long
# rationale headers that quote the forbidden statement on purpose) is not a hit.
_DATABASE_LEVEL_DDL = re.compile(
    r"^\s*(?:CREATE\s+SCHEMA|CREATE\s+DATABASE|ALTER\s+DATABASE)\b",
    re.IGNORECASE,
)
_COMMENT_LINE = re.compile(r"^\s*--")


def _deployable_migrations() -> list[Path]:
    """Every forward migration the runners actually execute, both loops."""
    flat = sorted(FORWARD_DIR.glob("*.sql"))
    node = sorted(FORWARD_DIR.glob("nodes/*/*.sql"))
    return flat + node


def _undeliverable_cross_db_files() -> frozenset[str]:
    """Filenames the OMN-15819 ledger declares as having no execution path."""
    ledger = yaml.safe_load(CROSS_DB_LEDGER.read_text(encoding="utf-8"))
    return frozenset(
        str(entry["file"])
        for entry in ledger["entries"]
        if entry["disposition"] == "undeliverable"
    )


def _database_level_statements(sql: str) -> list[str]:
    return [
        line.strip()
        for line in sql.splitlines()
        if not _COMMENT_LINE.match(line) and _DATABASE_LEVEL_DDL.match(line)
    ]


def test_the_corpus_under_test_is_not_empty() -> None:
    """Anti-vacuity: a glob that matches nothing would pass every assertion."""
    migrations = _deployable_migrations()

    assert len(migrations) > 100, (
        f"only {len(migrations)} migrations discovered under {FORWARD_DIR} -- "
        "the glob is wrong and this gate is vacuous"
    )


def test_the_exemption_ledger_is_not_empty() -> None:
    """Anti-vacuity in the other direction: prove the ledger really parses."""
    exempt = _undeliverable_cross_db_files()

    assert "098_create_omninode_internal_schema.sql" in exempt, (
        "the OMN-15819 cross-DB ledger no longer declares 098 undeliverable; "
        "if 098 became deliverable it must lose its CREATE SCHEMA, and this "
        "test's exemption path must be re-derived rather than widened"
    )


@pytest.mark.parametrize(
    "migration",
    _deployable_migrations(),
    ids=lambda path: path.name,
)
def test_no_deployable_migration_demands_a_database_level_privilege(
    migration: Path,
) -> None:
    """OMN-16249 + OMN-16759: CREATE SCHEMA is not executable by either role."""
    if migration.name in _undeliverable_cross_db_files():
        pytest.skip(
            f"{migration.name} is ledgered UNDELIVERABLE (OMN-15819) -- the "
            "runner never executes its SQL"
        )

    offending = _database_level_statements(migration.read_text(encoding="utf-8"))

    assert not offending, (
        f"{migration.relative_to(REPO_ROOT)} issues a DATABASE-level DDL "
        f"statement: {offending}. CREATE SCHEMA needs CREATE on the DATABASE, "
        "which neither role_omnibase_infra (flat loop, omnibase_infra) nor "
        "role_omnidash (node loop, omnidash_analytics) holds on the managed "
        "lane -- and IF NOT EXISTS does not help, because Postgres checks the "
        "privilege before it checks existence. This aborts the migrate Job, "
        "which runs BEFORE overlay-apply and the runtime digest pin, so it "
        "blocks every staging deploy. Assert the schema instead (see "
        "nodes/node_projection_registration/0005_create_projection_watermarks.sql "
        "for the pg_catalog.pg_namespace probe), or target a schema that "
        "already exists in the database this loop connects to."
    )
