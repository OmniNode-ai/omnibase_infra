# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Role DDL in a migration must degrade to a named error, never a raw abort (OMN-17301).

## Why this exists as a SECOND gate rather than a widening of the first

``test_migration_no_database_level_privilege_omn16759.py`` already guards this
family, and it did not catch OMN-17301. That is not an oversight in the older
module -- it is a scope boundary that was correct for what it asserted and wrong
for what the class actually is. Its matcher is::

    ^\\s*(?:CREATE\\s+SCHEMA|CREATE\\s+DATABASE|ALTER\\s+DATABASE)\\b

Every statement in it needs a privilege **on a database object**, and its remedy
is the OMN-16249 one: stop creating the object, assert it instead, or move the
statement to a loop whose role owns the target. That remedy is available because
another identity in the migration stream *does* hold the privilege.

``CREATE ROLE`` is a different privilege axis. Roles are **cluster**-scoped and
the attribute is ``CREATEROLE``, which *no* identity in the stream holds: the
migrate Job authenticates against the managed instance only as
``role_omnibase_infra`` (flat loop) and ``role_omnidash`` (node loop), and
omninode_infra's ``scripts/init-databases.sh`` provisions both with "no CREATEDB,
no SUPERUSER, no CREATEROLE". There is no escalation identity in the Job by
design -- the managed instance has no ``postgres`` role, and its master
credential is held in AWS Secrets Manager (terraform
``manage_master_user_password``). So relocation, the older gate's whole remedy,
is unavailable, and a matcher extended to include ``CREATE ROLE`` would have had
to either ban role migrations outright (094 and 103 both legitimately need one)
or say nothing.

The invariant that IS available is about failure shape. A migration may carry
role DDL; what it may not do is let the privilege check abort the file with a raw
``psql: ERROR: permission denied to create role``. That is what stalled
``Deploy onex-staging`` run 33341217605 at migration-order 1 of 6, before
overlay-apply and the runtime digest pin -- blocking every staging deploy on
every trigger, because the OMN-16493 resolver picks the newest CI-built bundle
fail-closed and every bundle at or after ``c5a3c2d27`` carries the file.

## What this asserts

For every deployable forward migration -- flat (``docker/migrations/forward/*.sql``)
and node-owned (``.../nodes/<node>/*.sql``):

1. Role DDL (``CREATE``/``ALTER``/``DROP ROLE``) may appear only INSIDE a
   PL/pgSQL ``DO`` block. A bare top-level statement cannot be guarded at all,
   because SQL has no handler construct outside PL/pgSQL.
2. Any ``DO`` block that carries role DDL must handle ``insufficient_privilege``.
   That is the SQLSTATE (42501) the managed lane raises, and handling it is what
   converts an opaque abort into a message that names the missing privilege and
   the remediation.

This is a failure-SHAPE gate, deliberately not a privilege ban. It does not
require the migration to succeed where the privilege is absent -- 103 still
exits non-zero there, and must, because reporting success over an absent
principal is the OMN-14950 masking outcome (``application_database.py`` binds
``tenant_projection`` -> ``tenant_projection_writer`` and OMN-16911 attests
``current_user`` on every projection connection, so a silent skip resurfaces as
total DLQ loss on the tenant projections). It requires only that the operator
reading the deploy log is told which privilege is missing and where to fix it.

Tickets: OMN-17301 (this instance), OMN-16759 / OMN-16249 (the database-level
half), OMN-15343 (the provisioning seam the remediation points at).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
CROSS_DB_LEDGER = FORWARD_DIR / "cross-database-flat-migrations.yaml"
APPLICATION_MANIFEST = FORWARD_DIR / "_ledger" / "application-migrations.tsv"

# Node migrations whose bytes are FROZEN APPLIED HISTORY and therefore cannot
# be brought into compliance by editing them. Each is declared in
# _ledger/application-migrations.tsv with a content_sha256 that the .201 dev
# lane has already recorded in platform_catalog.schema_migrations, so an
# in-place edit aborts forward-migration there (the OMN-16705 class), and
# check_migration_append_only.py refuses it. Probed read-only 2026-08-31 with
# scripts/migrations/check_migration_applied_on_lane.py:
#
#   node:node_log_persistence_effect:0000_create_log_entries.sql
#     APPLIED omnidash_analytics, applied_at 2026-08-17 02:30:59+00
#   node:node_projection_delegation_inference_response:0004_grant_tenant_projection_writer.sql
#     APPLIED omnidash_analytics, applied_at 2026-08-30 09:06:44+00
#
# A new-ordinal successor is the sanctioned escape for changing a migration,
# but it CANNOT help here: a successor does not un-run its predecessor on a
# lane that has never applied it, which is exactly the managed lane. What
# removes the hazard for these two is the provisioning seam -- once
# omninode_runtime and tenant_projection_writer exist, both files'  guarded
#   IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = ...)
# is false and the privileged statement is never reached. Both principals are
# declared in the topology and are provisioned by omninode_infra
# scripts/provision-cluster-roles.sh.
_FROZEN_APPLIED_HISTORY: dict[str, str] = {
    "0000_create_log_entries.sql": "OMN-17301",
    "0004_grant_tenant_projection_writer.sql": "OMN-17301",
}

# Migrations that DO carry the defect and whose bytes COULD be edited, but where
# the edit is blocked by a different gate on unrelated grounds. Distinct from
# _FROZEN_APPLIED_HISTORY (bytes immutable) because the remedy differs: the
# blocking condition has to be cleared first, then the entry is removed.
#
# SHRINK-ONLY, and mechanically so: test_known_outstanding_entries_still_carry_
# the_defect below asserts every entry still violates. Fix the file and that
# anti-vacuity test FAILS until the entry is deleted, so an exemption cannot
# outlive the defect it excuses.
#
# 052: the OMN-15361 application-database SQL gate lints a file IN FULL once it
# is touched at all, and 052 fails that gate in BOTH of its forms. As shipped it
# carries a procedural block whose dynamic SQL the gate cannot prove statically.
# Rewriting those blocks to be static (attempted, and measured) clears that and
# immediately surfaces the next layer: `public.waitlist_signups` and
# `public.admin_events_log` are "prohibited in public" -- omniweb-owned tables
# that no ownership manifest passed to that gate declares, and there is no
# omniweb manifest in that set at all. Clearing it therefore means settling
# omniweb's public-table ownership model, a cross-repo change on the
# OMN-15383 / OMN-16350 surface. Neither layer is caused by, nor fixable
# within, OMN-17301, so 052 stays byte-identical to dev and is tracked instead.
_KNOWN_OUTSTANDING: dict[str, str] = {
    "052_create_role_omniweb.sql": "OMN-17348",
}

# Anchored at line start so the long rationale headers these migrations carry --
# which quote the forbidden shapes on purpose -- are not hits. Comment lines are
# stripped before matching regardless.
_ROLE_DDL = re.compile(
    r"^\s*(?:CREATE|ALTER|DROP)\s+ROLE\b",
    re.IGNORECASE,
)
_COMMENT_LINE = re.compile(r"^\s*--")


def _deployable_migrations() -> list[Path]:
    """Every forward migration the runners actually execute, both loops."""
    flat = sorted(FORWARD_DIR.glob("*.sql"))
    node = sorted(FORWARD_DIR.glob("nodes/*/*.sql"))
    return flat + node


def _undeliverable_cross_db_files() -> frozenset[str]:
    """Filenames the OMN-15819 ledger declares as having no execution path.

    Derived from the ledger rather than listed here, exactly as the OMN-16759
    gate derives its own exemption: a file stops being exempt the moment it
    stops being ledgered undeliverable, with no edit to this module. These
    files carry role DDL the runner never executes -- it prints UNDELIVERABLE
    and moves on -- so their SQL cannot abort a deploy.
    """
    ledger = yaml.safe_load(CROSS_DB_LEDGER.read_text(encoding="utf-8"))
    return frozenset(
        str(entry["file"])
        for entry in ledger["entries"]
        if entry["disposition"] == "undeliverable"
    )


def _manifest_declared_artifacts() -> frozenset[str]:
    """Artifact paths declared in the canonical node-migration manifest.

    A file listed here has a recorded ``content_sha256``; the append-only guard
    freezes its bytes and the runner aborts a lane whose recorded checksum stops
    matching. This is the ledger fact the frozen-history exemption is derived
    from -- membership is not asserted by this module.
    """
    rows = APPLICATION_MANIFEST.read_text(encoding="utf-8").splitlines()
    return frozenset(row.split("\t")[0] for row in rows if row.strip())


def _strip_comments(sql: str) -> str:
    """Blank out ``--`` lines, preserving line count so offsets stay meaningful."""
    return "\n".join(
        "" if _COMMENT_LINE.match(line) else line for line in sql.splitlines()
    )


def _do_blocks(sql: str) -> list[str]:
    """Return the body of every top-level ``DO $tag$ ... $tag$`` block."""
    blocks: list[str] = []
    for match in re.finditer(r"\bDO\s+(\$[A-Za-z_]*\$)", sql, re.IGNORECASE):
        tag = match.group(1)
        start = match.end()
        end = sql.find(tag, start)
        if end == -1:
            # Unterminated block: treat the remainder as the body so its
            # contents are still subject to the handler requirement.
            blocks.append(sql[start:])
            continue
        blocks.append(sql[start:end])
    return blocks


def _role_ddl_lines(sql: str) -> list[str]:
    return [line.strip() for line in sql.splitlines() if _ROLE_DDL.match(line)]


def _role_ddl_outside_do_blocks(sql: str) -> list[str]:
    """Role DDL statements that are not inside any ``DO`` block."""
    stripped = _strip_comments(sql)
    inside = "\n".join(_do_blocks(stripped))
    inside_counts: dict[str, int] = {}
    for line in _role_ddl_lines(inside):
        inside_counts[line] = inside_counts.get(line, 0) + 1

    outside: list[str] = []
    for line in _role_ddl_lines(stripped):
        if inside_counts.get(line, 0) > 0:
            inside_counts[line] -= 1
        else:
            outside.append(line)
    return outside


def test_the_corpus_under_test_is_not_empty() -> None:
    """Anti-vacuity: a glob that matches nothing would pass every assertion."""
    migrations = _deployable_migrations()

    assert len(migrations) > 100, (
        f"only {len(migrations)} migrations discovered under {FORWARD_DIR} -- "
        "the glob is wrong and this gate is vacuous"
    )


def test_the_gate_actually_finds_role_ddl_in_the_corpus() -> None:
    """Anti-vacuity: the matcher must hit the files we know carry role DDL.

    If this fails, the matcher stopped recognising role DDL and every
    per-migration assertion below became a no-op that passes on anything.
    """
    carriers = {
        migration.name
        for migration in _deployable_migrations()
        if _role_ddl_lines(_strip_comments(migration.read_text(encoding="utf-8")))
    }

    assert "094_create_app_dashboard_role.sql" in carriers, (
        "094 provably contains CREATE ROLE app_dashboard; the matcher no longer "
        "sees it, so this gate is asserting nothing"
    )
    assert "103_create_tenant_projection_writer_role.sql" in carriers, (
        "103 provably contains CREATE ROLE tenant_projection_writer; the matcher "
        "no longer sees it, so this gate is asserting nothing"
    )


@pytest.mark.parametrize(
    "migration",
    _deployable_migrations(),
    ids=lambda path: path.name,
)
def test_role_ddl_is_never_issued_outside_a_plpgsql_block(migration: Path) -> None:
    """A bare top-level role statement has no way to handle a privilege error."""
    if migration.name in _undeliverable_cross_db_files():
        pytest.skip(
            f"{migration.name} is ledgered UNDELIVERABLE (OMN-15819) -- the "
            "runner never executes its SQL"
        )
    if migration.name in _FROZEN_APPLIED_HISTORY:
        pytest.skip(
            f"{migration.name} is frozen applied history "
            f"({_FROZEN_APPLIED_HISTORY[migration.name]}) -- its bytes cannot be "
            "edited, and the provisioning seam removes the hazard instead"
        )
    if migration.name in _KNOWN_OUTSTANDING:
        pytest.skip(
            f"{migration.name} is a KNOWN OUTSTANDING instance "
            f"({_KNOWN_OUTSTANDING[migration.name]}) -- editing it trips an "
            "unrelated gate; tracked, and asserted still-defective below"
        )

    sql = migration.read_text(encoding="utf-8")

    offending = _role_ddl_outside_do_blocks(sql)

    assert not offending, (
        f"{migration.relative_to(REPO_ROOT)} issues role DDL at the top level: "
        f"{offending}. Roles are CLUSTER-scoped and CREATE ROLE requires the "
        "CREATEROLE attribute, which no migration identity holds on the managed "
        "lane (role_omnibase_infra and role_omnidash are both provisioned "
        "NOCREATEROLE, and the instance has no postgres role this Job can "
        "authenticate as -- OMN-15343). A top-level statement cannot catch the "
        "resulting SQLSTATE 42501, so it aborts the file with a raw "
        "'permission denied to create role' and stalls the migrate Job -- which "
        "runs BEFORE overlay-apply and the runtime digest pin, blocking every "
        "staging deploy (OMN-17301). Move it inside a DO $$ ... $$ block with a "
        "'WHEN insufficient_privilege' handler that names the provisioning seam."
    )


@pytest.mark.parametrize(
    "migration",
    _deployable_migrations(),
    ids=lambda path: path.name,
)
def test_role_ddl_blocks_handle_insufficient_privilege(migration: Path) -> None:
    """OMN-17301: role DDL must fail with a named remediation, not a raw abort."""
    if migration.name in _undeliverable_cross_db_files():
        pytest.skip(
            f"{migration.name} is ledgered UNDELIVERABLE (OMN-15819) -- the "
            "runner never executes its SQL"
        )
    if migration.name in _FROZEN_APPLIED_HISTORY:
        pytest.skip(
            f"{migration.name} is frozen applied history "
            f"({_FROZEN_APPLIED_HISTORY[migration.name]}) -- its bytes cannot be "
            "edited, and the provisioning seam removes the hazard instead"
        )
    if migration.name in _KNOWN_OUTSTANDING:
        pytest.skip(
            f"{migration.name} is a KNOWN OUTSTANDING instance "
            f"({_KNOWN_OUTSTANDING[migration.name]}) -- editing it trips an "
            "unrelated gate; tracked, and asserted still-defective below"
        )

    stripped = _strip_comments(migration.read_text(encoding="utf-8"))

    unguarded = [
        block_index
        for block_index, block in enumerate(_do_blocks(stripped))
        if _role_ddl_lines(block) and "insufficient_privilege" not in block.lower()
    ]

    assert not unguarded, (
        f"{migration.relative_to(REPO_ROOT)} has DO block(s) {unguarded} that "
        "issue role DDL without a 'WHEN insufficient_privilege' handler. On the "
        "managed lane that statement raises SQLSTATE 42501 and aborts the whole "
        "file, and the operator sees only 'permission denied to create role' "
        "with no indication of which seam provisions the principal. This is the "
        "shape that stalled Deploy onex-staging run 33341217605 and, as "
        "CREATE SCHEMA, run 33080116991 before it (OMN-16759). Add the handler "
        "and RAISE with a MESSAGE naming the role and the executing identity, a "
        "DETAIL explaining why no identity in the stream holds CREATEROLE, and a "
        "HINT naming the provisioning seam. See "
        "103_create_tenant_projection_writer_role.sql for the reference shape. "
        "Handling the condition does NOT mean swallowing it -- the migration "
        "must still exit non-zero, or an absent principal is reported as success "
        "(the OMN-14950 masking class)."
    )


def test_the_exemption_is_derived_from_the_ledger_not_a_local_list() -> None:
    """Anti-vacuity: prove the ledger parses and really covers the skipped files."""
    exempt = _undeliverable_cross_db_files()

    for name in (
        "096_grant_role_omnidash_omnidash_analytics.sql",
        "099_create_omninode_internal_live_events.sql",
    ):
        assert name in exempt, (
            f"{name} carries role DDL and is skipped by this gate only because "
            "the OMN-15819 ledger declares it undeliverable. It no longer does, "
            "so either the file became deliverable (and must be guarded like "
            "103) or the ledger key changed and this exemption is now silently "
            "exempting nothing."
        )


def test_frozen_history_exemptions_are_really_frozen() -> None:
    """Anti-vacuity: every frozen-history exemption must be a real ledger fact.

    An entry that is NOT declared in the canonical manifest is not frozen at all
    -- its bytes are editable and it should have been fixed rather than exempted.
    This turns the exemption from an assertion into a derivation.
    """
    declared = _manifest_declared_artifacts()

    for name, ticket in _FROZEN_APPLIED_HISTORY.items():
        matches = [path for path in declared if path.rsplit("/", 1)[-1] == name]
        assert matches, (
            f"{name} is exempted as frozen applied history ({ticket}) but is not "
            "declared in _ledger/application-migrations.tsv, so nothing freezes "
            "its bytes. Either fix the file the way 103 was fixed, or remove the "
            "exemption -- it is currently excusing an editable migration."
        )


def test_known_outstanding_entries_still_carry_the_defect() -> None:
    """Anti-vacuity: an exemption must not outlive the defect it excuses.

    Every ``_KNOWN_OUTSTANDING`` entry is asserted to STILL have an unguarded
    role-DDL DO block. The moment someone fixes one, this fails and the entry
    must be deleted -- which is what makes the set shrink-only in mechanism
    rather than in intention. It also fails if the file is renamed or removed,
    so a dangling exemption cannot silently widen the skip.
    """
    assert _KNOWN_OUTSTANDING, "the set is empty; delete it and this test"

    by_name = {migration.name: migration for migration in _deployable_migrations()}

    for name, ticket in sorted(_KNOWN_OUTSTANDING.items()):
        migration = by_name.get(name)
        assert migration is not None, (
            f"{name} is exempted as known-outstanding ({ticket}) but no longer "
            "exists in the deployable corpus -- delete the stale entry"
        )

        stripped = _strip_comments(migration.read_text(encoding="utf-8"))
        unguarded = [
            block_index
            for block_index, block in enumerate(_do_blocks(stripped))
            if _role_ddl_lines(block) and "insufficient_privilege" not in block.lower()
        ]

        assert unguarded, (
            f"{name} is exempted as known-outstanding ({ticket}) but no longer "
            "has an unguarded role-DDL block -- the defect is fixed, so remove "
            "it from _KNOWN_OUTSTANDING and let the gate hold it to the full bar"
        )
