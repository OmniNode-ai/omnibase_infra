# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15336 — the compose runner must honour the same node-migration fence as
the k8s runner.

Defect, found by the 2026-07-28 operator-ordered migration audit and confirmed
live: the operator fence gating the tenant-RLS node migrations existed ONLY in
``omninode_infra/k8s/migrations/omnibase-infra-migrate.yaml``.
``scripts/run-forward-migrations.sh`` — the runner every *compose* lane executes
(dev, stability-test, judge, prod all mount it; see
``docker/catalog/services/forward-migration.yaml``) — had no fence and no skip
branch of any kind in its node loop, so it applied every ``nodes/*/*.sql`` it
discovered.

Live readback 2026-07-28/29 on ``.201`` (read-only, four lanes, db
``omnidash_analytics``):

===================  ===========================  ==========================
lane                 gated ids in ledger          relforcerowsecurity tables
===================  ===========================  ==========================
dev (compose)        all 6 (+ registration 0002)  6
stability-test       all 6 (+ registration 0002)  6
prod                 registration 0000/0001 only  0
judge                registration 0000/0001 only  0
===================  ===========================  ==========================

So the breach is wider than the ticket recorded (stability-test, the designated
proof lane, is also over the fence), AND the prod + judge compose lanes are
clean but *pending*: without this fence the next forward-migration run on either
would apply the gated tenant-RLS migrations unattended. Rolling back the two
breached lanes is deliberately NOT part of this change — that disposition is an
outstanding operator decision.

THE SEAM (cross-repo, and drift-prone by construction)
------------------------------------------------------
Both runners walk the SAME vendored SQL tree
(``docker/migrations/forward/nodes/<node>/*.sql``, kept in sync by
``scripts/sync-node-migrations.sh``) and both mint the SAME id
``node:<node>:<filename>``. The fence is therefore a shared list over a shared
id space, matched field-by-field:

===========================  ==========================================  ===========================================
field                        omninode_infra k8s Job                      this repo's compose runner
===========================  ==========================================  ===========================================
id grammar                   ``node:${node_name}:$(basename $sql_file)`` ``node:${node_name}:${filename}``
list contents + ORDER        ``FENCED_NODE_MIGRATION_IDS=( ... )``       ``FENCED_NODE_MIGRATION_IDS="..."``
match                        exact string equality per element           ``grep -Fxq`` (exact whole-line)
position in loop             before the already-applied probe            before the already-applied probe
on match                     log + ``node_skipped++`` + ``continue``     log + ``NODE_SKIPPED++`` + ``continue``
ledger row written           NO                                          NO
env-overridable              NO (literal array)                          NO (unconditional assignment)
===========================  ==========================================  ===========================================

Syntax differs and must: the k8s Job runs ``bash`` (arrays, ``set -euo
pipefail``); this runner is ``#!/bin/sh`` under busybox ash in the migration
container, where arrays do not exist. The SEAM is the id list and the
semantics, not the shell dialect.

"Skip + record" means record THE SKIP — counted into ``NODE_SKIPPED`` and named
on stdout. A fenced id is deliberately NOT inserted into ``schema_migrations``,
matching the k8s runner: a ledger row would make the eventual un-fencing a
silent no-op (the runner would read the row, call it already applied, and never
run the migration).

DRIFT HAZARD / follow-up
------------------------
There is no single source of truth for this list. It is duplicated in two repos
and today only prose and these tests keep them equal.

* Always on, gating every PR: ``EXPECTED_FENCE`` below pins the list read from
  ``omninode_infra@dev`` at authoring time, asserted exact and IN ORDER.
* Opt-in: ``test_fence_matches_omninode_infra_k8s_runner`` diffs this repo's
  list against the live k8s manifest. It is opt-in on purpose — it depends on
  an ``omninode_infra`` checkout whose freshness this repo cannot guarantee,
  and an always-on version went RED on the ``.200`` build host purely because
  that clone sat two commits behind dev. False REDs from another repo's local
  staleness are worse than the gap they close.

So the residual gap is real: a fence entry added to the k8s runner alone still
lands unnoticed here until someone runs the opt-in check or updates the pin.
Single-sourcing the list (one committed manifest consumed by both runners, or a
generated block) is a cross-repo architecture change, deliberately NOT done
here; it is proposed as a follow-up in the PR body.

Live proofs drive THE ARTIFACT THAT RUNS — the shipped
``scripts/run-forward-migrations.sh``, executed against a real Postgres.
``test_fence_free_runner_applies_the_fenced_migration`` is the RED control: the
same harness against a copy of the runner with the fence mechanically stripped,
asserting it reproduces the ``.201`` breach (fenced SQL applied, ledger row
written). Without that control the GREEN proof would be unfalsifiable — it would
also pass against a harness that never reached the node loop.

Database selection matches the sibling OMN-15291 module: external
``MIGRATION_LOCK_TEST_HOST``/``_PORT``/``_USER``/``_PASSWORD``/``_DB``, else an
ephemeral ``initdb``/``pg_ctl`` cluster, else skip — unless
``REQUIRE_MIGRATION_LOCK_DB`` is set, in which case fail. A check that silently
skips is a check that does not exist.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Bound by assignment rather than `from ... import`: the OMN-15291 module owns
# the scratch-Postgres harness, and re-exporting its `pg_target` fixture as an
# import makes every test that takes it as a parameter read as a redefinition.
from tests.scripts import test_forward_migration_advisory_lock as _advisory_lock

RUNNER = _advisory_lock.RUNNER
PgTarget = _advisory_lock.PgTarget
_find_pg_binary = _advisory_lock._find_pg_binary
_psql = _advisory_lock._psql
pg_target = _advisory_lock.pg_target

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]

FENCE_BEGIN = "# ---- BEGIN operator fence — node migration ids (OMN-15336) ----"
FENCE_END = "# ---- END operator fence — node migration ids (OMN-15336) ----"
SKIP_BEGIN = "# ---- BEGIN fenced-id skip (OMN-15336) ----"
SKIP_END = "# ---- END fenced-id skip (OMN-15336) ----"

# The four OMN-14974/OMN-15313 delegation ids. Kept as their own tuple so a
# failure says WHICH half of the fence moved.
FENCED_DELEGATION_IDS = (
    "node:node_projection_delegation:0023_delegation_rls_tenant_isolation.sql",
    "node:node_projection_delegation:0024_drop_unwired_routing_columns.sql",
    "node:node_projection_delegation:0025_delegation_judge_verdict_events_tenant_id.sql",
    "node:node_projection_delegation:"
    "0026_delegation_judge_verdict_events_rls_tenant_isolation.sql",
)
# The OMN-15335 registration pair, held pending the node_service_registry FORCE
# ruling.
FENCED_REGISTRATION_IDS = (
    "node:node_projection_registration:0000_create_node_service_registry.sql",
    "node:node_projection_registration:0001_add_heartbeat_columns.sql",
)
# Source of truth at authoring time: omninode_infra@dev commit f436ca6,
# k8s/migrations/omnibase-infra-migrate.yaml, FENCED_NODE_MIGRATION_IDS.
EXPECTED_FENCE = FENCED_DELEGATION_IDS + FENCED_REGISTRATION_IDS

# An id that is NOT fenced, used by the live proofs as the discriminator: if the
# node loop applied nothing at all, the fenced-ids-absent assertion would pass
# vacuously.
UNFENCED_CONTROL_ID = "node:node_projection_delegation:0099_unfenced_control.sql"


def _runner_text() -> str:
    return RUNNER.read_text()


def extract_fence_block(text: str | None = None) -> str:
    """Return the fence definition block, markers stripped.

    Raises with a specific message if the markers are missing or duplicated —
    that is how a silent removal of the fence surfaces as a test failure rather
    than as a vacuous pass.
    """
    return _extract_marked(
        text if text is not None else _runner_text(), FENCE_BEGIN, FENCE_END
    )


def extract_skip_branch(text: str | None = None) -> str:
    """Return the in-loop fenced-id skip branch, markers stripped."""
    return _extract_marked(
        text if text is not None else _runner_text(), SKIP_BEGIN, SKIP_END
    )


def _extract_marked(text: str, begin: str, end: str) -> str:
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == begin]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == end]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        msg = (
            f"scripts/run-forward-migrations.sh: expected exactly one block "
            f"delimited by {begin!r} / {end!r} (found {len(starts)} begin / "
            f"{len(ends)} end markers)"
        )
        raise AssertionError(msg)
    return "\n".join(lines[starts[0] + 1 : ends[0]]) + "\n"


def strip_fence(text: str) -> str:
    """The pre-OMN-15336 runner: byte-identical minus both fence blocks.

    Derived from the shipped artifact rather than pinned as a copy, so the RED
    control can never drift away from the thing it is the control for.
    """
    out = text
    for begin, end in ((FENCE_BEGIN, FENCE_END), (SKIP_BEGIN, SKIP_END)):
        lines = out.splitlines(keepends=True)
        start = next(i for i, ln in enumerate(lines) if ln.strip() == begin)
        stop = next(i for i, ln in enumerate(lines) if ln.strip() == end)
        out = "".join(lines[:start] + lines[stop + 1 :])
    return out


def parse_shell_fence_list(block: str) -> tuple[str, ...]:
    """Parse ``FENCED_NODE_MIGRATION_IDS="a\\nb\\n..."`` into an ordered tuple."""
    match = re.search(
        r'FENCED_NODE_MIGRATION_IDS="\\?\n?(?P<body>[^"]*)"', block, re.DOTALL
    )
    assert match is not None, (
        "FENCED_NODE_MIGRATION_IDS assignment not found in the fence block"
    )
    return tuple(
        line.strip() for line in match.group("body").splitlines() if line.strip()
    )


# --------------------------------------------------------------------------
# Static assertions — no database required, so they gate every PR.
# --------------------------------------------------------------------------


def test_runner_carries_the_operator_fence() -> None:
    block = extract_fence_block()
    assert "FENCED_NODE_MIGRATION_IDS=" in block, (
        "the OMN-15336 block must actually define the fence list"
    )
    assert "is_fenced_node_migration()" in block, (
        "the fence block must define the predicate the node loop calls"
    )


def test_fence_is_exactly_the_expected_ids_in_order() -> None:
    """Exact-set and ORDER, not containment.

    Silently dropping one id while adding another is precisely the regression
    this assertion exists to catch, and order is asserted because the two repos'
    lists are diffed element-for-element.
    """
    found = parse_shell_fence_list(extract_fence_block())
    assert found == EXPECTED_FENCE, (
        "the operator fence changed. Expected exactly:\n  "
        + "\n  ".join(EXPECTED_FENCE)
        + "\nFound:\n  "
        + "\n  ".join(found)
    )
    assert found[: len(FENCED_DELEGATION_IDS)] == FENCED_DELEGATION_IDS, (
        "the OMN-14974 delegation fence was disturbed"
    )
    assert found[len(FENCED_DELEGATION_IDS) :] == FENCED_REGISTRATION_IDS, (
        "the OMN-15335 registration hold is not the exact expected pair"
    )


def test_every_fenced_id_names_a_real_vendored_sql_file() -> None:
    """A typo'd id fences nothing and fails open, silently."""
    nodes_root = REPO_ROOT / "docker" / "migrations" / "forward" / "nodes"
    for fenced in EXPECTED_FENCE:
        _, node_name, filename = fenced.split(":", 2)
        path = nodes_root / node_name / filename
        assert path.is_file(), (
            f"fenced id {fenced} names {path.relative_to(REPO_ROOT)}, which does "
            "not exist — the fence entry matches nothing and gates nothing"
        )


def test_fence_is_not_overridable_by_environment() -> None:
    """Only a COMMITTED fence is honoured — same rule as the skip-manifest."""
    block = extract_fence_block()
    offenders = [
        ln
        for ln in block.splitlines()
        if re.search(r"FENCED_NODE_MIGRATION_IDS=\$?\{?FENCED_NODE_MIGRATION_IDS", ln)
        or re.search(r'FENCED_NODE_MIGRATION_IDS="\$\{', ln)
    ]
    assert not offenders, (
        "the fence list must be assigned unconditionally; a "
        "${FENCED_NODE_MIGRATION_IDS:-...} form lets an operator env var empty "
        f"the fence: {offenders}"
    )


def test_fence_is_checked_before_the_ledger_probe_and_the_apply() -> None:
    """Ordering is load-bearing: a fence evaluated after the already-applied
    probe could be defeated by a probe that fails open, and a fence evaluated
    after ``psql -f`` would not be a fence at all.
    """
    text = _runner_text()
    # Scope to the NODE loop: the flat loop above it carries a byte-identical
    # already-applied probe, so a whole-file index would compare against the
    # wrong occurrence and the assertion would be meaningless.
    node_loop = text[text.index("Auto-discover and apply node-owned migrations") :]
    fence_call = node_loop.index('if is_fenced_node_migration "${migration_id}"')
    probe = node_loop.index("already_applied=$(psql")
    apply_sql = node_loop.index('-v ON_ERROR_STOP=1 -f "$migration_file"')
    assert fence_call < probe < apply_sql, (
        "the fenced-id check must precede the already-applied probe, which must "
        f"precede the apply (node-loop offsets: fence={fence_call} "
        f"probe={probe} apply={apply_sql})"
    )


def test_fenced_skip_never_records_a_ledger_row() -> None:
    """Recording a fenced id would make un-fencing a silent no-op."""
    branch = extract_skip_branch()
    assert "continue" in branch, "the fence branch must skip the migration"
    assert "INSERT" not in branch.upper(), (
        "a fenced migration must NOT be written to schema_migrations — the "
        "runner would then read the row, call it already applied, and never run "
        f"it once un-fenced:\n{branch}"
    )
    assert "psql" not in branch, (
        f"the fence branch must not touch the database at all:\n{branch}"
    )
    assert "NODE_SKIPPED=$((NODE_SKIPPED + 1))" in branch, (
        "the skip must be RECORDED in the run's skipped counter, so a fenced "
        "run is distinguishable from a run that discovered nothing"
    )


def test_fence_predicate_is_posix_sh() -> None:
    """The container shell is busybox ash; bashisms would fail only in prod."""
    assert _runner_text().splitlines()[0] == "#!/bin/sh"
    block = extract_fence_block()
    assert not re.search(r"^\s*local\s", block, re.MULTILINE), (
        "`local` is not POSIX and is unavailable in every /bin/sh this runner "
        "may execute under"
    )
    assert "=(" not in block, (
        "bash arrays do not exist in busybox ash; the fence is a newline-"
        "delimited string matched with grep -Fxq"
    )


K8S_MANIFEST_RELPATH = "k8s/migrations/omnibase-infra-migrate.yaml"


def _omninode_infra_root() -> Path | None:
    """Locate a reachable ``omninode_infra`` checkout."""
    candidates = []
    explicit = os.environ.get("OMNINODE_INFRA_ROOT")
    if explicit:
        candidates.append(Path(explicit))
    omni_home = os.environ.get("OMNI_HOME")
    if omni_home:
        candidates.append(Path(omni_home) / "omninode_infra")
    # Worktrees live at $OMNI_HOME/omni_worktrees/<ticket>/<repo>; the canonical
    # clones are three levels up. A plain sibling checkout is covered too.
    candidates.append(REPO_ROOT.parents[2] / "omninode_infra")
    candidates.append(REPO_ROOT.parent / "omninode_infra")
    for root in candidates:
        if (root / K8S_MANIFEST_RELPATH).is_file():
            return root
    return None


def _k8s_manifest_source(root: Path) -> tuple[str, str]:
    """Return (manifest text, provenance) for the k8s runner.

    ``origin/dev`` is preferred over the working tree: dev is what actually
    deploys, and a checkout parked on an older commit is the common case on a
    shared build host. Provenance is returned so a failure can be read as
    "stale clone" rather than misdiagnosed as real drift — a stale canonical
    clone producing phantom findings is a known, repeated trap here, and
    ``git show`` reads whatever ``origin/dev`` was last FETCHED to, which this
    test deliberately does not refresh (tests do no network I/O).
    """
    show = subprocess.run(
        ["git", "-C", str(root), "show", f"origin/dev:{K8S_MANIFEST_RELPATH}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if show.returncode == 0:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "origin/dev"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return show.stdout, f"{root} origin/dev@{sha or '?'} (last local fetch)"
    return (
        (root / K8S_MANIFEST_RELPATH).read_text(),
        f"{root / K8S_MANIFEST_RELPATH} (working tree; origin/dev unreadable)",
    )


def test_fence_matches_omninode_infra_k8s_runner() -> None:
    """THE cross-repo seam assertion: the two lists must be identical.

    OPT-IN via ``REQUIRE_CROSS_REPO_FENCE_PARITY`` — deliberately, and this is
    the honest shape rather than the convenient one. The comparison depends on
    an artifact outside this repo whose freshness this repo cannot guarantee:
    run against a stale ``omninode_infra`` clone it reports drift that does not
    exist (measured — the ``.200`` build host's clone sat 2 commits behind dev
    while this change was being gated, and an always-on version of this test
    went RED there for exactly that reason). Gating every PR on another repo's
    local checkout freshness manufactures false REDs, so the always-on leg is
    ``EXPECTED_FENCE`` above; this is the sharper check you run deliberately,
    after fetching, on a host that has both repos.
    """
    if not os.environ.get("REQUIRE_CROSS_REPO_FENCE_PARITY"):
        pytest.skip(
            "cross-repo fence diff is opt-in: fetch omninode_infra, then run "
            "with REQUIRE_CROSS_REPO_FENCE_PARITY=1 (optionally "
            "OMNINODE_INFRA_ROOT=<path>). The pinned EXPECTED_FENCE assertion "
            "gates every PR regardless."
        )

    root = _omninode_infra_root()
    if root is None:
        pytest.fail(
            "REQUIRE_CROSS_REPO_FENCE_PARITY is set but no omninode_infra "
            "checkout was found (tried OMNINODE_INFRA_ROOT, "
            "$OMNI_HOME/omninode_infra, and sibling paths)"
        )

    text, provenance = _k8s_manifest_source(root)
    match = re.search(r"FENCED_NODE_MIGRATION_IDS=\((?P<body>.*?)\)", text, re.DOTALL)
    assert match is not None, (
        f"{provenance}: FENCED_NODE_MIGRATION_IDS array not found — the k8s "
        "runner lost its fence, or the array was reshaped"
    )
    k8s_fence = tuple(re.findall(r'"([^"]+)"', match.group("body")))
    compose_fence = parse_shell_fence_list(extract_fence_block())
    assert compose_fence == k8s_fence, (
        "CROSS-REPO FENCE DRIFT: the compose runner and the k8s runner gate "
        "different sets of node migrations over the SAME id space.\n"
        f"  compose ({RUNNER.relative_to(REPO_ROOT)}):\n    "
        + "\n    ".join(compose_fence)
        + f"\n  k8s ({provenance}):\n    "
        + "\n    ".join(k8s_fence)
        + "\nOnly in k8s: "
        + str(sorted(set(k8s_fence) - set(compose_fence)))
        + "\nOnly in compose: "
        + str(sorted(set(compose_fence) - set(k8s_fence)))
        + "\nBEFORE treating this as real drift, confirm the k8s source above "
        "is the live dev tip: `git -C <omninode_infra> fetch origin dev` and "
        "re-run. A clone parked behind dev reports drift that does not exist."
    )


# --------------------------------------------------------------------------
# Live proofs against a real Postgres.
# --------------------------------------------------------------------------

MARKER_PREFIX = "omn15336_marker_"


def _marker_for(migration_id: str) -> str:
    """A table name unique per migration id and legal as an identifier."""
    return MARKER_PREFIX + re.sub(r"\W", "_", migration_id.split(":", 2)[2])[:40]


@pytest.fixture
def node_tree(tmp_path: Path) -> Path:
    """A vendored-shaped node tree: every fenced id plus one unfenced control.

    Each file creates a marker table, so "was this applied?" is answered by the
    database, not by parsing the runner's own log.
    """
    forward = tmp_path / "migrations" / "forward"
    forward.mkdir(parents=True)
    # A flat migration must exist for the infra phase; keep it trivial.
    (forward / "001_noop.sql").write_text("SELECT 1;\n")

    for migration_id in (*EXPECTED_FENCE, UNFENCED_CONTROL_ID):
        _, node_name, filename = migration_id.split(":", 2)
        node_dir = forward / "nodes" / node_name
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / filename).write_text(
            f"CREATE TABLE public.{_marker_for(migration_id)} (id INT);\n"
        )
    return forward


@pytest.fixture
def node_db(pg_target: PgTarget) -> Iterator[str]:
    """A SEPARATE node database, as compose configures (omnidash_analytics)."""
    admin = PgTarget(
        host=pg_target.host,
        port=pg_target.port,
        user=pg_target.user,
        password=pg_target.password,
        dbname="postgres",
    )
    name = f"omn15336_node_{int(time.time() * 1000) % 100_000_000}"
    _psql(admin, f'CREATE DATABASE "{name}"')
    try:
        yield name
    finally:
        _psql(admin, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _runner_env(
    target: PgTarget, migrations: Path, node_db_name: str
) -> dict[str, str]:
    psql_dir = str(Path(_find_pg_binary("psql") or "psql").parent)
    return {
        **os.environ,
        "PATH": f"{psql_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "POSTGRES_USER": target.user,
        "POSTGRES_PASSWORD": target.password,
        "POSTGRES_HOST": target.host,
        "POSTGRES_PORT": str(target.port),
        "POSTGRES_DB": target.dbname,
        "NODE_POSTGRES_DB": node_db_name,
        "MIGRATIONS_DIR": str(migrations),
        "NODE_MIGRATIONS_DIR": str(migrations / "nodes"),
        "MIGRATION_LOCK_WAIT_SECONDS": "60",
    }


def _run(
    runner: Path, target: PgTarget, migrations: Path, node_db_name: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/sh", str(runner)],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env=_runner_env(target, migrations, node_db_name),
    )


def _table_exists(target: PgTarget, dbname: str, table: str) -> bool:
    scoped = PgTarget(
        host=target.host,
        port=target.port,
        user=target.user,
        password=target.password,
        dbname=dbname,
    )
    return _psql(scoped, f"SELECT to_regclass('public.{table}') IS NOT NULL") == "t"


def _ledger_ids(target: PgTarget, dbname: str) -> set[str]:
    scoped = PgTarget(
        host=target.host,
        port=target.port,
        user=target.user,
        password=target.password,
        dbname=dbname,
    )
    rows = _psql(scoped, "SELECT migration_id FROM public.schema_migrations")
    return {line.strip() for line in rows.splitlines() if line.strip()}


@pytest.mark.integration
def test_shipped_runner_skips_and_records_fenced_node_migrations(
    pg_target: PgTarget,
    node_tree: Path,
    node_db: str,
) -> None:
    """GREEN: every fenced id is skipped, counted, never applied, never recorded.

    The unfenced control in the same node directory IS applied — without it the
    "no fenced marker exists" assertion would pass on a runner that never
    reached the node loop.
    """
    result = _run(RUNNER, pg_target, node_tree, node_db)
    assert result.returncode == 0, result.stdout + result.stderr

    for fenced in EXPECTED_FENCE:
        assert "SKIP (operator-gated" in result.stdout and fenced in result.stdout, (
            f"{fenced} was not reported as operator-gated:\n{result.stdout}"
        )
        assert not _table_exists(pg_target, node_db, _marker_for(fenced)), (
            f"FENCE BREACH: {fenced} was APPLIED — its DDL took effect"
        )

    # The skip is RECORDED in the run summary, and only there.
    assert f"{len(EXPECTED_FENCE)} node skipped" in result.stdout, (
        f"expected {len(EXPECTED_FENCE)} node skips in the summary line:\n"
        f"{result.stdout}"
    )

    ledger = _ledger_ids(pg_target, node_db)
    assert not (ledger & set(EXPECTED_FENCE)), (
        "a fenced id was written to schema_migrations; un-fencing it later "
        f"would then be a silent no-op: {sorted(ledger & set(EXPECTED_FENCE))}"
    )

    assert _table_exists(pg_target, node_db, _marker_for(UNFENCED_CONTROL_ID)), (
        "the unfenced control migration was NOT applied — the node loop did not "
        "run, so the fenced-ids-absent assertions above are vacuous"
    )
    assert UNFENCED_CONTROL_ID in ledger, (
        "the unfenced control was applied but not recorded — the ledger write "
        "path is broken, so 'no fenced row' proves nothing"
    )


@pytest.mark.integration
def test_fence_free_runner_applies_the_fenced_migration(
    pg_target: PgTarget,
    node_tree: Path,
    node_db: str,
    tmp_path: Path,
) -> None:
    """RED control: strip the fence and the ``.201`` breach reproduces exactly.

    Derived mechanically from the shipped runner, so this control cannot drift
    away from the artifact it is the control for.
    """
    legacy = tmp_path / "run-forward-migrations.prefence.sh"
    legacy.write_text(strip_fence(_runner_text()))

    result = _run(legacy, pg_target, node_tree, node_db)
    assert result.returncode == 0, result.stdout + result.stderr

    applied = [
        f for f in EXPECTED_FENCE if _table_exists(pg_target, node_db, _marker_for(f))
    ]
    assert applied == list(EXPECTED_FENCE), (
        "the fence-free runner was expected to apply EVERY gated migration (the "
        "OMN-15336 defect, reproduced live on the .201 dev and stability-test "
        f"lanes) but applied only {applied}. The GREEN proof above cannot "
        "distinguish the fixed runner from this one and is vacuous.\n"
        f"{result.stdout}"
    )
    ledger = _ledger_ids(pg_target, node_db)
    assert set(EXPECTED_FENCE) <= ledger, (
        "the fence-free runner did not record the gated ids either — the RED "
        f"control does not reproduce the live ledger state: {sorted(ledger)}"
    )
    assert "operator-gated" not in result.stdout, (
        "the fence block was not actually stripped from the RED control"
    )
