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

* dev (compose)   — all 6 gated ids in the ledger (+ registration 0002); 6
  tables with ``relforcerowsecurity``
* stability-test  — all 6 gated ids in the ledger (+ registration 0002); 6
  tables with ``relforcerowsecurity``
* prod            — only registration 0000/0001; ZERO forced tables
* judge           — only registration 0000/0001; ZERO forced tables

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

Each field is stated as ``<field>: k8s Job -> compose runner``.

* id grammar: ``node:${node_name}:$(basename "$sql_file")`` ->
  ``node:${node_name}:${filename}`` — identical strings.
* list contents AND order: ``FENCED_NODE_MIGRATION_IDS=( ... )`` (bash array) ->
  ``FENCED_NODE_MIGRATION_IDS="..."`` (newline-delimited string); same seven
  ids, same order.
* match: exact string equality per element -> ``grep -Fxq`` (exact whole-line).
* position in loop: before the already-applied probe -> before the
  already-applied probe.
* on match: log + ``node_skipped++`` + ``continue`` -> log + ``NODE_SKIPPED++``
  + ``continue``.
* ledger row written: NO -> NO.
* env-overridable: NO (literal array) -> NO (unconditional assignment).

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

OMN-15379 — LANE-SCOPED RELEASE (operator ruling 15, 2026-07-29)
----------------------------------------------------------------
Two changes land on top of the above.

1. ``node:node_projection_registration:0002_node_service_registry_tenant_rls.sql``
   joins the fence. It was in the k8s list from the start and missing from this
   one. That was not cosmetic: with 0000 (the CREATE) fenced and 0002 (the
   dependent ALTER) not, every COLD compose lane ran 0002 against a table that
   had never been created and died —
   ``0002_node_service_registry_tenant_rls.sql:41 ERROR: relation
   "node_service_registry" does not exist`` — taking the forward-migration
   one-shot to exit 3 and the whole lane with it (measured on ``.201``,
   2026-07-29). A fence that gates a CREATE but not its dependent ALTER is worse
   than no fence. This also makes the two lists identical again at SEVEN ids.

2. Operator ruling 15 extends node_service_registry FORCE ROW LEVEL SECURITY to
   the LAB LANE ONLY, so the compose runner gains a lane-scoped release: with
   ``ONEX_MIGRATION_LANE=dev`` the registration TRIO (0000/0001/0002) applies in
   full — CREATE, heartbeat columns, ENABLE + FORCE RLS — making the lab the
   proving ground that generates the evidence the staging un-fence is waiting
   on. The omninode_infra k8s fence is UNCHANGED.

The release is fail-closed on three independent axes, one test each:

* ``ONEX_MIGRATION_LANE`` unset -> release nothing. An UNKNOWN value -> release
  nothing, and say so on stderr.
* The release SET is committed in the runner. The env var selects among literal
  policies; it never carries ids, and the release is only consulted for ids the
  fence already covers, so a lane can only ever un-gate a SUBSET of the fence.
* The indicator is NOT in ``docker-compose.infra.yml``. Every lane overlay
  MERGES that base (stability-test's forward-migration override is a lone
  ``container_name:`` line — it inherits the base ``environment:`` wholesale),
  so a value there would be inherited by stability-test, prod, judge and any
  future lane. It lives in ``docker/docker-compose.dev-lane.yml``, which only
  the dev/lab project loads.

The live half drives the shipped runner over the REAL vendored registration SQL
BOTH WAYS against one Postgres — dev lane applies the trio and
``pg_class.relforcerowsecurity`` reads true; default lane skips it and
``node_service_registry`` does not exist. Neither leg can pass vacuously: a
runner that ignored the indicator would fail one of them whichever way it
defaulted. A fenced delegation id rides along in the same run as the negative
control, since ruling 15 released the registration trio and nothing else.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

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
# The OMN-15335/OMN-15343 registration TRIO. 0002 was in the k8s list from the
# start and missing here until OMN-15379 — a fence that gated the CREATE (0000)
# but not the dependent ALTER (0002) took every cold compose lane to exit 3 with
# `relation "node_service_registry" does not exist`.
FENCED_REGISTRATION_IDS = (
    "node:node_projection_registration:0000_create_node_service_registry.sql",
    "node:node_projection_registration:0001_add_heartbeat_columns.sql",
    "node:node_projection_registration:0002_node_service_registry_tenant_rls.sql",
)
# Source of truth at authoring time: omninode_infra@dev commit 966463a,
# k8s/migrations/omnibase-infra-migrate.yaml, FENCED_NODE_MIGRATION_IDS.
EXPECTED_FENCE = FENCED_DELEGATION_IDS + FENCED_REGISTRATION_IDS

# --- OMN-15379 lane-scoped release (operator ruling 15, 2026-07-29) ----------
# Ruling 15: node_service_registry FORCE ROW LEVEL SECURITY extends to the LAB
# LANE ONLY. The lab (compose dev lane, project `omnibase-infra`) applies the
# registration trio in full as the proving ground; the omninode_infra k8s fence
# is unchanged at all seven ids.
LANE_INDICATOR_ENV = "ONEX_MIGRATION_LANE"
DEV_LANE_VALUE = "dev"
# Spelled out rather than aliased to FENCED_REGISTRATION_IDS: if the two are
# meant to be equal, that equality is an assertion, not a definition. Aliasing
# would let a change to one silently move the other.
LANE_RELEASED_IDS = (
    "node:node_projection_registration:0000_create_node_service_registry.sql",
    "node:node_projection_registration:0001_add_heartbeat_columns.sql",
    "node:node_projection_registration:0002_node_service_registry_tenant_rls.sql",
)

BASE_COMPOSE_RELPATH = "docker/docker-compose.infra.yml"
DEV_LANE_OVERLAY_RELPATH = "docker/docker-compose.dev-lane.yml"
CATALOG_SERVICE_RELPATH = "docker/catalog/services/forward-migration.yaml"
# Every lane that MERGES the base compose file. The lane indicator must not be
# reachable from any of them.
NON_DEV_OVERLAY_RELPATHS = (
    "docker/docker-compose.stability-test.yml",
    "docker/docker-compose.prod.yml",
    "docker/docker-compose.judge.yml",
)

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


def parse_lane_release_policies(block: str) -> dict[str, tuple[str, ...]]:
    """Parse the OMN-15379 ``case "${ONEX_MIGRATION_LANE}" in ... esac`` policy.

    Returns ``{case-label: released ids}``. A label whose arm assigns the empty
    string maps to ``()`` — i.e. FULLY FENCED. A label whose arm makes no
    assignment at all maps to ``None``, which every caller treats as a defect:
    an un-assigned arm would inherit whatever the previous arm left behind.
    """
    case_match = re.search(
        r'case\s+"\$\{ONEX_MIGRATION_LANE\}"\s+in\n(?P<body>.*?)\nesac',
        block,
        re.DOTALL,
    )
    assert case_match is not None, (
        'the lane-release policy must be a `case "${ONEX_MIGRATION_LANE}" in '
        "... esac` over COMMITTED arms. If the shape moved, fix this parser — "
        "do not restate the policy here."
    )
    policies: dict[str, tuple[str, ...] | None] = {}
    for arm in case_match.group("body").split(";;"):
        label_match = re.match(r"\s*(?P<label>[^\s)]+)\)", arm)
        if label_match is None:
            continue
        label = label_match.group("label").strip('"')
        assign = re.search(
            r'LANE_RELEASED_NODE_MIGRATION_IDS="\\?\n?(?P<body>[^"]*)"',
            arm,
            re.DOTALL,
        )
        policies[label] = (
            None
            if assign is None
            else tuple(
                line.strip()
                for line in assign.group("body").splitlines()
                if line.strip()
            )
        )
    return policies  # type: ignore[return-value]


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
        "the OMN-15335/OMN-15343 registration hold is not the exact expected trio"
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
    declaration = node_loop.index('resolve_application_migration "$artifact_path"')
    probe = node_loop.index("if migration_is_applied")
    apply_sql = node_loop.index('-v ON_ERROR_STOP=1 -f "$migration_file"')
    assert fence_call < declaration < probe < apply_sql, (
        "the fenced-id check must precede declaration resolution and the "
        "canonical already-applied probe, which must precede the apply "
        f"(node-loop offsets: fence={fence_call} declaration={declaration} "
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


# --------------------------------------------------------------------------
# OMN-15379 — lane-scoped fence release. Static half.
#
# Three independent properties, one test each, so a failure names WHICH one
# broke:
#   1. the default and unknown lanes are FULLY fenced (fail-closed),
#   2. the release set is COMMITTED and is a strict subset of the fence,
#   3. the lane indicator is NOT reachable from the base compose file, so no
#      non-dev lane can inherit it.
# --------------------------------------------------------------------------


def test_lane_release_arms_are_exactly_dev_default_and_unknown() -> None:
    """Every arm must ASSIGN. An arm that falls through inherits the last value.

    ``case`` in POSIX sh does not reset the variable between arms, so an arm
    that matches but assigns nothing would silently carry whatever the previous
    arm set. That is the shape of a fail-open bug, so the arm set is pinned.
    """
    policies = parse_lane_release_policies(extract_fence_block())
    assert set(policies) == {DEV_LANE_VALUE, "", "*"}, (
        "the lane-release policy must have exactly three arms — the dev/lab "
        f"lane, the unset default, and the unknown-value catch-all. Found: "
        f"{sorted(policies)}"
    )
    unassigned = [label for label, ids in policies.items() if ids is None]
    assert not unassigned, (
        "these case arms match but never assign "
        f"LANE_RELEASED_NODE_MIGRATION_IDS, so they inherit the preceding "
        f"arm's value: {unassigned}"
    )


def test_unset_and_unknown_lane_are_fully_fenced() -> None:
    """THE fail-closed property, asserted statically as well as live below."""
    policies = parse_lane_release_policies(extract_fence_block())
    assert policies[""] == (), (
        "an unset ONEX_MIGRATION_LANE must release NOTHING — the full operator "
        f"fence applies. Found: {policies['']}"
    )
    assert policies["*"] == (), (
        "an UNKNOWN ONEX_MIGRATION_LANE must fail closed to the full fence, not "
        f"be treated as dev. Found: {policies['*']}"
    )


def test_dev_lane_releases_exactly_the_registration_trio() -> None:
    """Ruling 15 is scoped to node_service_registry and to nothing else."""
    policies = parse_lane_release_policies(extract_fence_block())
    assert policies[DEV_LANE_VALUE] == LANE_RELEASED_IDS, (
        "the dev/lab lane release set changed. Operator ruling 15 releases "
        "exactly the node_projection_registration trio:\n  "
        + "\n  ".join(LANE_RELEASED_IDS)
        + "\nFound:\n  "
        + "\n  ".join(policies[DEV_LANE_VALUE])
    )
    # The equality that the two constants are DEFINED separately to express.
    assert LANE_RELEASED_IDS == FENCED_REGISTRATION_IDS, (
        "the released trio and the fenced registration trio have diverged"
    )


def test_no_lane_can_release_anything_outside_the_fence() -> None:
    """A release is only ever consulted for an already-fenced id.

    Asserted at the list level too, so the invariant is legible without reading
    the node loop: an id in a release arm but not in the fence would be dead
    weight at best and a mis-stated policy at worst.
    """
    fence = set(parse_shell_fence_list(extract_fence_block()))
    for label, released in parse_lane_release_policies(extract_fence_block()).items():
        stray = sorted(set(released or ()) - fence)
        assert not stray, (
            f"lane '{label}' claims to release ids that the fence does not "
            f"cover: {stray}"
        )


def test_delegation_ids_are_not_releasable_on_any_lane() -> None:
    """Ruling 15 did not touch the delegation tenant-RLS hold.

    Those four gate live, actively-written tables and their un-gate is a
    separate, still-pending ruling. No lane may release them.
    """
    for label, released in parse_lane_release_policies(extract_fence_block()).items():
        leaked = sorted(set(released or ()) & set(FENCED_DELEGATION_IDS))
        assert not leaked, (
            f"lane '{label}' releases OMN-14974/OMN-15313 delegation migrations, "
            f"which no operator ruling has un-gated: {leaked}"
        )


def test_lane_release_set_is_not_supplied_by_environment() -> None:
    """The env var selects a COMMITTED policy; it never carries ids.

    Same rule as the fence list and the skip-manifest. A
    ``LANE_RELEASED_NODE_MIGRATION_IDS="${SOMETHING}"`` form would let an
    operator env var release arbitrary fenced migrations on any lane.
    """
    block = extract_fence_block()
    offenders = [
        ln
        for ln in block.splitlines()
        if re.search(r'LANE_RELEASED_NODE_MIGRATION_IDS="?\$', ln)
    ]
    assert not offenders, (
        "the release set must be assigned from literals inside the committed "
        f"case arms, never interpolated from the environment: {offenders}"
    )


def test_release_is_checked_inside_the_fenced_branch_not_beside_it() -> None:
    """Structural: the release may only ever narrow the fence.

    If ``is_lane_released_node_migration`` were called at the top of the loop
    instead of inside ``if is_fenced_node_migration``, it would become an
    independent gate over the whole node corpus rather than a carve-out of the
    fence, and its blast radius would no longer be bounded by the fence list.
    """
    branch = extract_skip_branch()
    assert "is_lane_released_node_migration" in branch, (
        "the lane-release check must live inside the fenced-id skip branch"
    )
    text = _runner_text()
    node_loop = text[text.index("Auto-discover and apply node-owned migrations") :]
    assert node_loop.count("is_lane_released_node_migration") == 1, (
        "the release predicate is called more than once in the node loop; the "
        "only sanctioned call site is inside the fenced-id branch"
    )
    fence_call = node_loop.index('if is_fenced_node_migration "${migration_id}"')
    release_call = node_loop.index(
        'if is_lane_released_node_migration "${migration_id}"'
    )
    assert fence_call < release_call, (
        "the release check must be nested INSIDE the fence check "
        f"(offsets: fence={fence_call} release={release_call})"
    )


def _forward_migration_environment(relpath: str) -> dict[str, str] | None:
    """The ``forward-migration`` service's ``environment:`` in one compose file.

    ``None`` when the file does not define that service at all — which is a
    meaningfully different statement from "defines it with no environment".
    """
    doc = yaml.safe_load((REPO_ROOT / relpath).read_text(encoding="utf-8"))
    service = (doc.get("services") or {}).get("forward-migration")
    if service is None:
        return None
    env = service.get("environment") or {}
    if isinstance(env, list):
        pairs = [item.split("=", 1) for item in env]
        return {k: (v[0] if v else "") for k, *v in pairs}
    return {str(k): str(v) for k, v in env.items()}


def test_lane_indicator_is_absent_from_the_base_compose_file() -> None:
    """THE fail-closed mechanism, asserted at the compose seam.

    Compose merges a lane overlay's ``environment:`` ON TOP of the base's — it
    is a union with the overlay winning per key. So the merged value of
    ``ONEX_MIGRATION_LANE`` is absent for a lane iff it is absent from BOTH the
    base and that lane's overlay. Every non-dev lane overlay merges
    ``docker-compose.infra.yml`` (stability-test's forward-migration override is
    a single ``container_name:`` line — it inherits the base ``environment:``
    block wholesale), so the indicator sitting in the base would be inherited by
    stability-test, prod, judge, and by any lane added later. That is fail-OPEN.

    This assertion plus ``test_only_the_dev_lane_overlay_carries_the_indicator``
    below are jointly the proof that no non-dev lane can reach it.
    """
    env = _forward_migration_environment(BASE_COMPOSE_RELPATH)
    assert env is not None, (
        f"{BASE_COMPOSE_RELPATH} no longer defines forward-migration — this "
        "check's premise moved"
    )
    assert LANE_INDICATOR_ENV not in env, (
        f"{LANE_INDICATOR_ENV} is set in {BASE_COMPOSE_RELPATH}. Every lane "
        "overlay merges that file, so this releases the fenced registration "
        "migrations on stability-test, prod, judge and every future lane. It "
        f"belongs in {DEV_LANE_OVERLAY_RELPATH}, which only the dev/lab project "
        "loads."
    )
    # The catalog manifest GENERATES the base compose service, so the same
    # statement has to hold one level up or a regenerate reintroduces it.
    catalog = yaml.safe_load(
        (REPO_ROOT / CATALOG_SERVICE_RELPATH).read_text(encoding="utf-8")
    )
    for field in ("hardcoded_env", "operational_defaults", "required_env"):
        block = catalog.get(field) or {}
        keys = block if isinstance(block, list) else list(block)
        assert LANE_INDICATOR_ENV not in keys, (
            f"{LANE_INDICATOR_ENV} is declared in {CATALOG_SERVICE_RELPATH} "
            f"under {field}; regenerating the compose catalog would put it back "
            "into the base file that every lane merges"
        )


def test_only_the_dev_lane_overlay_carries_the_indicator() -> None:
    """The dev overlay sets it to exactly ``dev``; no other overlay mentions it."""
    dev_env = _forward_migration_environment(DEV_LANE_OVERLAY_RELPATH)
    assert dev_env is not None, (
        f"{DEV_LANE_OVERLAY_RELPATH} must define the forward-migration service"
    )
    assert dev_env.get(LANE_INDICATOR_ENV) == DEV_LANE_VALUE, (
        f"{DEV_LANE_OVERLAY_RELPATH} must set "
        f"{LANE_INDICATOR_ENV}={DEV_LANE_VALUE}; found {dev_env!r}"
    )

    for relpath in NON_DEV_OVERLAY_RELPATHS:
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert LANE_INDICATOR_ENV not in text, (
            f"{relpath} mentions {LANE_INDICATOR_ENV}. Non-dev lanes must carry "
            "no lane indicator at all — the runner's fail-closed default is what "
            "keeps them fenced, and a value here (even a benign-looking one) is "
            "a release waiting on a typo."
        )


def test_dev_lane_overlay_is_wired_into_both_lane_mappings() -> None:
    """Two files own lane -> compose-file mapping; both must layer the overlay.

    ``scripts/deploy-runtime.sh`` is what actually brings the ``.201`` lab lane
    up, and ``deploy_agent.executor._LANE_CONFIGS`` is the tested mapping the
    deploy agent uses. They are already required to agree (the comment in
    ``resolve_compose_file_args`` says so); a release wired into only one of
    them would apply the trio through one path and not the other.
    """
    overlay_filename = Path(DEV_LANE_OVERLAY_RELPATH).name

    deploy_runtime = (REPO_ROOT / "scripts" / "deploy-runtime.sh").read_text(
        encoding="utf-8"
    )
    resolver = re.search(
        r"^resolve_compose_file_args\s*\(\)\s*\{.*?\n\}",
        deploy_runtime,
        re.DOTALL | re.MULTILINE,
    )
    assert resolver is not None, "resolve_compose_file_args() not found"
    assert overlay_filename in resolver.group(0), (
        f"resolve_compose_file_args() does not layer {overlay_filename} for the "
        "dev lane, so `deploy-runtime.sh` brings the lab lane up with no lane "
        "indicator and the registration trio stays fenced there"
    )

    executor = (
        REPO_ROOT / "scripts" / "deploy-agent" / "deploy_agent" / "executor.py"
    ).read_text(encoding="utf-8")
    dev_config = re.search(
        r"EnumRuntimeLane\.DEV:\s*ModelLaneConfig\((?P<body>.*?)\n    \),",
        executor,
        re.DOTALL,
    )
    assert dev_config is not None, "_LANE_CONFIGS[EnumRuntimeLane.DEV] not found"
    assert "_DEV_LANE_OVERLAY" in dev_config.group("body"), (
        "_LANE_CONFIGS[DEV].compose_files does not include _DEV_LANE_OVERLAY"
    )
    assert re.search(
        rf'_DEV_LANE_OVERLAY\s*=\s*f?"[^"]*{re.escape(overlay_filename)}"', executor
    ), f"_DEV_LANE_OVERLAY does not point at {overlay_filename}"


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
    _write_application_ledger_contract(forward)
    return forward


def _write_application_ledger_contract(forward: Path) -> None:
    """Derive a typed ledger contract from a synthetic node migration tree.

    OMN-15413 rejects undeclared application migrations. These fence harnesses
    construct their own SQL, so their manifest must bind those exact bytes
    rather than copy production checksums and fail for an unrelated reason.
    """
    ledger_dir = forward / "_ledger"
    ledger_dir.mkdir()
    bootstrap = REPO_ROOT / "docker/migrations/forward/_ledger/bootstrap.sql"
    (ledger_dir / "bootstrap.sql").write_text(
        bootstrap.read_text(encoding="utf-8"), encoding="utf-8"
    )

    declarations: list[str] = []
    for migration in sorted((forward / "nodes").glob("*/*.sql")):
        node_name = migration.parent.name
        version = f"node:{node_name}:{migration.name}"
        checksum = hashlib.sha256(migration.read_bytes()).hexdigest()
        declarations.append(
            "\t".join(
                (
                    f"nodes/{node_name}/{migration.name}",
                    f"node:{node_name}",
                    f"node:{node_name}",
                    "tenant",
                    version,
                    checksum,
                )
            )
        )
    (ledger_dir / "application-migrations.tsv").write_text(
        "\n".join(declarations) + "\n", encoding="utf-8"
    )
    (ledger_dir / "application-migration-blocks.tsv").write_text("", encoding="utf-8")
    (ledger_dir / "cloud-migration-aliases.tsv").write_text("", encoding="utf-8")


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
    target: PgTarget,
    migrations: Path,
    node_db_name: str,
    lane: str | None = None,
) -> dict[str, str]:
    psql_dir = str(Path(_find_pg_binary("psql") or "psql").parent)
    env = {
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
    # OMN-15379: `lane is None` means the variable is genuinely ABSENT, which is
    # the state every non-dev lane is in. Popping it rather than setting "" also
    # stops an ambient ONEX_MIGRATION_LANE on the test host from leaking in
    # through the `**os.environ` splat and silently un-fencing the default-lane
    # proof — the exact way a fail-closed check turns vacuous.
    if lane is None:
        env.pop(LANE_INDICATOR_ENV, None)
    else:
        env[LANE_INDICATOR_ENV] = lane
    return env


def _run(
    runner: Path,
    target: PgTarget,
    migrations: Path,
    node_db_name: str,
    lane: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/sh", str(runner)],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env=_runner_env(target, migrations, node_db_name, lane),
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
    rows = _psql(
        scoped,
        "SELECT version FROM platform_catalog.schema_migrations "
        "WHERE migration_stream LIKE 'node:%'",
    )
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


# --------------------------------------------------------------------------
# OMN-15379 — lane-scoped fence release. Live half.
#
# These drive the SHIPPED runner against a real Postgres, over the REAL
# vendored registration SQL (not marker stubs), and answer the question from
# the database: does node_service_registry exist, and is FORCE ROW LEVEL
# SECURITY actually on it?
#
# The pair is the discriminator. A runner that ignored the lane indicator
# entirely would fail one of the two legs whichever way it defaulted, so
# neither leg can pass vacuously:
#   * dev lane      -> the trio is APPLIED and relforcerowsecurity is true
#   * default lane  -> the trio is SKIPPED and the table does not exist
# --------------------------------------------------------------------------

REGISTRATION_NODE = "node_projection_registration"
REGISTRY_TABLE = "node_service_registry"
VENDORED_NODES = REPO_ROOT / "docker" / "migrations" / "forward" / "nodes"


@pytest.fixture
def real_registration_tree(tmp_path: Path) -> Path:
    """A migrations tree carrying the REAL vendored registration trio.

    Marker stubs would prove the runner reached the file; they would not prove
    FORCE ROW LEVEL SECURITY landed, which is the whole point of ruling 15. The
    files are copied verbatim from ``docker/migrations/forward/nodes`` so this
    can never drift from what the lab lane actually executes.

    The flat migration creates the ``app_dashboard`` role, reproducing the real
    dependency: node 0002 opens with a DO block that RAISEs unless that role
    exists (it is created by flat migration 094 in the real corpus). Keeping the
    dependency real means this proof also fails if that ordering breaks.
    """
    forward = tmp_path / "migrations" / "forward"
    forward.mkdir(parents=True)
    (forward / "001_create_app_dashboard_role.sql").write_text(
        "DO $$\n"
        "BEGIN\n"
        "  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_dashboard') "
        "THEN\n"
        "    CREATE ROLE app_dashboard NOLOGIN;\n"
        "  END IF;\n"
        "END;\n"
        "$$;\n"
    )

    node_dir = forward / "nodes" / REGISTRATION_NODE
    node_dir.mkdir(parents=True)
    copied = []
    for released in LANE_RELEASED_IDS:
        _, node_name, filename = released.split(":", 2)
        source = VENDORED_NODES / node_name / filename
        assert source.is_file(), (
            f"vendored migration {source} is missing; the released id names "
            "nothing and this proof would be vacuous"
        )
        (node_dir / filename).write_text(source.read_text(encoding="utf-8"))
        copied.append(filename)
    assert len(copied) == 3, copied

    # One fenced DELEGATION id in the same run, as the negative control: ruling
    # 15 released the registration trio and nothing else, so this must still be
    # skipped even on the dev lane.
    delegation_id = FENCED_DELEGATION_IDS[0]
    _, del_node, del_file = delegation_id.split(":", 2)
    del_dir = forward / "nodes" / del_node
    del_dir.mkdir(parents=True, exist_ok=True)
    (del_dir / del_file).write_text(
        f"CREATE TABLE public.{_marker_for(delegation_id)} (id INT);\n"
    )
    _write_application_ledger_contract(forward)
    return forward


def _relrowsecurity(target: PgTarget, dbname: str, table: str) -> tuple[bool, bool]:
    """``(relrowsecurity, relforcerowsecurity)`` straight from ``pg_class``.

    The catalog, not the migration text — "the SQL says FORCE" is not evidence
    that FORCE is in force.
    """
    scoped = PgTarget(
        host=target.host,
        port=target.port,
        user=target.user,
        password=target.password,
        dbname=dbname,
    )
    row = _psql(
        scoped,
        # S608 is suppressed on all three catalog reads in this module: the
        # interpolated name is a module constant (REGISTRY_TABLE), never
        # external input, and SQL has no bind-parameter form for an identifier.
        "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "  # noqa: S608
        f"WHERE oid = to_regclass('public.{table}')",
    )
    enabled, forced = row.split("|")
    return enabled == "t", forced == "t"


@pytest.mark.integration
def test_dev_lane_applies_the_registration_trio_with_force_rls(
    pg_target: PgTarget,
    real_registration_tree: Path,
    node_db: str,
) -> None:
    """ONEX_MIGRATION_LANE=dev: the trio applies and FORCE is LIVE.

    This is the acceptance for operator ruling 15 at the runner boundary — the
    same assertion the ``.201`` lab-lane readback makes, run here against an
    ephemeral cluster so it gates every PR rather than one machine.
    """
    result = _run(
        RUNNER, pg_target, real_registration_tree, node_db, lane=DEV_LANE_VALUE
    )
    assert result.returncode == 0, result.stdout + result.stderr

    for released in LANE_RELEASED_IDS:
        assert (
            "RELEASED on lane 'dev'" in result.stdout and released in result.stdout
        ), f"{released} was not reported as released on the dev lane:\n{result.stdout}"

    assert _table_exists(pg_target, node_db, REGISTRY_TABLE), (
        f"{REGISTRY_TABLE} does not exist — 0000 did not apply, so every "
        "assertion below would be about a table that was never created"
    )
    enabled, forced = _relrowsecurity(pg_target, node_db, REGISTRY_TABLE)
    assert enabled, f"{REGISTRY_TABLE}.relrowsecurity is false — 0002 did not apply"
    assert forced, (
        f"{REGISTRY_TABLE}.relforcerowsecurity is FALSE. Operator ruling 15 "
        "makes the lab lane the FORCE proving ground; without FORCE the table "
        "owner is exempt from the tenant policy and the lane proves nothing"
    )

    ledger = _ledger_ids(pg_target, node_db)
    assert set(LANE_RELEASED_IDS) <= ledger, (
        "the released ids were applied but not RECORDED, so the next run would "
        f"re-apply them: {sorted(set(LANE_RELEASED_IDS) - ledger)}"
    )

    # 0001's heartbeat column and 0002's tenant column both landed — proof the
    # whole trio ran, not just the CREATE.
    scoped = PgTarget(
        host=pg_target.host,
        port=pg_target.port,
        user=pg_target.user,
        password=pg_target.password,
        dbname=node_db,
    )
    columns = set(
        _psql(
            scoped,
            "SELECT column_name FROM information_schema.columns "  # noqa: S608
            f"WHERE table_schema='public' AND table_name='{REGISTRY_TABLE}'",
        ).splitlines()
    )
    assert {"last_heartbeat_at", "uptime_seconds"} <= columns, (
        f"0001 did not apply — heartbeat columns absent: {sorted(columns)}"
    )
    assert "tenant_id" in columns, f"0002 did not apply — no tenant_id: {columns}"
    policies = _psql(
        scoped,
        "SELECT polname FROM pg_policy "  # noqa: S608
        f"WHERE polrelid = to_regclass('public.{REGISTRY_TABLE}')",
    )
    assert "tenant_isolation" in policies, (
        f"0002's tenant_isolation policy is absent: {policies!r}"
    )

    # NEGATIVE CONTROL: ruling 15 is registration-scoped. The delegation id in
    # the same run must still be fenced ON the dev lane.
    delegation_id = FENCED_DELEGATION_IDS[0]
    assert not _table_exists(pg_target, node_db, _marker_for(delegation_id)), (
        f"FENCE BREACH: the dev-lane release leaked to {delegation_id}, which "
        "no operator ruling has un-gated"
    )
    assert delegation_id not in ledger, (
        f"{delegation_id} was recorded on the dev lane; the release set is not "
        "scoped to the registration trio"
    )


@pytest.mark.integration
def test_default_lane_skips_the_trio_and_the_registry_table_is_absent(
    pg_target: PgTarget,
    real_registration_tree: Path,
    node_db: str,
) -> None:
    """No lane indicator: FULL fence. Same tree, same runner, opposite outcome.

    This is the half that makes the pair a proof rather than a demonstration.
    ``lane=None`` unsets the variable entirely, which is the state of
    stability-test, prod, judge, CI, and a fresh-volume
    ``docker compose -f docker-compose.infra.yml up``.
    """
    result = _run(RUNNER, pg_target, real_registration_tree, node_db, lane=None)
    assert result.returncode == 0, result.stdout + result.stderr

    assert "RELEASED on lane" not in result.stdout, (
        f"a lane release fired with no lane indicator set:\n{result.stdout}"
    )
    for released in LANE_RELEASED_IDS:
        assert "SKIP (operator-gated" in result.stdout and released in result.stdout, (
            f"{released} was not reported as operator-gated:\n{result.stdout}"
        )

    assert not _table_exists(pg_target, node_db, REGISTRY_TABLE), (
        f"FENCE BREACH: {REGISTRY_TABLE} EXISTS on a lane with no indicator — "
        "0000 applied unfenced"
    )
    ledger = _ledger_ids(pg_target, node_db)
    assert not (ledger & set(LANE_RELEASED_IDS)), (
        "a fenced id was recorded on the default lane, which would make the "
        f"eventual un-fence a silent no-op: {sorted(ledger & set(LANE_RELEASED_IDS))}"
    )


@pytest.mark.integration
def test_unknown_lane_value_fails_closed_to_the_full_fence(
    pg_target: PgTarget,
    real_registration_tree: Path,
    node_db: str,
) -> None:
    """An unrecognised lane is fenced, loudly — not silently treated as dev.

    ``stability-test`` is used as the value deliberately: it is a real lane
    name, so a runner that pattern-matched loosely (prefix, glob, "any non-empty
    value means release") would betray itself here rather than on an obviously
    bogus string.
    """
    result = _run(
        RUNNER, pg_target, real_registration_tree, node_db, lane="stability-test"
    )
    assert result.returncode == 0, result.stdout + result.stderr

    assert not _table_exists(pg_target, node_db, REGISTRY_TABLE), (
        f"FENCE BREACH: an UNKNOWN lane value released the trio — "
        f"{REGISTRY_TABLE} exists"
    )
    assert "RELEASED on lane" not in result.stdout, result.stdout
    assert "unknown ONEX_MIGRATION_LANE" in result.stderr, (
        "failing closed silently is still a silent failure — the runner must "
        f"say it did not recognise the lane:\n{result.stderr}"
    )
