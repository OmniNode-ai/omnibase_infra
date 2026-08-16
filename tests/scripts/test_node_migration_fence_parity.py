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

DRIFT HAZARD / follow-up — RESOLVED for the baseline by OMN-15349
------------------------------------------------------------------
Originally there was no single source of truth for this list: it was
duplicated in two repos and only prose and these tests kept them equal.
OMN-15349 (Option 1, operator ruling R-f 2026-08-05) removed that duplication
for the BASELINE fence: ``docker/migrations/forward/fenced-node-migrations.yaml``
is now the sole committed list, and ``scripts/run-forward-migrations.sh``
parses it at runtime instead of carrying its own literal copy (see the
"single-sourced" comment in the OMN-15336 fence block there). The k8s Job
(``omninode_infra``) parses the same manifest out of the same
``omnibase-infra-migrate`` image it already pulls.

What single-sourcing the baseline does NOT do: make the two runners' EFFECTIVE
fences equal. Each runner layers its own, independently operator-ruled,
lane-scoped RELEASE on top of the shared baseline (see the OMN-15379 section
below and ``manifest_ids()``/``K8S_RULING_21_RELEASE`` here) — a release is an
environment-specific operator decision, not fence data, so it deliberately
stays out of the manifest. ``test_fence_matches_omninode_infra_k8s_runner``
below therefore asserts baseline-superset + known-release-subtraction, not
raw equality.

* Always on, gating every PR: ``test_manifest_pins_the_known_baseline_fence``
  pins the manifest's content, exact and IN ORDER — the same change-control
  friction the old pinned-tuple test gave the shell script, now pointed at
  the actual single source. ``test_manifest_shell_parse_matches_yaml_parse``
  guards the OTHER hazard single-sourcing introduces: both runners parse this
  YAML with a plain ``sed`` one-liner (no YAML library in ``/bin/sh`` or the
  k8s Job's minimal bash), so a manifest reformatted in a way ``yaml.safe_load``
  still accepts but the sed grammar cannot (e.g. single-quoted ids, an
  unindented list) would silently ship a truncated or empty fence in
  production while every YAML-aware tool kept reading it fine.
* Opt-in: ``test_fence_matches_omninode_infra_k8s_runner`` diffs the manifest
  baseline against the live k8s manifest's effective (post-ruling-21) list. It
  is opt-in on purpose — it depends on an ``omninode_infra`` checkout whose
  freshness this repo cannot guarantee, and an always-on version went RED on
  the ``.200`` build host purely because that clone sat two commits behind
  dev. False REDs from another repo's local staleness are worse than the gap
  they close.

Residual gap, unchanged by OMN-15349: a NEW baseline id added to the k8s side
alone (as opposed to a release-policy change) would still land unnoticed here
until someone runs the opt-in check — single-sourcing the data removes the
"two literals to keep in sync" hazard, it does not make one repo's CI aware of
the other repo's uncommitted intentions.

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
import shutil
import subprocess
import tempfile
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
_free_port = _advisory_lock._free_port
_unavailable = _advisory_lock._unavailable
pg_target = _advisory_lock.pg_target

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]

FENCE_BEGIN = "# ---- BEGIN operator fence — node migration ids (OMN-15336) ----"
FENCE_END = "# ---- END operator fence — node migration ids (OMN-15336) ----"
SKIP_BEGIN = "# ---- BEGIN fenced-id skip (OMN-15336) ----"
SKIP_END = "# ---- END fenced-id skip (OMN-15336) ----"
# OMN-15336 item 4: the unclassified-FORCE-RLS guard. Two blocks — the
# predicate's own definition (sits beside is_fenced_node_migration) and the
# call site (sits in the node loop, after the already-applied probe).
FORCE_RLS_GUARD_DEF_BEGIN = (
    "# ---- BEGIN unclassified FORCE ROW LEVEL SECURITY guard (OMN-15336 item 4) ----"
)
FORCE_RLS_GUARD_DEF_END = (
    "# ---- END unclassified FORCE ROW LEVEL SECURITY guard (OMN-15336 item 4) ----"
)
FORCE_RLS_GUARD_CALL_BEGIN = (
    "# ---- BEGIN unclassified FORCE ROW LEVEL SECURITY guard call "
    "(OMN-15336 item 4) ----"
)
FORCE_RLS_GUARD_CALL_END = (
    "# ---- END unclassified FORCE ROW LEVEL SECURITY guard call "
    "(OMN-15336 item 4) ----"
)
# OMN-15336 item 4 repair (D1, 2026-08-05): the grandfather-snapshot block —
# the fix for the guard's over-fire against the established, pre-guard tree.
GRANDFATHER_BLOCK_BEGIN = (
    "# ---- BEGIN FORCE ROW LEVEL SECURITY grandfather snapshot "
    "(OMN-15336 item 4 repair) ----"
)
GRANDFATHER_BLOCK_END = (
    "# ---- END FORCE ROW LEVEL SECURITY grandfather snapshot "
    "(OMN-15336 item 4 repair) ----"
)

# --- OMN-15349 single-sourced manifest ---------------------------------------
MANIFEST_RELPATH = "docker/migrations/forward/fenced-node-migrations.yaml"
MANIFEST_PATH = REPO_ROOT / MANIFEST_RELPATH

# The regex both runners actually execute against the manifest at runtime
# (POSIX `sed`; neither /bin/sh nor the k8s Job's bash has a YAML library).
# This is the REAL parse path in production.
_MANIFEST_ID_LINE = re.compile(r'^\s*-\s*id:\s*"([^"]*)"', re.MULTILINE)


def parse_shell_manifest_ids(text: str) -> tuple[str, ...]:
    """Reproduce the shell `sed` extraction of `- id: "..."` lines verbatim."""
    return tuple(_MANIFEST_ID_LINE.findall(text))


def _load_manifest() -> list[dict[str, str]]:
    """Independent oracle: a real YAML parse of the manifest, schema-checked."""
    doc = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(doc, dict) and "fenced_node_migrations" in doc, (
        f"{MANIFEST_RELPATH} must be a mapping with a top-level "
        "'fenced_node_migrations' key"
    )
    entries = doc["fenced_node_migrations"]
    assert isinstance(entries, list) and entries, (
        f"{MANIFEST_RELPATH}'s fenced_node_migrations must be a non-empty list"
    )
    for i, entry in enumerate(entries):
        assert isinstance(entry, dict) and "id" in entry, (
            f"{MANIFEST_RELPATH} entry {i} is not a mapping with an 'id' key: {entry!r}"
        )
    return entries


def manifest_ids() -> tuple[str, ...]:
    """Ordered ids from the manifest, via the YAML-parse oracle."""
    return tuple(entry["id"] for entry in _load_manifest())


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
# The OMN-15717/OMN-15376 node_pr_review_bot id. Fenced (not SQL-edited) to
# satisfy the OMN-15376 shape-reconciliation gate without changing the file's
# content sha256, which is bound to an already-applied production row (see
# fenced-node-migrations.yaml's own rationale comment for the full argument).
# Not releasable on any lane today — no ONEX_MIGRATION_LANE value un-gates it.
FENCED_PR_REVIEW_BOT_IDS = (
    "node:node_pr_review_bot:001_create_review_bot_bypass_log.sql",
)
# OMN-15336 item 4 / OMN-15656: contract-declared TENANT domain (unlike
# node_service_registry, this is not a misclassification), held for the same
# OMN-15301 writer-tenant-context reason as the delegation quartet. Was never
# in this manifest on any runner before this entry — see the manifest's own
# docstring for the incident.
FENCED_INFERENCE_RESPONSE_IDS = (
    "node:node_projection_delegation_inference_response:"
    "0003_inference_response_text_rls_tenant_isolation.sql",
)
# Pinned expectation for the manifest content (OMN-15349): the baseline fence,
# exact and in order. A manifest edit that moves this must update the pin in
# the same PR — same change-control friction the pre-OMN-15349 shell-literal
# pin gave, now pointed at the actual single source instead of a copy of it.
EXPECTED_FENCE = (
    FENCED_DELEGATION_IDS
    + FENCED_REGISTRATION_IDS
    + FENCED_PR_REVIEW_BOT_IDS
    + FENCED_INFERENCE_RESPONSE_IDS
)

# --- OMN-15336 item 4 repair (D1, 2026-08-05): FORCE-RLS grandfather snapshot
# --------------------------------------------------------------------------
# See docker/migrations/forward/grandfathered-force-rls-migrations.yaml's own
# header for what this is and why it is a snapshot, not an allowlist. Pinned
# here with the SAME change-control friction as EXPECTED_FENCE: growing this
# tuple without a corresponding manifest edit (or vice versa) fails
# test_grandfather_pins_the_snapshot_baseline closed.
GRANDFATHER_MANIFEST_RELPATH = (
    "docker/migrations/forward/grandfathered-force-rls-migrations.yaml"
)
GRANDFATHER_MANIFEST_PATH = REPO_ROOT / GRANDFATHER_MANIFEST_RELPATH

EXPECTED_GRANDFATHER = (
    "node:node_canary_score_reducer:0002_capability_scores_tenant_id_and_rls.sql",
    "node:node_projection_context_roi:003_context_roi_scores_tenant_id_and_rls.sql",
    "node:node_projection_cost_summary:0002_llm_cost_aggregates_tenant_id_and_rls.sql",
    "node:node_projection_dep_health:002_dep_health_findings_tenant_id_and_rls.sql",
    "node:node_projection_instruction_eval:"
    "0002_instruction_eval_aggregate_snapshots_tenant_id_and_rls.sql",
    "node:node_projection_pattern_learning:"
    "0001_pattern_learning_artifacts_tenant_id_and_rls.sql",
    "node:node_projection_routing_decision:"
    "0022_agent_routing_decisions_tenant_id_and_rls.sql",
    "node:node_projection_savings:081_savings_estimates_rls_tenant_isolation.sql",
    "node:node_projection_skill_executions:"
    "0002_skill_execution_snapshots_tenant_id_and_rls.sql",
)


def _load_grandfather_manifest() -> list[dict[str, str]]:
    """Independent oracle: a real YAML parse of the grandfather manifest."""
    doc = yaml.safe_load(GRANDFATHER_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(doc, dict) and "grandfathered_force_rls_migrations" in doc, (
        f"{GRANDFATHER_MANIFEST_RELPATH} must be a mapping with a top-level "
        "'grandfathered_force_rls_migrations' key"
    )
    entries = doc["grandfathered_force_rls_migrations"]
    assert isinstance(entries, list) and entries, (
        f"{GRANDFATHER_MANIFEST_RELPATH}'s grandfathered_force_rls_migrations "
        "must be a non-empty list"
    )
    for i, entry in enumerate(entries):
        assert isinstance(entry, dict) and "id" in entry, (
            f"{GRANDFATHER_MANIFEST_RELPATH} entry {i} is not a mapping with "
            f"an 'id' key: {entry!r}"
        )
    return entries


def grandfather_manifest_ids() -> tuple[str, ...]:
    """Ordered ids from the grandfather manifest, via the YAML-parse oracle."""
    return tuple(entry["id"] for entry in _load_grandfather_manifest())


# --- OMN-15349 k8s-side release (operator ruling 21, OMN-15332 comment
# 1a067542, 2026-07-31T14:05Z GO) --------------------------------------------
# Unlike the lab-lane release below (env-gated, ruling 15), ruling 21
# authorized a DURABLE release of the registration trio on the k8s Job, which
# serves exactly one environment (staging/onex-dev). This is a k8s-side
# runner policy, not manifest data (see the manifest file's own docstring for
# why) — pinned here only so the opt-in cross-repo test
# (`test_fence_matches_omninode_infra_k8s_runner`) can state the expected
# baseline-minus-release relationship instead of asserting raw equality.
K8S_RULING_21_RELEASE = (
    "node:node_projection_registration:0000_create_node_service_registry.sql",
    "node:node_projection_registration:0001_add_heartbeat_columns.sql",
    "node:node_projection_registration:0002_node_service_registry_tenant_rls.sql",
)

# --- OMN-15379 lane-scoped release (operator ruling 15, 2026-07-29) ----------
# Ruling 15: node_service_registry FORCE ROW LEVEL SECURITY extends to the LAB
# LANE ONLY. The lab (compose dev lane, project `omnibase-infra`) applies the
# registration trio in full as the proving ground. The omninode_infra k8s
# fence is a SEPARATE release (ruling 21, K8S_RULING_21_RELEASE above) — not
# "unchanged," and not the same mechanism (durable vs env-gated). See the
# CORRECTION comment in run-forward-migrations.sh's LANE-SCOPED FENCE RELEASE
# block for the same fix applied at the runner.
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


def extract_force_rls_guard_def(text: str | None = None) -> str:
    """Return the unclassified-FORCE-RLS predicate's own definition block."""
    return _extract_marked(
        text if text is not None else _runner_text(),
        FORCE_RLS_GUARD_DEF_BEGIN,
        FORCE_RLS_GUARD_DEF_END,
    )


def extract_force_rls_guard_call(text: str | None = None) -> str:
    """Return the in-loop call site of the unclassified-FORCE-RLS guard."""
    return _extract_marked(
        text if text is not None else _runner_text(),
        FORCE_RLS_GUARD_CALL_BEGIN,
        FORCE_RLS_GUARD_CALL_END,
    )


def extract_grandfather_block(text: str | None = None) -> str:
    """Return the FORCE-RLS grandfather-snapshot block, markers stripped."""
    return _extract_marked(
        text if text is not None else _runner_text(),
        GRANDFATHER_BLOCK_BEGIN,
        GRANDFATHER_BLOCK_END,
    )


def strip_force_rls_guard(text: str) -> str:
    """The pre-OMN-15336-item-4 runner: byte-identical minus the guard.

    Derived from the shipped artifact, same discipline as ``strip_fence``
    above, so the RED-control-for-the-RED-control can never drift from the
    thing it is the control for.
    """
    out = text
    for begin, end in (
        (FORCE_RLS_GUARD_DEF_BEGIN, FORCE_RLS_GUARD_DEF_END),
        (FORCE_RLS_GUARD_CALL_BEGIN, FORCE_RLS_GUARD_CALL_END),
    ):
        lines = out.splitlines(keepends=True)
        start = next(i for i, ln in enumerate(lines) if ln.strip() == begin)
        stop = next(i for i, ln in enumerate(lines) if ln.strip() == end)
        out = "".join(lines[:start] + lines[stop + 1 :])
    return out


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


def test_manifest_pins_the_known_baseline_fence() -> None:
    """Exact-set and ORDER, not containment — now pinned against the manifest.

    OMN-15349: the shell script no longer carries its own literal copy of the
    fence, so there is nothing left to pin THERE. This is the same
    change-control friction as the pre-OMN-15349 test, pointed at the actual
    single source: a manifest edit that silently drops one id while adding
    another is precisely the regression this assertion exists to catch.
    """
    found = manifest_ids()
    assert found == EXPECTED_FENCE, (
        f"{MANIFEST_RELPATH} changed. Expected exactly:\n  "
        + "\n  ".join(EXPECTED_FENCE)
        + "\nFound:\n  "
        + "\n  ".join(found)
    )
    assert found[: len(FENCED_DELEGATION_IDS)] == FENCED_DELEGATION_IDS, (
        "the OMN-14974 delegation fence was disturbed"
    )
    registration_end = len(FENCED_DELEGATION_IDS) + len(FENCED_REGISTRATION_IDS)
    assert (
        found[len(FENCED_DELEGATION_IDS) : registration_end] == FENCED_REGISTRATION_IDS
    ), "the OMN-15335/OMN-15343 registration hold is not the exact expected trio"
    pr_review_bot_end = registration_end + len(FENCED_PR_REVIEW_BOT_IDS)
    assert found[registration_end:pr_review_bot_end] == FENCED_PR_REVIEW_BOT_IDS, (
        "the OMN-15717/OMN-15376 node_pr_review_bot hold is not the exact expected id"
    )
    assert found[pr_review_bot_end:] == FENCED_INFERENCE_RESPONSE_IDS, (
        "the OMN-15336 item-4 inference-response hold is not the expected id"
    )


def test_manifest_shell_parse_matches_yaml_parse() -> None:
    """The REAL production parse path (`sed`) must agree with `yaml.safe_load`.

    Neither runner has a YAML library available. A manifest edit that a real
    YAML parser still accepts but the shell one-liner cannot (single-quoted
    ids, an unindented list, a missing closing quote) would ship a truncated
    or empty fence in production while looking correct to every YAML-aware
    tool, including this test file's own `manifest_ids()` oracle if it were
    the only check.
    """
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    shell_parsed = parse_shell_manifest_ids(text)
    yaml_parsed = manifest_ids()
    assert shell_parsed == yaml_parsed, (
        "the sed one-liner both runners execute against the manifest "
        "disagrees with a real YAML parse — the manifest format is not "
        "parseable the way production actually parses it.\n"
        f"  sed:  {shell_parsed}\n  yaml: {yaml_parsed}"
    )


def test_manifest_ids_are_well_formed_and_unique() -> None:
    """Schema/shape validator (OMN-15349): grammar + no duplicates.

    A typo'd id or an accidental duplicate fences nothing extra (or hides a
    duplicate entry silently) without this check — `grep -Fxq` in the runner
    would just never match, or match the same line twice.
    """
    ids = manifest_ids()
    assert len(ids) == len(set(ids)), (
        f"{MANIFEST_RELPATH} has duplicate ids: {[i for i in ids if ids.count(i) > 1]}"
    )
    for entry_id in ids:
        parts = entry_id.split(":")
        assert len(parts) == 3 and parts[0] == "node" and all(parts), (
            f"{entry_id!r} does not match the node:<node>:<filename> grammar"
        )
        assert parts[2].endswith(".sql"), f"{entry_id!r} does not name a .sql file"


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


def test_unclassified_force_rls_guard_is_defined() -> None:
    """OMN-15336 item 4: the predicate and its call site both exist."""
    definition = extract_force_rls_guard_def()
    assert "migration_declares_unclassified_force_rls()" in definition, (
        "the guard predicate must be defined"
    )
    call = extract_force_rls_guard_call()
    assert "migration_declares_unclassified_force_rls" in call, (
        "the node loop must call the guard predicate"
    )
    assert "exit 1" in call, "the guard must FATAL, not warn, on a match"


def test_unclassified_force_rls_guard_excludes_no_force_statements() -> None:
    """`NO FORCE ROW LEVEL SECURITY` (a disabling statement) must never trip
    the guard — otherwise a future FORCE-strip migration could never ship.
    """
    definition = extract_force_rls_guard_def()
    assert "NO[[:space:]]+FORCE" in definition, (
        "the predicate must explicitly exclude the NO FORCE (disabling) form"
    )


def test_unclassified_force_rls_guard_is_checked_only_for_unfenced_ids() -> None:
    """The guard must not re-litigate an id someone already classified.

    Both the fenced-and-held and the fenced-and-released cases must bypass
    it: the manifest entry itself is the classification the guard exists to
    require, whether or not this lane also carries a release for it.
    """
    call = extract_force_rls_guard_call()
    assert re.search(r"!\s*is_fenced_node_migration\s+\"\$\{migration_id\}\"", call), (
        "the guard must be gated on `! is_fenced_node_migration ...`"
    )


def test_unclassified_force_rls_guard_runs_after_the_already_applied_probe() -> None:
    """Ordering is load-bearing the OTHER way from the fence-skip check above:
    this guard must run AFTER ``migration_is_applied`` returns false, never
    before — a guard that ran earlier would FATAL on every subsequent run of
    a lane where an unclassified id already applied before this guard
    existed (e.g. the real .201 dev lane's 0003/081), bricking that lane's
    every future deploy over already-committed history it cannot undo.
    """
    text = _runner_text()
    node_loop = text[text.index("Auto-discover and apply node-owned migrations") :]
    probe = node_loop.index("if migration_is_applied")
    guard_call = node_loop.index('if ! is_fenced_node_migration "${migration_id}" \\')
    apply_sql = node_loop.index('-v ON_ERROR_STOP=1 -f "$migration_file"')
    assert probe < guard_call < apply_sql, (
        "the unclassified-FORCE-RLS guard must run strictly between the "
        "already-applied probe and the apply "
        f"(node-loop offsets: probe={probe} guard={guard_call} apply={apply_sql})"
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
    fence = set(manifest_ids())
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
    """THE cross-repo seam assertion.

    Pre-OMN-15349 shape: assert raw equality of two independently-maintained
    literal lists. Post-OMN-15349 shape: the two runners no longer maintain
    independent lists at all — the k8s Job parses the SAME manifest baseline
    this repo owns (``manifest_ids()``) and layers its own, operator-ruled
    release (ruling 21, ``K8S_RULING_21_RELEASE``) on top. So this test
    asserts three things instead of raw equality:

      1. the k8s Job still references the shared manifest by name (it did not
         quietly revert to a hardcoded baseline copy),
      2. the k8s Job's committed release policy still matches the expected
         ruling-21 registration trio, exactly (drift here IS a real defect —
         unlike the baseline, there is only one copy of this release list),
      3. the k8s Job's resulting EFFECTIVE fence (baseline minus release)
         still equals the delegation quartet — the only ids ruling 21 left
         fenced there.

    OPT-IN via ``REQUIRE_CROSS_REPO_FENCE_PARITY`` — deliberately, and this is
    the honest shape rather than the convenient one. The comparison depends on
    an artifact outside this repo whose freshness this repo cannot guarantee:
    run against a stale ``omninode_infra`` clone it reports drift that does not
    exist (measured — the ``.200`` build host's clone sat 2 commits behind dev
    while this change was being gated, and an always-on version of this test
    went RED there for exactly that reason). Gating every PR on another repo's
    local checkout freshness manufactures false REDs, so the always-on leg is
    ``test_manifest_pins_the_known_baseline_fence`` above; this is the sharper
    check you run deliberately, after fetching, on a host that has both repos.
    """
    if not os.environ.get("REQUIRE_CROSS_REPO_FENCE_PARITY"):
        pytest.skip(
            "cross-repo fence diff is opt-in: fetch omninode_infra, then run "
            "with REQUIRE_CROSS_REPO_FENCE_PARITY=1 (optionally "
            "OMNINODE_INFRA_ROOT=<path>). The pinned "
            "test_manifest_pins_the_known_baseline_fence assertion gates "
            "every PR regardless."
        )

    root = _omninode_infra_root()
    if root is None:
        pytest.fail(
            "REQUIRE_CROSS_REPO_FENCE_PARITY is set but no omninode_infra "
            "checkout was found (tried OMNINODE_INFRA_ROOT, "
            "$OMNI_HOME/omninode_infra, and sibling paths)"
        )

    text, provenance = _k8s_manifest_source(root)
    manifest_filename = MANIFEST_RELPATH.rsplit("/", 1)[-1]
    assert manifest_filename in text, (
        f"{provenance}: the k8s Job no longer references the shared manifest "
        f"{manifest_filename!r} — it may have reverted to a hardcoded fence "
        "baseline, reopening the exact duplication OMN-15349 removed"
    )

    release_match = re.search(
        r"K8S_RULING_21_RELEASED_NODE_MIGRATION_IDS=\((?P<body>.*?)\)",
        text,
        re.DOTALL,
    )
    assert release_match is not None, (
        f"{provenance}: K8S_RULING_21_RELEASED_NODE_MIGRATION_IDS array not "
        "found — the k8s Job's ruling-21 release policy was removed or "
        "reshaped"
    )
    k8s_release = tuple(re.findall(r'"([^"]+)"', release_match.group("body")))
    assert k8s_release == K8S_RULING_21_RELEASE, (
        "CROSS-REPO RELEASE-POLICY DRIFT: the k8s Job's ruling-21 release no "
        "longer matches the expected registration trio.\n"
        f"  k8s ({provenance}):\n    "
        + "\n    ".join(k8s_release)
        + "\n  expected:\n    "
        + "\n    ".join(K8S_RULING_21_RELEASE)
    )

    baseline = manifest_ids()
    stray = sorted(set(k8s_release) - set(baseline))
    assert not stray, (
        f"{provenance}: the k8s release names ids the shared manifest "
        f"baseline does not cover: {stray}"
    )
    effective_k8s_fence = tuple(i for i in baseline if i not in k8s_release)
    expected_effective_k8s_fence = (
        FENCED_DELEGATION_IDS + FENCED_PR_REVIEW_BOT_IDS + FENCED_INFERENCE_RESPONSE_IDS
    )
    assert effective_k8s_fence == expected_effective_k8s_fence, (
        "the k8s Job's effective (post-release) fence no longer equals the "
        "delegation quartet plus the OMN-15717 node_pr_review_bot hold plus "
        "the OMN-15336 item-4 inference-response hold — either the shared "
        "manifest baseline or the k8s release changed: "
        f"{effective_k8s_fence}"
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
    _write_fence_manifest(forward, EXPECTED_FENCE)
    _write_grandfather_manifest(forward)
    _write_application_ledger_contract(forward)
    return forward


def _write_fence_manifest(forward: Path, ids: tuple[str, ...]) -> None:
    """Write a synthetic fence manifest into a test harness's migrations tree.

    OMN-15349: the shipped runner unconditionally requires
    ``${MIGRATIONS_DIR}/fenced-node-migrations.yaml`` now — a harness that
    built its own ``forward/`` tree without one would only prove the runner's
    FATAL-if-missing guard, not the fence itself. Written in the exact format
    the shipped ``sed`` one-liner parses (mirrors the committed manifest under
    ``docker/migrations/forward/fenced-node-migrations.yaml``).
    """
    lines = ["fenced_node_migrations:"]
    for migration_id in ids:
        lines.append(f'  - id: "{migration_id}"')
    (forward / "fenced-node-migrations.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_grandfather_manifest(forward: Path, ids: tuple[str, ...] = ()) -> None:
    """Write a synthetic FORCE-RLS grandfather snapshot into a test harness's
    migrations tree (OMN-15336 item 4 repair).

    The shipped runner unconditionally requires
    ``${MIGRATIONS_DIR}/grandfathered-force-rls-migrations.yaml`` now, same
    discipline as ``_write_fence_manifest`` above. Defaults to an empty list:
    none of the synthetic fixtures in this module vendor real FORCE-RLS SQL
    bodies (their marker files are plain ``CREATE TABLE ... (id INT);``), so
    an empty grandfather snapshot is correct unless a fixture explicitly
    passes ids.
    """
    lines = ["grandfathered_force_rls_migrations:"]
    for migration_id in ids:
        lines.append(f'  - id: "{migration_id}"')
    (forward / "grandfathered-force-rls-migrations.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


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
    # OMN-15717 (#2678) added LEGACY_NODE_MIGRATION_DECLARATIONS as a fourth
    # unconditionally-required manifest file in validate_application_migration_
    # manifest() -- empty is valid (mirrors the two files above: the awk
    # per-record validators never fire on zero input lines, so an empty file
    # passes every format/duplicate/overlap check untouched).
    (ledger_dir / "legacy-node-migrations.tsv").write_text("", encoding="utf-8")


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
    _write_fence_manifest(forward, EXPECTED_FENCE)
    _write_grandfather_manifest(forward)
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


# --------------------------------------------------------------------------
# OMN-15336 item 4 — unclassified FORCE ROW LEVEL SECURITY guard. Live half.
#
# The required-fix item this closes: "Reconsider whether 0003/081/0002
# belong in the fence list — they carry the same hazard and are currently
# ungated on every runner." 0002 was added by OMN-15379/OMN-15349; this
# guard is the durable mechanism so the NEXT one (there is no reason to
# believe 0003/081 are the last) is refused instead of silently applying —
# closing the gap the proof stage found: nothing in either runner, and
# nothing wired into omnibase_infra CI, inspected a migration's SQL text to
# block an unfenced FORCE ROW LEVEL SECURITY statement.
# --------------------------------------------------------------------------

UNCLASSIFIED_FORCE_RLS_CONTROL_ID = (
    "node:node_projection_delegation:0098_unclassified_force_rls_control.sql"
)


@pytest.fixture
def unclassified_force_rls_tree(tmp_path: Path) -> Path:
    """``node_tree`` plus one migration that enables FORCE ROW LEVEL SECURITY
    and is deliberately ABSENT from the fence manifest — the exact OMN-15336
    item-4 scenario that produced the real, ungated 0003 and 081 incidents:
    a live hazard with no classification at all (as opposed to a classified
    id someone has reviewed and either held or released).
    """
    forward = tmp_path / "migrations" / "forward"
    forward.mkdir(parents=True)
    (forward / "001_noop.sql").write_text("SELECT 1;\n")

    for migration_id in (*EXPECTED_FENCE, UNFENCED_CONTROL_ID):
        _, node_name, filename = migration_id.split(":", 2)
        node_dir = forward / "nodes" / node_name
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / filename).write_text(
            f"CREATE TABLE public.{_marker_for(migration_id)} (id INT);\n"
        )

    _, control_node, control_filename = UNCLASSIFIED_FORCE_RLS_CONTROL_ID.split(":", 2)
    control_dir = forward / "nodes" / control_node
    control_dir.mkdir(parents=True, exist_ok=True)
    marker = _marker_for(UNCLASSIFIED_FORCE_RLS_CONTROL_ID)
    control_dir.joinpath(control_filename).write_text(
        "-- a real DDL shape, matching the actual 0003/081 migrations: a\n"
        "-- prose comment mentioning FORCE ROW LEVEL SECURITY must NOT alone\n"
        "-- trip the guard (comment-blind matching), only the DDL below does.\n"
        f"CREATE TABLE public.{marker} (id INT, tenant_id TEXT);\n"
        f"ALTER TABLE public.{marker} ENABLE ROW LEVEL SECURITY;\n"
        f"ALTER TABLE public.{marker} FORCE ROW LEVEL SECURITY;\n"
    )

    _write_fence_manifest(forward, EXPECTED_FENCE)
    _write_grandfather_manifest(forward)
    _write_application_ledger_contract(forward)
    return forward


@pytest.mark.integration
def test_unclassified_force_rls_migration_is_refused(
    pg_target: PgTarget,
    unclassified_force_rls_tree: Path,
    node_db: str,
) -> None:
    """RED control: a node migration enabling FORCE ROW LEVEL SECURITY with
    no fence entry at all must be REFUSED, not applied.

    Before this guard, this is exactly what happened to
    node_projection_delegation_inference_response/0003 and
    node_projection_savings/081 — neither was ever in the fence manifest on
    any runner, and both applied unattended on the .201 dev lane (see the
    module docstring's incident description and OMN-15336's required-fix
    item 4).
    """
    result = _run(RUNNER, pg_target, unclassified_force_rls_tree, node_db)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        "the runner must refuse an unclassified FORCE ROW LEVEL SECURITY "
        f"migration, not apply it silently:\n{combined}"
    )
    assert UNCLASSIFIED_FORCE_RLS_CONTROL_ID in combined, (
        f"the FATAL must name the offending migration id:\n{combined}"
    )
    assert "FATAL" in combined and "FORCE ROW LEVEL SECURITY" in combined, (
        f"expected a FATAL naming the FORCE ROW LEVEL SECURITY hazard:\n{combined}"
    )
    assert not _table_exists(
        pg_target, node_db, _marker_for(UNCLASSIFIED_FORCE_RLS_CONTROL_ID)
    ), "FENCE BREACH: the unclassified FORCE RLS migration was APPLIED"
    ledger = _ledger_ids(pg_target, node_db)
    assert UNCLASSIFIED_FORCE_RLS_CONTROL_ID not in ledger, (
        "a refused migration must not be recorded as applied — that would "
        "make later classification a silent no-op"
    )


@pytest.mark.integration
def test_guard_free_runner_applies_the_unclassified_migration(
    pg_target: PgTarget,
    unclassified_force_rls_tree: Path,
    node_db: str,
    tmp_path: Path,
) -> None:
    """RED control FOR the RED control: without the guard, the exact same
    scenario reproduces the OMN-15336 item-4 incident — silent apply.

    Without this, ``test_unclassified_force_rls_migration_is_refused`` could
    be passing for an unrelated reason (a checksum mismatch, a missing
    manifest declaration) and still look like proof the guard works.
    """
    legacy = tmp_path / "run-forward-migrations.preguard.sh"
    legacy.write_text(strip_force_rls_guard(_runner_text()))

    result = _run(legacy, pg_target, unclassified_force_rls_tree, node_db)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _table_exists(
        pg_target, node_db, _marker_for(UNCLASSIFIED_FORCE_RLS_CONTROL_ID)
    ), (
        "the guard-free runner was expected to apply the unclassified FORCE "
        f"RLS migration, reproducing the item-4 incident:\n{result.stdout}"
    )
    ledger = _ledger_ids(pg_target, node_db)
    assert UNCLASSIFIED_FORCE_RLS_CONTROL_ID in ledger, (
        f"the guard-free runner did not record the migration either: {sorted(ledger)}"
    )


# ==============================================================================
# OMN-15336 item 4 REPAIR (D1, empirically reproduced 2026-08-05)
# ==============================================================================
# Defect: the guard above fires for ANY FORCE-enabling node migration absent
# from the operator fence, with no notion of "already part of the tree." The
# vendored tree carries 13 FORCE-enabling node migrations; the fence
# classifies only 4. The other 9 were ordinary, already-shipped migrations
# that had been applying on every warm lane since before the guard existed —
# but the guard could not distinguish them from a brand-new, unreviewed one.
# Reproduced live: shipped runner against a virgin PG16 -> exit 1, FATAL at
# node:node_canary_score_reducer:0002 (the first of the 9 in sort order), 1
# node migration applied, 87 withheld. A cold lane bring-up (CI, a fresh
# compose volume) could never converge.
#
# Fix: docker/migrations/forward/grandfathered-force-rls-migrations.yaml, a
# frozen snapshot (not a rolling allowlist) of exactly those 9 ids, consulted
# by the guard as a SECOND, independent bypass alongside the fence. See that
# file's own header and the "FORCE ROW LEVEL SECURITY grandfather snapshot"
# block in scripts/run-forward-migrations.sh for the full rationale.
#
# GUARD_INTRODUCTION_COMMIT is the commit that first shipped the unclassified-
# FORCE-RLS guard; every grandfathered id must have existed in the tree at its
# PARENT (i.e. immediately before the guard could ever have fired for it).
#
# OMN-15831: this constant has now rotted to an unreachable commit TWICE
# (bbac5205 -> 7a957a0a -> 90cd78a5) because each prior pin named a #2666
# BRANCH commit that squash-merge + branch deletion later orphaned (no ref
# contains it, `merge-base --is-ancestor` fails). The value below is the
# #2666 SQUASH MERGE commit itself, which is permanent history on `dev` and
# cannot be orphaned by branch cleanup the way a branch-tip commit can.
# `test_guard_introduction_commit_is_reachable` (below) makes this a fail-
# closed, self-diagnosing assertion instead of a silent future recurrence.
GUARD_INTRODUCTION_COMMIT = "3bc7fcaf2e0858b04dda5f3fd3e695a7df88b754"


def _sql_declares_unclassified_force_rls(sql_text: str) -> bool:
    """Python mirror of migration_declares_unclassified_force_rls()'s pipeline.

    Kept as an independent re-implementation (not a subprocess call into the
    shell function) so a bug shared between the shell predicate and this
    oracle cannot hide a mis-scoped grandfather entry from both.
    """
    stripped = re.sub(r"--.*$", "", sql_text, flags=re.MULTILINE)
    force_lines = [
        line
        for line in stripped.splitlines()
        if re.search(r"FORCE[ \t]+ROW[ \t]+LEVEL[ \t]+SECURITY", line, re.IGNORECASE)
    ]
    qualifying = [
        line
        for line in force_lines
        if not re.search(
            r"NO[ \t]+FORCE[ \t]+ROW[ \t]+LEVEL[ \t]+SECURITY", line, re.IGNORECASE
        )
    ]
    return len(qualifying) > 0


def test_grandfather_manifest_pins_the_snapshot_baseline() -> None:
    """The ratchet: the grandfather manifest's content is pinned, exact and IN
    ORDER. Growing this list — the only way to widen what the guard silently
    lets through — requires editing BOTH the committed manifest AND this pin
    in the same PR, exactly the friction EXPECTED_FENCE gives the operator
    fence. A manifest edit that is not matched here fails CI closed.
    """
    found = grandfather_manifest_ids()
    assert found == EXPECTED_GRANDFATHER, (
        "grandfathered-force-rls-migrations.yaml drifted from the pinned "
        "snapshot baseline. If this is a deliberate change it must be "
        "justified same as any other change to what the FORCE-RLS guard lets "
        "through unclassified — update EXPECTED_GRANDFATHER in the same PR.\n"
        f"  found:    {found}\n"
        f"  expected: {EXPECTED_GRANDFATHER}"
    )


def test_grandfather_manifest_shell_parse_matches_yaml_parse() -> None:
    """The sed one-liner the shipped runner actually executes must extract the
    same ids as a real YAML parse — same hazard class as the fence manifest's
    own parity test.
    """
    text = GRANDFATHER_MANIFEST_PATH.read_text(encoding="utf-8")
    assert parse_shell_manifest_ids(text) == grandfather_manifest_ids()


def test_every_grandfathered_id_names_a_real_vendored_sql_file() -> None:
    for grandfathered in EXPECTED_GRANDFATHER:
        _, node_name, filename = grandfathered.split(":", 2)
        path = (
            REPO_ROOT
            / "docker"
            / "migrations"
            / "forward"
            / "nodes"
            / node_name
            / filename
        )
        assert path.is_file(), f"grandfathered id names a missing file: {path}"


def test_grandfathered_ids_actually_declare_force_rls() -> None:
    """Every grandfathered id must genuinely need grandfathering — i.e. its
    vendored SQL must trip the same predicate the guard tests. An id that
    does NOT declare FORCE ROW LEVEL SECURITY has no business on this list;
    it would be dead weight at best and a laundering vector at worst (padding
    the snapshot with ids that don't need it, making a REAL future addition
    look like "just one more" in review).
    """
    for grandfathered in EXPECTED_GRANDFATHER:
        _, node_name, filename = grandfathered.split(":", 2)
        path = (
            REPO_ROOT
            / "docker"
            / "migrations"
            / "forward"
            / "nodes"
            / node_name
            / filename
        )
        assert _sql_declares_unclassified_force_rls(path.read_text(encoding="utf-8")), (
            f"{grandfathered} is grandfathered but its SQL does not declare "
            "FORCE ROW LEVEL SECURITY — remove it from the snapshot"
        )


def test_guard_introduction_commit_is_reachable() -> None:
    """OMN-15831: fail closed with a DIAGNOSIS, not a mystery, the moment
    GUARD_INTRODUCTION_COMMIT next rots.

    This pin has already gone unreachable twice (bbac5205 -> 7a957a0a ->
    90cd78a5) because each prior value named a PR-branch commit that a later
    squash-merge + branch deletion orphaned — `git show <pin>~1:<path>` then
    fails on any fresh checkout with no stale local objects, and the bare
    `returncode == 0` assert in test_grandfathered_ids_predate_the_guard_commit
    gave no hint why. This test runs first (alphabetically before
    ...predate...) and asserts the pin is an ancestor of HEAD before anything
    downstream tries to dereference it.
    """
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", GUARD_INTRODUCTION_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"GUARD_INTRODUCTION_COMMIT ({GUARD_INTRODUCTION_COMMIT}) is not an "
        "ancestor of HEAD — this is the squash-orphaning failure mode "
        "documented at OMN-15831 (third occurrence in the OMN-15336 lane: "
        "the pin named a PR-branch commit that was squash-merged and "
        "then had its branch deleted, so the commit object is unreachable "
        "from any ref. Repoint GUARD_INTRODUCTION_COMMIT to the SQUASH MERGE "
        "commit SHA for the PR that introduced the guard (verify with "
        "`git merge-base --is-ancestor <candidate> origin/dev`), not a "
        "branch-tip commit that will be deleted after merge.\n"
        f"stderr: {result.stderr}"
    )


def test_grandfathered_ids_predate_the_guard_commit() -> None:
    """Entry criterion #2 from the manifest's own header: every grandfathered
    id must have existed in the tree BEFORE the guard could ever have fired
    for it. Checked against the actual git history, not by assertion.
    """
    for grandfathered in EXPECTED_GRANDFATHER:
        _, node_name, filename = grandfathered.split(":", 2)
        relpath = f"docker/migrations/forward/nodes/{node_name}/{filename}"
        result = subprocess.run(
            ["git", "show", f"{GUARD_INTRODUCTION_COMMIT}~1:{relpath}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"{grandfathered} is grandfathered but did not exist at "
            f"{GUARD_INTRODUCTION_COMMIT}~1 ({relpath}) — a genuinely new "
            "migration must go through the operator fence, not the "
            f"grandfather snapshot:\n{result.stderr}"
        )


def test_grandfathered_ids_are_disjoint_from_the_fence() -> None:
    """The two lists are independent mechanisms with different semantics (a
    frozen historical fact vs. an operator-editable gate). An id on both
    would be dead code on whichever list is checked second and would blur the
    line the manifest's own header draws between them.
    """
    overlap = set(EXPECTED_GRANDFATHER) & set(EXPECTED_FENCE)
    assert not overlap, (
        f"ids present in BOTH the fence and the grandfather list: {overlap}"
    )


def test_runner_carries_the_grandfather_snapshot() -> None:
    """Static structural check: the grandfather block exists, is unconditional
    (FATAL if the manifest file is missing), and is committed-file-only (no
    operator env-var fallback) — same discipline as the fence manifest.
    """
    block = extract_grandfather_block()
    assert "GRANDFATHER_MANIFEST=" in block
    assert 'if [ ! -f "${GRANDFATHER_MANIFEST}" ]; then' in block
    assert "is_grandfathered_force_rls_migration" in block
    offenders = [
        ln
        for ln in block.splitlines()
        if re.search(
            r"GRANDFATHERED_FORCE_RLS_IDS=\$?\{?GRANDFATHERED_FORCE_RLS_IDS", ln
        )
        or re.search(r'GRANDFATHERED_FORCE_RLS_IDS="\$\{', ln)
    ]
    assert not offenders, (
        "the grandfather list must be assigned unconditionally from the "
        f"committed manifest, never from an operator env var: {offenders}"
    )


def test_grandfather_guard_is_consulted_at_the_call_site() -> None:
    """The call site must check BOTH the fence and the grandfather snapshot
    before FATALing — this is the actual repair, so it is asserted directly
    against the call site text, not just inferred from the live proofs below.
    """
    call = extract_force_rls_guard_call()
    assert "is_fenced_node_migration" in call
    assert "is_grandfathered_force_rls_migration" in call
    # Both must be negated conditions ANDed together ahead of the FATAL — a
    # call site that checked is_grandfathered_force_rls_migration but forgot
    # the `!` would silently invert the repair into "only grandfathered ids
    # are ever refused."
    assert re.search(
        r"!\s*is_fenced_node_migration.*\n.*!\s*is_grandfathered_force_rls_migration",
        call,
    ), f"expected both checks negated and ANDed ahead of the FATAL:\n{call}"


@pytest.fixture
def virgin_pg_target() -> Iterator[PgTarget]:
    """A GENUINELY empty scratch database — deliberately NOT the shared
    ``pg_target`` fixture from the OMN-15291 advisory-lock module.

    That fixture's ``SETUP_SQL`` pre-seeds a minimal ``public.db_metadata`` /
    ``public.apply_probe`` / ``public.schema_migrations`` specifically so the
    lock-race tests can treat the runner's own bootstrap DDL as a no-op —
    exactly the opposite of what a "does the real tree converge on a virgin
    database" proof needs. Reusing it here made
    ``029_create_db_metadata.sql`` see a pre-existing, schema-incompatible
    ``db_metadata`` and fail on a missing ``owner_service`` column — a
    fixture mismatch, not a FORCE-RLS defect. This fixture applies nothing
    before yielding: same connection/bring-up logic as ``pg_target``,
    minus the seed.
    """
    if not _find_pg_binary("psql"):
        _unavailable("psql client not available")

    host = os.environ.get("MIGRATION_LOCK_TEST_HOST")
    if host:
        admin = PgTarget(
            host=host,
            port=int(os.environ.get("MIGRATION_LOCK_TEST_PORT", "5432")),
            user=os.environ.get("MIGRATION_LOCK_TEST_USER", "postgres"),
            password=os.environ.get("MIGRATION_LOCK_TEST_PASSWORD", "postgres"),
            dbname=os.environ.get("MIGRATION_LOCK_TEST_DB", "postgres"),
        )
        scratch = f"omn15336_virgin_{int(time.time() * 1000) % 100_000_000}"
        _psql(admin, f'CREATE DATABASE "{scratch}"')
        scoped = PgTarget(
            host=admin.host,
            port=admin.port,
            user=admin.user,
            password=admin.password,
            dbname=scratch,
        )
        try:
            yield scoped
        finally:
            _psql(admin, f'DROP DATABASE IF EXISTS "{scratch}" WITH (FORCE)')
        return

    initdb = _find_pg_binary("initdb")
    pg_ctl = _find_pg_binary("pg_ctl")
    if not initdb or not pg_ctl:
        _unavailable(
            "no MIGRATION_LOCK_TEST_HOST and no local initdb/pg_ctl to start an "
            "ephemeral cluster"
        )
        return

    with tempfile.TemporaryDirectory(dir="/tmp", prefix="omn15336-virgin-") as base:
        datadir = Path(base) / "pgdata"
        subprocess.run(
            [initdb, "-D", str(datadir), "-U", "postgres", "--auth=trust", "--no-sync"],
            check=True,
            capture_output=True,
        )
        port = _free_port()
        subprocess.run(
            [
                pg_ctl,
                "-D",
                str(datadir),
                "-l",
                str(Path(base) / "pg.log"),
                "-o",
                f"-p {port} -c listen_addresses=127.0.0.1 "
                f"-c unix_socket_directories={base}",
                "-w",
                "start",
            ],
            check=True,
            capture_output=True,
        )
        target = PgTarget(
            host="127.0.0.1",
            port=port,
            user="postgres",
            password="postgres",
            dbname="postgres",
        )
        try:
            yield target
        finally:
            subprocess.run(
                [pg_ctl, "-D", str(datadir), "-m", "immediate", "-w", "stop"],
                check=False,
                capture_output=True,
            )


@pytest.fixture
def virgin_node_db(virgin_pg_target: PgTarget) -> Iterator[str]:
    """A separate node database against ``virgin_pg_target``'s coordinates —
    mirrors ``node_db`` above, but bound to the un-seeded target so both
    databases in the pair are genuinely virgin.

    Named LITERALLY ``omnidash_analytics``, not randomly: several REAL flat
    migrations (e.g. ``083_create_log_entries.sql``) hardcode
    ``\\connect omnidash_analytics`` rather than reading ``NODE_POSTGRES_DB``
    -- a pre-existing production assumption (that database is provisioned
    ahead of the migration run, same as ``omnibase_infra``'s own PGDB) that
    has nothing to do with this ticket's FORCE-RLS guard. A random name here
    would make ``\\connect`` fail with "database ... does not exist" for an
    unrelated reason, not prove or disprove the guard fix. ``virgin_pg_target``
    (a private ephemeral cluster, or a scratch external server used by one
    test at a time) makes the literal name safe.
    """
    admin = PgTarget(
        host=virgin_pg_target.host,
        port=virgin_pg_target.port,
        user=virgin_pg_target.user,
        password=virgin_pg_target.password,
        dbname="postgres",
    )
    name = "omnidash_analytics"
    _psql(admin, f'CREATE DATABASE "{name}"')
    try:
        yield name
    finally:
        _psql(admin, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


# --- Live proofs against the REAL vendored tree -----------------------------
# Everything above is static/structural. These two are the acceptance proof
# itself, automated: the shipped runner, unmodified, against
# docker/migrations/forward exactly as committed (no synthetic stand-in tree),
# on a virgin database. Without these, every static check above could pass
# while the runner still FATALs on a cold lane — which is exactly what
# happened: the pre-repair guard passed all 30 of its own synthetic tests
# while failing against the real tree it actually runs against in production.
REAL_FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"


@pytest.mark.integration
def test_virgin_database_applies_the_full_real_vendored_tree(
    virgin_pg_target: PgTarget,
    virgin_node_db: str,
) -> None:
    """THE acceptance proof: a virgin database converges cleanly against the
    real, committed migration tree. Before the repair this FATALed at
    node:node_canary_score_reducer:0002 with 1 node migration applied and 87
    withheld; after the repair it must reach the sentinel.
    """
    result = _run(RUNNER, virgin_pg_target, REAL_FORWARD_DIR, virgin_node_db)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        "the shipped runner must converge a virgin database against the real "
        f"vendored tree without any guard FATAL:\n{combined}"
    )
    assert "FATAL" not in combined, f"unexpected FATAL in an exit-0 run:\n{combined}"
    assert "Sentinel set. Migration gate will report HEALTHY." in combined, (
        f"expected the sentinel to be set at the end of a clean run:\n{combined}"
    )
    # Confirm the fix is real DDL, not a silent skip: at least one previously
    # ungated grandfathered table must actually carry FORCE ROW LEVEL SECURITY.
    assert (
        _psql(
            PgTarget(
                host=virgin_pg_target.host,
                port=virgin_pg_target.port,
                user=virgin_pg_target.user,
                password=virgin_pg_target.password,
                dbname=virgin_node_db,
            ),
            "SELECT relforcerowsecurity FROM pg_class WHERE relname = "
            "'capability_scores'",
        )
        == "t"
    ), "capability_scores must have FORCE ROW LEVEL SECURITY actually applied"


@pytest.mark.integration
def test_virgin_database_still_refuses_a_new_unfenced_force_rls_migration(
    virgin_pg_target: PgTarget,
    virgin_node_db: str,
    tmp_path: Path,
) -> None:
    """The original RED control, re-run against a copy of the REAL vendored
    tree (not the minimal synthetic fixture) plus one genuinely new,
    unclassified FORCE-RLS migration. Proves the repair narrows the guard's
    blind spot to exactly the pre-existing 9 — it does not disable the guard.
    """
    forward = tmp_path / "forward"
    shutil.copytree(REAL_FORWARD_DIR, forward)

    control_id = "node:node_projection_zz_new_control:0001_new_unfenced_force_rls.sql"
    control_node, control_file = (
        "node_projection_zz_new_control",
        "0001_new_unfenced_force_rls.sql",
    )
    control_dir = forward / "nodes" / control_node
    control_dir.mkdir(parents=True)
    control_sql = (
        "CREATE TABLE public.zz_new_control_marker (id INT, tenant_id TEXT);\n"
        "ALTER TABLE public.zz_new_control_marker ENABLE ROW LEVEL SECURITY;\n"
        "ALTER TABLE public.zz_new_control_marker FORCE ROW LEVEL SECURITY;\n"
    )
    (control_dir / control_file).write_text(control_sql, encoding="utf-8")
    checksum = hashlib.sha256(control_sql.encode("utf-8")).hexdigest()
    with (forward / "_ledger" / "application-migrations.tsv").open(
        "a", encoding="utf-8"
    ) as ledger:
        ledger.write(
            f"nodes/{control_node}/{control_file}\t{control_id.rsplit(':', 1)[0]}\t"
            f"{control_id.rsplit(':', 1)[0]}\ttenant\t{control_id}\t{checksum}\n"
        )

    result = _run(RUNNER, virgin_pg_target, forward, virgin_node_db)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        f"a genuinely new unfenced FORCE-RLS migration must still be refused:\n{combined}"
    )
    assert control_id in combined and "FATAL" in combined, (
        f"expected a FATAL naming the new control migration:\n{combined}"
    )
    assert not _table_exists(
        virgin_pg_target, virgin_node_db, "zz_new_control_marker"
    ), "FENCE BREACH: the new unfenced FORCE RLS migration was APPLIED"
