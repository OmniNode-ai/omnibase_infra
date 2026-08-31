# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17150 — a fence entry may not gate the sole CREATE of a gate-required table.

THE DEFECT THIS CLOSES
----------------------
``docker/migrations/forward/fenced-node-migrations.yaml`` fenced
``node:node_projection_registration:0000_create_node_service_registry.sql``, the
only migration in the corpus that creates ``node_service_registry``. At the same
time ``scripts/check_migrations_complete.sh`` listed ``node_service_registry`` in
``REQUIRED_PROJECTION_TABLES`` and refused to exit 0 until it existed. The
runtime tier (``omninode-runtime``, ``runtime-effects``, ``projection-api``) is
gated behind that healthcheck via ``depends_on: migration-gate:
{condition: service_healthy}``.

Both halves were behaving exactly as written, which is why neither looked broken:

* ``forward-migration`` completes, reports ``13 node skipped``, sets
  ``db_metadata.migrations_complete = TRUE``, and exits 0.
* ``migration-gate`` exhausts all 30 retries and sits ``unhealthy`` forever,
  because the sentinel is only the FIRST of its two checks.

Every existing lane passed only because none had been cold-booted since the
fence entry landed (2026-07-29): a lane whose ``node_service_registry`` predates
the fence keeps the table and the gate keeps passing. Found by the first lane
built from scratch against the current fence — ``omnibase-infra-lakshman``,
2026-08-31, reproduced across three clean boots. Latent, not absent: the next
clean rebuild of stability-test, prod or judge would have hit the same wall,
during an incident rebuild rather than a bring-up.

WHY THIS TEST AND NOT JUST THE FIX
----------------------------------
The fix (releasing 0000/0001 from the baseline) repairs one instance. This
repairs the class: the fence is an operator surface that gets appended to
whenever a migration needs holding, and nothing in the append path ever
consulted the healthcheck's required-table list. A future entry could recreate
the identical deadlock with an entirely different table and the same clean bill
of health from every other test in the tree.

The invariant, stated once:

    every table named in REQUIRED_PROJECTION_TABLES must be creatable by the
    BASELINE-unfenced migration set — with NO lane release applied

The "no lane release" part is the whole point. ``ONEX_MIGRATION_LANE=dev``
already released the registration trio on the lab lane, which is exactly why the
lab lane never saw this. An invariant that permitted lane releases would have
been satisfied by the dev lane and would have missed the defect on the other
four.

Static and always-on: it parses committed files, needs no database and no
Docker, and therefore gates every PR and pre-commit run rather than being an
opt-in or integration-marked check. The live end-to-end proof (drive the shipped
runner cold, then ask the shipped healthcheck) is
``test_default_lane_cold_boot_satisfies_the_migration_gate`` in
``test_node_migration_fence_parity.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
FENCE_MANIFEST = FORWARD_DIR / "fenced-node-migrations.yaml"
GATE_HEALTHCHECK = REPO_ROOT / "scripts" / "check_migrations_complete.sh"

# Every committed declaration of the gate's required-table list. Unioned rather
# than read from one authority, because they are separately maintained and a
# lane that adds a table to its own list is making the same promise: the
# invariant must hold for whatever ANY of them can demand.
#
#   * the healthcheck's own default (the value a lane inherits when its compose
#     file sets nothing),
#   * each compose lane's explicit REQUIRED_PROJECTION_TABLES,
#   * the service catalog manifest the compose files are generated against,
#   * scripts/deploy-runtime.sh and the deploy agent, which run the SAME
#     required-table check as a post-deploy verification (so a table missing
#     here fails a deploy, not only a boot).
REQUIRED_TABLE_SOURCES: tuple[tuple[str, str], ...] = (
    ("scripts/check_migrations_complete.sh", r'REQUIRED_PROJECTION_TABLES:-([^"}]+)\}'),
    (
        "docker/docker-compose.infra.yml",
        r'REQUIRED_PROJECTION_TABLES:\s*"([^"]+)"',
    ),
    (
        "docker/docker-compose.judge.yml",
        r'REQUIRED_PROJECTION_TABLES:\s*"([^"]+)"',
    ),
    (
        "docker/docker-compose.lakshman.yml",
        r'REQUIRED_PROJECTION_TABLES:\s*"([^"]+)"',
    ),
    (
        "docker/catalog/services/migration-gate.yaml",
        r"REQUIRED_PROJECTION_TABLES:\s*(.+)",
    ),
)

# The two array/tuple-shaped declarations, which do not fit the flat
# whitespace-separated grammar above.
DEPLOY_RUNTIME = "scripts/deploy-runtime.sh"
DEPLOY_AGENT = "scripts/deploy-agent/deploy_agent/executor.py"

# CREATE TABLE targets. Unqualified or explicitly public-qualified only: the
# healthcheck asks `to_regclass('public.<table>')`, so a table created in
# another schema (e.g. omninode_internal.projection_watermarks) does NOT satisfy
# it and must not be counted here.
_CREATE_TABLE = re.compile(
    r"""CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?
        (?:"?public"?\.)?
        "?(?P<table>[a-zA-Z_][a-zA-Z0-9_]*)"?
        \s*\(""",
    re.IGNORECASE | re.VERBOSE,
)

_FENCE_ID = re.compile(r'^\s*-\s*id:\s*"([^"]+)"', re.MULTILINE)


def _read(relpath: str) -> str:
    path = REPO_ROOT / relpath
    assert path.is_file(), (
        f"{relpath} does not exist. This test's whole job is to keep the gate's "
        "required-table list and the fence in agreement; if a declaration was "
        "renamed or removed, update REQUIRED_TABLE_SOURCES rather than letting "
        "the check silently cover less than it claims."
    )
    return path.read_text(encoding="utf-8")


def required_projection_tables() -> frozenset[str]:
    """Union of every committed REQUIRED_PROJECTION_TABLES declaration."""
    tables: set[str] = set()
    for relpath, pattern in REQUIRED_TABLE_SOURCES:
        text = _read(relpath)
        found = re.findall(pattern, text)
        assert found, (
            f"no REQUIRED_PROJECTION_TABLES declaration matched in {relpath}. "
            "Either the declaration moved (update the pattern) or it was "
            "deleted (update this list) — a silently-zero match would make "
            "this whole test vacuous."
        )
        for group in found:
            tables.update(group.split())

    # scripts/deploy-runtime.sh: bash array literal.
    runtime = _read(DEPLOY_RUNTIME)
    runtime_block = re.search(
        r"readonly REQUIRED_PROJECTION_TABLES=\((?P<body>.*?)\)", runtime, re.DOTALL
    )
    assert runtime_block is not None, (
        f"REQUIRED_PROJECTION_TABLES array not found in {DEPLOY_RUNTIME}"
    )
    tables.update(re.findall(r'"([^"]+)"', runtime_block.group("body")))

    # scripts/deploy-agent: python tuple literal.
    agent = _read(DEPLOY_AGENT)
    agent_block = re.search(
        r"REQUIRED_PROJECTION_TABLES:\s*tuple\[str, \.\.\.\]\s*=\s*\((?P<body>.*?)\)",
        agent,
        re.DOTALL,
    )
    assert agent_block is not None, (
        f"REQUIRED_PROJECTION_TABLES tuple not found in {DEPLOY_AGENT}"
    )
    tables.update(re.findall(r'"([^"]+)"', agent_block.group("body")))

    assert tables, "no required projection tables were parsed from any source"
    return frozenset(tables)


def fenced_ids() -> frozenset[str]:
    """The BASELINE fence, parsed the way the shipped runners parse it."""
    ids = frozenset(_FENCE_ID.findall(FENCE_MANIFEST.read_text(encoding="utf-8")))
    assert ids, (
        "the fence manifest parsed to an empty list. That is a legitimate "
        "future state, but it is also the symptom of a malformed manifest, and "
        "an empty fence would make every assertion below pass vacuously."
    )
    return ids


def _migration_id(sql_file: Path) -> str | None:
    """The id the runners mint for a node migration, or None for a flat one.

    Flat (non-node) migrations under ``forward/*.sql`` are outside this fence's
    id space entirely — it gates ``node:<node>:<file>`` only — so they are
    always 'unfenced' for the purposes of this invariant.
    """
    try:
        relative = sql_file.relative_to(FORWARD_DIR)
    except ValueError:  # pragma: no cover - defensive
        return None
    parts = relative.parts
    if len(parts) == 3 and parts[0] == "nodes":
        return f"node:{parts[1]}:{parts[2]}"
    return None


def creators_by_table() -> dict[str, list[tuple[str, str | None]]]:
    """``table -> [(relpath, migration_id | None), ...]`` for public-schema CREATEs."""
    creators: dict[str, list[tuple[str, str | None]]] = {}
    for sql_file in sorted(FORWARD_DIR.rglob("*.sql")):
        if "_ledger" in sql_file.parts:
            continue
        text = sql_file.read_text(encoding="utf-8", errors="replace")
        for match in _CREATE_TABLE.finditer(text):
            table = match.group("table")
            entry = (
                str(sql_file.relative_to(REPO_ROOT)),
                _migration_id(sql_file),
            )
            creators.setdefault(table, []).append(entry)
    return creators


@pytest.mark.unit
def test_every_gate_required_table_has_a_creating_migration() -> None:
    """The weaker half, separated so a failure says WHICH thing is wrong.

    A required table with no CREATE anywhere in the corpus is the same class of
    deadlock as a fenced CREATE, arrived at differently: the gate demands
    something the migration set cannot produce on any lane, fenced or not.
    """
    creators = creators_by_table()
    missing = sorted(t for t in required_projection_tables() if t not in creators)
    assert not missing, (
        "these tables are in REQUIRED_PROJECTION_TABLES but NO migration in "
        f"docker/migrations/forward creates them in the public schema: {missing}\n"
        "migration-gate will never report HEALTHY on a cold lane, so the "
        "runtime tier will never start. Either add the creating migration or "
        "remove the table from the required list — the two must agree."
    )


@pytest.mark.unit
def test_no_fence_entry_gates_the_sole_creator_of_a_required_table() -> None:
    """THE invariant. OMN-17150.

    RED before the fix: ``node_service_registry``'s only creator is
    ``node:node_projection_registration:0000_create_node_service_registry.sql``
    and that id is in the baseline fence.

    Deliberately evaluated against the BASELINE fence with NO lane release
    applied. ``ONEX_MIGRATION_LANE=dev`` released the registration trio on the
    lab lane, so an invariant that honoured lane releases would have been
    satisfied by dev and blind to the four lanes that actually broke.
    """
    creators = creators_by_table()
    fenced = fenced_ids()

    offenders: list[str] = []
    for table in sorted(required_projection_tables()):
        entries = creators.get(table)
        if not entries:
            # Covered, with a better message, by the test above.
            continue
        unfenced = [
            relpath
            for relpath, migration_id in entries
            if migration_id is None or migration_id not in fenced
        ]
        if not unfenced:
            gated = ", ".join(
                migration_id or relpath for relpath, migration_id in entries
            )
            offenders.append(f"{table} (every creator is fenced: {gated})")

    assert not offenders, (
        "FENCE/GATE CONTRADICTION (OMN-17150). These tables are required by "
        "scripts/check_migrations_complete.sh before migration-gate may report "
        "HEALTHY, but every migration that creates them is held by the baseline "
        "operator fence:\n  " + "\n  ".join(offenders) + "\n\n"
        "This is not a deferred decision — it is a permanent deadlock on every "
        "cold boot of every lane without a fence release. forward-migration "
        "exits 0 and sets the sentinel TRUE; migration-gate stays unhealthy "
        "forever; omninode-runtime / runtime-effects / projection-api never "
        "start. Existing lanes hide it because their tables predate the fence "
        "entry.\n\n"
        "Resolve it in THIS commit, one of two ways:\n"
        "  (a) do not fence the CREATE — hold only the dependent migration "
        "that carries the actual hazard (this is what OMN-17150 did: 0002 "
        "stays fenced, 0000/0001 do not); or\n"
        "  (b) remove the table from REQUIRED_PROJECTION_TABLES in every "
        "declaration, which is only honest if nothing downstream reads it.\n"
        "A lane-scoped override of the required list is NOT a resolution: it "
        "converts a fleet-wide defect into a local workaround and leaves the "
        "contradiction armed for the next cold rebuild."
    )


@pytest.mark.unit
def test_registry_table_creator_is_not_fenced() -> None:
    """The named regression, pinned so a revert is loud rather than latent.

    The generic invariant above would catch a re-fence of 0000 too. This states
    the specific case, because the generic message describes a class and this
    one names the file, the ticket and the lane that paid for it.
    """
    create_id = (
        "node:node_projection_registration:0000_create_node_service_registry.sql"
    )
    assert create_id not in fenced_ids(), (
        f"{create_id} is back in the baseline fence. It is the only migration "
        "that creates node_service_registry, which the migration gate requires, "
        "so re-fencing it re-arms the OMN-17150 deadlock on every cold boot "
        "outside the dev lane. If the tenant-RLS posture needs holding, hold "
        "0002 — that is the id that carries FORCE ROW LEVEL SECURITY."
    )


@pytest.mark.unit
def test_registration_rls_migration_is_still_fenced() -> None:
    """The other direction: OMN-17150 released the CREATE, NOT the RLS posture.

    Without this, "fix the deadlock" could be satisfied by unfencing the whole
    trio, which would (a) hand 0002 to the OMN-15336 item-4 unclassified-FORCE
    guard as a FATAL on any lane where it has never applied, and (b) turn FORCE
    ROW LEVEL SECURITY on for prod and judge with no path back, since the
    unfenced 0004 that reverses FORCE is already recorded as applied there and
    would be skipped.
    """
    rls_id = (
        "node:node_projection_registration:0002_node_service_registry_tenant_rls.sql"
    )
    assert rls_id in fenced_ids(), (
        f"{rls_id} left the baseline fence. It enables FORCE ROW LEVEL "
        "SECURITY and is not in grandfathered-force-rls-migrations.yaml, so an "
        "unfenced 0002 is FATAL under the OMN-15336 item-4 guard on every "
        "database where it has never applied — prod and judge included. "
        "Releasing it is a live operator decision about tenancy posture."
    )
