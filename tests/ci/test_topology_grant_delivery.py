# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The topology's declared TABLE grants must be delivered by a migration.

OMN-17374. See ``scripts/validation/check_topology_grant_delivery.py`` for the
defect class and why the bound is a ratchet rather than a clean zero.

The incident this suite replays is the OMN-17374 half of it, and it is replayed
against the real checked-in topology and the real checked-in migration corpus --
no fixture topology, no synthetic corpus. A gate proven only against inputs that
cannot exhibit the failure is the OMN-15547 defect.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validation.check_topology_grant_delivery import (
    _IDENTITY_COLUMN_RE,
    _SERIAL_COLUMN_RE,
    MAX_UNDELIVERED,
    MAX_UNDELIVERED_SEQUENCES,
    GrantKey,
    SequenceKey,
    declared_grants,
    delivered_grants,
    delivered_sequences,
    sequence_backed_columns,
    undelivered,
    undelivered_sequences,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The relation whose absent grant refused BOTH the delegation writer's identity
# lookup and node_projection_tenant_registry's own INSERT on the .201 dev lane,
# 2026-09-01. Named explicitly because this pair is the incident, and a ratchet
# that merely counts would stay green if this exact grant were reverted while an
# unrelated one landed.
OMN_17374_INCIDENT = GrantKey("omninode_runtime", "public", "tenant_registry_mirror")

# OMN-17440: the subset this change delivers, named relation by relation.
#
# The subset is chosen MECHANICALLY, not by taste: it is the set of owning
# nodes OMN-17447 derived as sequence-backed -- the relations where a delivered
# TABLE grant is STILL not sufficient to write, because the INSERT fails at the
# sequence behind the key before it ever reaches the table. Delivering the
# table half here and the sequence half in OMN-17447 makes those write paths
# work end to end; delivering either alone does not.
#
# CORRECTION, measured rather than assumed: OMN-17447's derivation listed the
# three `baselines_*` child tables as BIGSERIAL because it read the CREATE
# TABLE in `node_projection_baselines/0001`. It is wrong about them. `0002`
# DROPs and recreates all three with `id TEXT PRIMARY KEY` -- the producer's
# own row id -- so in the APPLIED end state they carry no sequence at all.
# Proven by applying 0001 then 0002 to a scratch Postgres and reading
# pg_attrdef: `id type=text default=NONE`, and `pg_get_serial_sequence` returns
# NULL for each. They stay in this tranche because their TABLE grants are just
# as undelivered; they simply owe OMN-17447 nothing. That ticket's real
# deliverable is 7 sequences, not 10.
#
# Three of them (`merge_state_transitions`, `pr_lifecycle_ledger_entries`,
# `receipt_gate_rows`) are the relations OMN-17377 found sitting at ZERO ROWS
# and OMN-17379 proved refusing on the real wired path. They are not a
# hypothetical residual; they are a live outage with a measured row count.
#
# A node's grant file covers every relation that node owns and the topology
# declares -- including the ones with no sequence (`baselines_snapshots`,
# `gate_metrics`, `overnight_sessions`). Splitting a node's own relations
# across changes on a property they do not share is how a relation gets
# forgotten, which is the OMN-15701 shape.
OMN_17440_DELIVERED = frozenset(
    {
        # node_projection_baselines / 0001_create_baselines_tables.sql
        GrantKey("omninode_runtime", "public", "baselines_breakdown"),
        GrantKey("omninode_runtime", "public", "baselines_comparisons"),
        GrantKey("omninode_runtime", "public", "baselines_snapshots"),
        GrantKey("omninode_runtime", "public", "baselines_trend"),
        # node_contract_registry / 0000_create_contract_registry.sql
        GrantKey("omninode_runtime", "public", "contract_registry"),
        # node_omnigate_projection / 0000_create_gate_projection_tables.sql
        GrantKey("omninode_runtime", "public", "gate_activity"),
        GrantKey("omninode_runtime", "public", "gate_metrics"),
        # node_projection_intent_classification / 0000
        GrantKey("omninode_runtime", "public", "intent_classification_events"),
        # node_merge_state_projection / 0001 -- OMN-17379 live refusal
        GrantKey("omninode_runtime", "public", "merge_state_transitions"),
        # node_projection_overnight / 0000
        GrantKey("omninode_runtime", "public", "overnight_session_phases"),
        GrantKey("omninode_runtime", "public", "overnight_sessions"),
        # node_pr_lifecycle_state_reducer / 0001 -- OMN-17377 zero rows
        GrantKey("omninode_runtime", "public", "pr_lifecycle_ledger_entries"),
        # node_projection_receipt_gate / 0000 -- OMN-17377 zero rows
        GrantKey("omninode_runtime", "public", "receipt_gate_rows"),
    }
)

# The one relation in the declared set that NO migration can deliver: nothing in
# the corpus issues a CREATE TABLE for it, so there is no lineage to land a
# grant in. Named here so the residual stays visible as a fact rather than as an
# unexplained gap between the bound and zero. OMN-17440's own residual.
UNDELIVERABLE_NO_CREATING_MIGRATION = GrantKey(
    "omninode_runtime", "public", "nightly_loop_configs"
)


@pytest.mark.unit
def test_incident_grant_is_delivered_by_a_migration() -> None:
    """OMN-17374 replay: the registry-mirror grant reaches a real database.

    RED before ``0001_grant_omninode_runtime_tenant_registry_mirror.sql``:
    this pair was declared in all three topology instances and issued by nothing
    in the corpus, so the runtime -- which resolves this relation through the
    ``omninode_runtime_service`` binding because both contracts classify it
    ``omninode_internal`` -- got ``permission denied for table
    tenant_registry_mirror`` on every read and every write.
    """
    delivered = delivered_grants(REPO_ROOT / "docker/migrations/forward")
    assert OMN_17374_INCIDENT in delivered, (
        f"{OMN_17374_INCIDENT} is declared by the topology and issued by no "
        "migration. That is the OMN-17374 outage exactly: the delegation "
        "writer's tenant lookup and the registry projection's own INSERT both "
        "ride this grant."
    )


@pytest.mark.unit
def test_incident_grant_is_declared_by_the_topology() -> None:
    """The gate's subject is the generated topology, not a hand list.

    If the declaration ever disappears, this suite must fail rather than pass
    vacuously -- a delivered grant nobody declares is a different defect, and a
    green run for the reason "we stopped asking" is the worst outcome available.
    """
    declared = declared_grants(
        REPO_ROOT / "src/omnibase_infra/topology/instances/local.yaml"
    )
    assert OMN_17374_INCIDENT in declared


@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(OMN_17440_DELIVERED, key=str))
def test_omn_17440_subset_is_delivered_by_a_migration(key: GrantKey) -> None:
    """OMN-17440: each grant in the delivered subset reaches a real database.

    RED before this change: every one of these was declared by the generated
    topology and issued by NOTHING in the corpus, so the ``.201`` dev lane held
    them only because somebody granted them by hand -- and a fresh staging,
    onex-dev or prod lane held none of them, refusing every projection write on
    first traffic while the runtime reported healthy and committed offsets.

    Parametrized one relation per case deliberately. A single aggregate
    assertion would report "13 missing" and make the next reader re-derive
    WHICH; naming each pair means a revert of one grant fails as that grant.
    """
    delivered = delivered_grants(REPO_ROOT / "docker/migrations/forward")
    assert key in delivered, (
        f"{key} is declared by the topology and issued by no migration. "
        "Land the GRANT in the owning node's own migration lineage, next to "
        "the file that creates the relation."
    )


@pytest.mark.unit
def test_omn_17440_subset_is_declared_by_the_topology() -> None:
    """The delivered subset is a subset of what the topology actually declares.

    Guards the direction the count cannot see: a migration that grants a
    relation NOBODY declares is drift outward, not delivery. If a name here
    ever leaves the generated topology, this fails instead of the suite quietly
    asserting delivery of something the platform no longer intends to grant.
    """
    declared = declared_grants(
        REPO_ROOT / "src/omnibase_infra/topology/instances/local.yaml"
    )
    undeclared = OMN_17440_DELIVERED - declared
    assert not undeclared, (
        "these grants are delivered by a migration but declared by no "
        f"contract-derived topology entry: {sorted(map(str, undeclared))}"
    )


@pytest.mark.unit
def test_residual_relation_has_no_creating_migration() -> None:
    """The gap between the bound and zero is one relation, and it is explained.

    ``nightly_loop_configs`` is declared SELECT-only for ``omninode_runtime``
    and no migration in the corpus creates the table at all, so there is no
    lineage in which to land its grant. This asserts that stated reason is
    still TRUE rather than trusting the prose: the day someone adds the
    CREATE TABLE, this fails and the grant becomes deliverable.
    """
    corpus = REPO_ROOT / "docker/migrations/forward"
    table = UNDELIVERABLE_NO_CREATING_MIGRATION.table
    creating = [
        path.name
        for path in sorted(corpus.rglob("*.sql"))
        if re.search(
            rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:public\.)?{table}\b",
            path.read_text(encoding="utf-8", errors="replace"),
            re.IGNORECASE,
        )
    ]
    assert not creating, (
        f"{table} now has a creating migration ({creating}), so its declared "
        "grant is deliverable. Land it in that node's lineage and lower "
        "MAX_UNDELIVERED in the same change."
    )


@pytest.mark.unit
def test_undelivered_count_is_exactly_the_ratchet_bound() -> None:
    """The bound bites in both directions.

    Above it: a new topology grant landed with no migration to issue it, which
    is the next OMN-16993/OMN-17374 waiting for its relation to take traffic.

    Below it: somebody closed part of the residual and left the bound stale, so
    the gate would silently tolerate a regression back up to the old number. The
    bound moves in the same change that closes the grant, or not at all.
    """
    missing = undelivered(REPO_ROOT)
    rendered = "\n".join(f"  {key}" for key in missing)
    assert len(missing) == MAX_UNDELIVERED, (
        f"{len(missing)} undelivered topology TABLE grants, bound is "
        f"{MAX_UNDELIVERED}. Update MAX_UNDELIVERED in the same change that "
        f"moves the count.\n{rendered}"
    )


@pytest.mark.unit
def test_checker_cli_exits_zero_on_the_checked_in_tree() -> None:
    """The wired entry point is the one CI and pre-commit actually run."""
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/validation/check_topology_grant_delivery.py"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.unit
def test_checker_fails_when_the_residual_grows() -> None:
    """A tightened bound must actually reject the current tree.

    This is the RED control: it proves the gate can fail, using the real corpus
    and the real topology, without editing either.
    """
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/validation/check_topology_grant_delivery.py"),
            "--max-undelivered",
            str(MAX_UNDELIVERED - 1),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "UNDELIVERED" in completed.stdout


@pytest.mark.unit
def test_grant_on_all_tables_is_not_read_as_a_named_delivery() -> None:
    """``GRANT ... ON ALL TABLES IN SCHEMA`` delivers no NAMED relation.

    The bootstrap issues blanket grants for ``role_omnidash``. Reading one as
    satisfying a per-relation declaration would make the gate green for exactly
    the principal whose authorization is NOT blanket -- ``omninode_runtime``,
    which the bootstrap's own LOGIN_ONLY_ROLE_MAP note deliberately excludes.
    """
    delivered = delivered_grants(REPO_ROOT / "docker/migrations/forward")
    for key in delivered:
        assert key.table.upper() != "ALL"


# ---------------------------------------------------------------------------
# OMN-17447 -- the SEQUENCE half of the same defect class.
# ---------------------------------------------------------------------------

# The relations whose sequence-backed key had a delivered TABLE grant and no
# sequence grant, so every INSERT failed at the sequence before reaching the
# table. The seven omninode_runtime entries are OMN-17447's own list minus the
# three it got wrong (see OMN_17447_NOT_SEQUENCE_BACKED); the three
# tenant_projection_writer entries are ones its omninode_runtime-only scope
# never looked at, found by deriving the requirement instead of hand-listing it.
OMN_17447_DELIVERED_SEQUENCES = frozenset(
    {
        SequenceKey("omninode_runtime", "public", "contract_registry", "id"),
        SequenceKey("omninode_runtime", "public", "gate_activity", "id"),
        SequenceKey("omninode_runtime", "public", "intent_classification_events", "id"),
        SequenceKey(
            "omninode_runtime", "public", "merge_state_transitions", "projection_cursor"
        ),
        SequenceKey("omninode_runtime", "public", "overnight_session_phases", "id"),
        SequenceKey("omninode_runtime", "public", "pr_lifecycle_ledger_entries", "id"),
        SequenceKey("omninode_runtime", "public", "receipt_gate_rows", "id"),
        SequenceKey("tenant_projection_writer", "public", "capability_scores", "id"),
        SequenceKey(
            "tenant_projection_writer",
            "public",
            "delegation_routing_tenant_overlay",
            "id",
        ),
        SequenceKey("tenant_projection_writer", "public", "dep_health_findings", "id"),
    }
)

# The three OMN-17447 listed as BIGSERIAL and which are NOT sequence-backed in
# the applied end state. `node_projection_baselines/0001` does declare them
# BIGSERIAL; `0002` then DROPs and recreates all three with `id TEXT PRIMARY
# KEY`. A gate that read only CREATE statements would demand a sequence grant
# for them, and the delivering migration's own fail-loud guard would then
# RAISE on a NULL pg_get_serial_sequence -- turning the gate into a broken
# deploy. Verified on a scratch Postgres: applying 0001 then 0002 leaves
# `id type=text default=NONE` and pg_get_serial_sequence returning NULL.
OMN_17447_NOT_SEQUENCE_BACKED = frozenset(
    {"baselines_breakdown", "baselines_comparisons", "baselines_trend"}
)


@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(OMN_17447_DELIVERED_SEQUENCES, key=str))
def test_omn_17447_sequence_is_delivered_by_a_migration(key: SequenceKey) -> None:
    """Each sequence-backed key behind a declared INSERT has a USAGE grant.

    RED before this change: `GRANT USAGE ON SEQUENCE` is a statement shape the
    pre-existing `_GRANT_RE` cannot parse AT ALL, so sequence grants were
    invisible to this gate rather than merely filtered out of it -- a table
    could pass every check here and still fail every write. That is exactly
    what happened to `pr_merged_events`, which sat 24 days behind its topic at
    consumer LAG 0 while every INSERT raised `InsufficientPrivilege:
    permission denied for sequence pr_merged_events_projection_cursor_seq`.
    """
    delivered = delivered_sequences(REPO_ROOT / "docker/migrations/forward")
    assert key in delivered, (
        f"{key} is a sequence-backed column behind a declared INSERT grant "
        "with no delivering GRANT USAGE ON SEQUENCE. A TABLE grant alone does "
        "not make it writable."
    )


@pytest.mark.unit
def test_literal_sequence_name_resolves_table_and_column_with_underscores() -> None:
    delivered = delivered_sequences(REPO_ROOT / "docker/migrations/forward")

    assert (
        SequenceKey(
            "omninode_runtime",
            "public",
            "merge_state_transitions",
            "projection_cursor",
        )
        in delivered
    )


@pytest.mark.unit
def test_undelivered_sequence_count_is_exactly_the_ratchet_bound() -> None:
    """The sequence bound bites in both directions, like the table one."""
    missing = undelivered_sequences(REPO_ROOT)
    rendered = "\n".join(f"  {key}" for key in missing)
    assert len(missing) == MAX_UNDELIVERED_SEQUENCES, (
        f"{len(missing)} undelivered sequence grants, bound is "
        f"{MAX_UNDELIVERED_SEQUENCES}. Update MAX_UNDELIVERED_SEQUENCES in the "
        f"same change that moves the count.\n{rendered}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("table", sorted(OMN_17447_NOT_SEQUENCE_BACKED))
def test_recreated_tables_are_not_reported_as_sequence_backed(table: str) -> None:
    """A later DROP+recreate wins over the original BIGSERIAL declaration.

    This is the regression guard for the one thing OMN-17447's filed
    derivation got wrong. If `sequence_backed_columns` ever reverts to reading
    CREATE statements without honouring the DROP that precedes the recreate,
    these three come back as gapped sequences, someone lands grant migrations
    for them, and every lane's deploy fails on the fail-loud NULL guard.
    """
    columns = sequence_backed_columns(REPO_ROOT / "docker/migrations/forward")
    assert not columns.get(("public", table)), (
        f"{table} is reported as sequence-backed, but node_projection_baselines"
        "/0002 recreates it with `id TEXT PRIMARY KEY`. Reading only the "
        "CREATE in 0001 is the error this test exists to catch."
    )


@pytest.mark.unit
def test_identity_columns_are_not_required_to_carry_a_sequence_grant() -> None:
    """An IDENTITY column's sequence rides the table's own INSERT privilege.

    Only SERIAL/BIGSERIAL create a STANDALONE sequence with its own ACL. Asking
    for a separate USAGE grant on an identity column's implicit sequence would
    be wrong, and the delivering migration would fail loud on it.
    """
    body = (
        "CREATE TABLE public.example (\n"
        "  a BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\n"
        "  b BIGSERIAL,\n"
        "  c TEXT\n"
        ");\n"
    )
    identity = {m.group("column") for m in _IDENTITY_COLUMN_RE.finditer(body)}
    serial = {m.group("column") for m in _SERIAL_COLUMN_RE.finditer(body)}
    assert "a" in identity
    assert serial - identity == {"b"}


@pytest.mark.unit
def test_checker_fails_when_the_sequence_residual_grows() -> None:
    """RED control for the sequence half, against the real corpus.

    Proves the sequence gate can actually fail without editing the corpus or
    the topology -- the OMN-15547 requirement that a gate be shown to bite.
    """
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/validation/check_topology_grant_delivery.py"),
            "--max-undelivered-sequences",
            "-1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "sequence grant delivery" in completed.stdout
