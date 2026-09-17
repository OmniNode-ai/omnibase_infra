# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The cloud migrations_log alias declaration may not go SILENTLY stale (OMN-18553).

## The defect

``scripts/run-forward-migrations.sh`` imports ``omninode_cloud``'s applied-set
history into the ledger. Before it does, it refuses any row in that database's
``public.migrations_log`` whose ``migration_name`` is absent from
``docker/migrations/forward/_ledger/cloud-migration-aliases.tsv``::

    RAISE EXCEPTION 'unknown cloud migrations_log alias';

That TSV is checked into THIS repository and lists names that ``omninode_infra``'s
migration corpus owns. Nothing connects the two. On 2026-09-17, once OMN-18544
let that corpus apply in full on the ``.201`` dev lane for the first time, 13 of
the 42 names the corpus had written were absent from the file and the one-shot
aborted at exit 3.

## Why this module does not assert completeness, and what it asserts instead

Completeness is the assertion you would want, and it is not decidable here: the
corpus lives in another repository and reaches the lane inside a built image, so
no test in this tree can enumerate the names the corpus will write. That is the
same boundary OMN-18544 hit, and pretending otherwise would produce a check that
passes because it can see nothing.

What IS decidable, and is the difference between this recurring and this
recurring *expensively*, is that the refusal names its offenders. The message
above named none, which is why identifying those 13 took a diagnosis pass against
a live database rather than a glance at a log line. A refusal that enumerates
what is missing cannot go silently stale: it goes loudly stale, and the repair is
reading the error and adding the rows.

The rest of this module pins the declaration's own invariants -- sorted, unique,
shaped, and identity-mapped unless something genuinely needs mapping -- because
those are the properties the runner's own `\\copy` and joins depend on, and
because the identity-mapping count is the measurement the OMN-18553 ruling on
redesigning this surface turns on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

ALIAS_TSV = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "_ledger"
    / "cloud-migration-aliases.tsv"
)
FORWARD_RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"

#: The names measured absent from the declaration on the `.201` dev lane at
#: 2026-09-17T02:1xZ, read live from `public.migrations_log` in `omninode_cloud`.
#: Listed so the fix is falsifiable against the observation that produced it: a
#: tree missing any of them is a tree the dev lane still aborts on.
MEASURED_ABSENT_2026_09_17: tuple[str, ...] = (
    "20260812_tenants_acl_state_column",
    "20260812_tenants_created_at_column",
    "20260813_gateway_sessions",
    "20260813_tenants_quota_state_column",
    "20260814_tenants_acl_state_three_state_reconciliation",
    "20260815_gateway_sessions_terminal_states",
    "20260825_backfill_gateway_workflows_completed_at",
    "20260825_gateway_workflows_result_content",
    "20260906_gateway_workflows_failure_attribution",
    "20260906_tenant_api_keys_unique_active_name",
    "20260910_gateway_workflows_terminal_provenance",
    "20260911_gateway_workflows_terminal_credential_source",
    "20260913_gateway_workflows_terminal_rule_evaluations",
)

#: A name that was ALREADY declared before this change. The positive control: it
#: must resolve in the same run the measured-absent names are checked in, so a
#: test that flags every name is distinguishable from one that checks membership.
ALREADY_DECLARED_CONTROL = "20260727_storage_bytes_metering"


def _rows() -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for number, line in enumerate(
        ALIAS_TSV.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        fields = line.split("\t")
        assert len(fields) == 2, (
            f"{ALIAS_TSV.name}:{number} has {len(fields)} tab-separated fields, not 2. "
            "The runner loads this file with \\copy and a malformed row aborts the "
            "whole import"
        )
        parsed.append((fields[0], fields[1]))
    return parsed


@pytest.mark.unit
def test_every_name_measured_absent_on_the_lab_is_now_declared() -> None:
    """The 13 names the dev lane aborted on resolve, and a prior name still does.

    RED before the fix on all 13. The control is checked in the same run, so a
    declaration file that had been emptied, or a parser that matched nothing,
    cannot pass this by flagging or resolving everything uniformly.
    """
    declared = {name for name, _ in _rows()}
    assert ALREADY_DECLARED_CONTROL in declared, (
        f"the positive control {ALREADY_DECLARED_CONTROL!r} is not declared, so this "
        "run cannot distinguish a membership check from a check of nothing"
    )
    missing = [name for name in MEASURED_ABSENT_2026_09_17 if name not in declared]
    assert not missing, (
        "these names were measured in the dev lane's migrations_log and are still "
        f"absent from {ALIAS_TSV.name}, so the forward-migration one-shot still "
        f"aborts on them: {missing}"
    )


@pytest.mark.unit
def test_the_refusal_names_the_rows_it_refuses() -> None:
    """A refusal that lists nothing is what made this cost a diagnosis pass.

    Completeness cannot be checked in this repository -- the corpus is in another
    one. So the property that keeps this cheap is not that the declaration is
    complete, but that a gap is self-describing when it is hit.
    """
    text = FORWARD_RUNNER.read_text(encoding="utf-8")
    refusal = re.search(
        r"RAISE EXCEPTION\s*\n?\s*'unknown cloud migrations_log alias[^;]*;",
        text,
    )
    assert refusal is not None, (
        f"{FORWARD_RUNNER.name} no longer raises on an unresolved migrations_log "
        "alias; that refusal is what stops an unmappable row entering the ledger"
    )
    raised = refusal.group(0)
    assert "%" in raised, (
        f"{FORWARD_RUNNER.name}'s unresolved-alias refusal names none of the rows it "
        f"refuses: {raised.strip()!r}. Identifying them then needs a live database "
        "probe, which is what this cost the first time"
    )


@pytest.mark.unit
def test_the_declaration_holds_the_invariants_the_runner_depends_on() -> None:
    """Sorted, unique on both columns, and shaped as the runner's own guard requires.

    The runner re-checks shape and duplicates at `run-forward-migrations.sh:770-779`
    and aborts the deploy when they fail. Catching it here makes that a failed test
    rather than a failed lane bring-up. Sortedness is not the runner's requirement;
    it is this file's, so an added row lands where a reader looks for it.
    """
    rows = _rows()
    assert rows, f"{ALIAS_TSV.name} is empty; the runner's \\copy would load no aliases"

    names = [name for name, _ in rows]
    versions = [version for _, version in rows]
    assert names == sorted(names), (
        f"{ALIAS_TSV.name} is not sorted by name; new rows must be inserted in order"
    )
    assert len(set(names)) == len(names), (
        f"{ALIAS_TSV.name} has a duplicate name, which the runner refuses outright"
    )
    assert len(set(versions)) == len(versions), (
        f"{ALIAS_TSV.name} has a duplicate runner_version, which the runner refuses"
    )
    for name, version in rows:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", name), f"malformed alias name {name!r}"
        assert re.fullmatch(r"[A-Za-z0-9_.-]+\.sql", version), (
            f"malformed runner_version {version!r} for {name!r}"
        )


@pytest.mark.unit
def test_the_identity_mapping_measurement_this_surface_s_ruling_turns_on() -> None:
    """Record, mechanically, how many rows carry real mapping information.

    OMN-18553 weighed replacing this allowlist with an identity default plus an
    override file. That redesign is behaviour-preserving only while every row is
    an identity mapping, and it was DECLINED for a different reason: it would
    attach a ``;migrations_log:`` provenance suffix to export rows that today
    carry none, and those rows land in a content-addressed ledger whose earlier
    imports are immutable.

    This test is the measurement that ruling rests on, kept live rather than
    quoted from a ticket. If a genuine non-identity mapping is ever added, this
    goes red and whoever adds it has to say so -- which is the point, because the
    ruling's premise would no longer hold.
    """
    non_identity = [
        (name, version) for name, version in _rows() if version != f"{name}.sql"
    ]
    assert not non_identity, (
        "this declaration now carries rows that are NOT identity mappings:\n"
        + "\n".join(f"  {name} -> {version}" for name, version in non_identity)
        + "\nThat is legitimate, and it changes the premise of the OMN-18553 ruling "
        "recorded in this module's docstring. Update that ruling in the same change."
    )
