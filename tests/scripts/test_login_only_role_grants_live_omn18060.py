# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18060 — the runner must deliver ``chain_canary_reader``'s grants, live.

``node_chain_canary_effect`` proves OMN-16025 link 2 by reading the delegation
projection back as a least-privilege identity. omnibase_infra#3345 provisioned
that identity's LOGIN credential in the two deployment-owned seams; its
AUTHORIZATION was parked at ``docker/migrations/_blocked/`` because a grant on
``omnibase_infra.public.delegation_workflow_state`` can only be delivered by a
FLAT forward migration, and the flat stream is frozen byte-identically by
``tests/unit/db/test_migration_104_retired_omn17923.py``.

These proofs drive **the artifact that actually runs** — the shipped
``scripts/run-forward-migrations.sh`` — against a throwaway Postgres, and read
the outcome back out of ``has_column_privilege`` / ``has_table_privilege``
rather than out of the runner's own log lines. A seam that prints ``ok`` and
grants nothing is precisely the failure mode being closed.

Controls, because a privilege assertion with no control is unfalsifiable:

* **RED** — the same harness against a copy of the runner with the
  marker-delimited grant seam mechanically stripped. Derived from the shipped
  file rather than pinned as a literal, so the control cannot drift away from
  the thing it controls for.
* **NEGATIVE** — the granted role must NOT be able to read ``payload`` or
  ``tenant_id``. A column-scoped grant that is silently relation-wide reads
  identical to a correct one on the two columns it is supposed to cover.
* **POSITIVE** — the superuser CAN read ``payload`` on the same relation in the
  same cluster, so a ``False`` from the negative control is a privilege fact
  and not a broken probe or a missing column.

The scratch cluster is reused from the OMN-15291 advisory-lock proofs rather
than re-declared here: a second hand-maintained copy of a fixture whose job is
to make security proofs honest is how the two copies drift and one of them
quietly stops proving anything.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.scripts.test_forward_migration_advisory_lock import (
    PgTarget,
    _find_pg_binary,
    _psql,
    pg_target,
)

# `pg_target` is imported for its fixture registration in THIS module's
# namespace, not called by name here. Reused rather than re-declared: a second
# hand-maintained scratch-cluster fixture is how two security proofs drift.
__all__ = ["pg_target"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"

BEGIN_MARKER = "# ---- BEGIN login-only role grant seam (OMN-18060) ----"
END_MARKER = "# ---- END login-only role grant seam (OMN-18060) ----"

ROLE = "chain_canary_reader"
RELATION = "public.delegation_workflow_state"
GRANTED_COLUMNS = ("correlation_id", "state")
# Deliberately NOT granted: the delegation's own request/response material and
# the tenant discriminator. A CI liveness probe needs one enum-ish string.
WITHHELD_COLUMNS = ("payload", "tenant_id")

# A throwaway value for a throwaway cluster that is created and destroyed
# inside one test. It is NOT the lane credential and is never read from the
# lane store: `CHAIN_CANARY_READER_PASSWORD` on the .201 dev lane is referenced
# by name only and is not rotated, minted or printed by anything here.
# Hex-only because both seams validate the shape (`openssl rand -hex 32`).
FIXTURE_ROLE_PASSWORD = "00" * 32

# The relation the flat stream owns. Reproduced here at its minimum shape --
# the two granted columns plus the two withheld ones -- because the frozen flat
# stream may not be imported into a fixture and must not be edited.
RELATION_DDL = """\
CREATE TABLE public.delegation_workflow_state (
  correlation_id TEXT PRIMARY KEY,
  state          TEXT NOT NULL,
  tenant_id      TEXT,
  payload        JSONB
);
"""


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _seam_bounds(lines: list[str]) -> tuple[int, int]:
    starts = [i for i, ln in enumerate(lines) if ln.strip() == BEGIN_MARKER]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == END_MARKER]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        msg = (
            "scripts/run-forward-migrations.sh: expected exactly one OMN-18060 "
            f"grant seam delimited by its markers (found {len(starts)} begin / "
            f"{len(ends)} end)"
        )
        raise AssertionError(msg)
    return starts[0], ends[0]


def extract_grant_seam() -> str:
    """The grant seam, markers stripped.

    Raises with a specific message when the markers are missing — that is how a
    silent removal of the seam surfaces as a failure rather than a vacuous pass.
    """
    lines = _runner_text().splitlines()
    start, end = _seam_bounds(lines)
    return "\n".join(lines[start + 1 : end]) + "\n"


def strip_grant_seam(text: str) -> str:
    """The pre-OMN-18060 runner: byte-identical minus the grant seam."""
    lines = text.splitlines(keepends=True)
    start, end = _seam_bounds([ln.rstrip("\n") for ln in lines])
    return "".join(lines[:start] + lines[end + 1 :])


# ---------------------------------------------------------------------------
# Live proofs
# ---------------------------------------------------------------------------


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    """A minimal flat stream that creates the relation the grant targets.

    The real ``docker/migrations/forward`` tree is NOT used: it is frozen
    byte-identically by OMN-17923's retirement proof, and pointing the runner at
    it would make this test's outcome depend on 88 unrelated migrations.
    """
    forward = tmp_path / "migrations" / "forward"
    forward.mkdir(parents=True)
    (forward / "001_delegation_workflow_state.sql").write_text(
        RELATION_DDL, encoding="utf-8"
    )
    (forward / "fenced-node-migrations.yaml").write_text(
        "fenced_node_migrations: []\n", encoding="utf-8"
    )
    (forward / "grandfathered-force-rls-migrations.yaml").write_text(
        "grandfathered_force_rls_migrations: []\n", encoding="utf-8"
    )
    return forward


@pytest.fixture
def empty_migrations_dir(tmp_path: Path) -> Path:
    """A flat stream that creates NOTHING — the relation-absent lane state."""
    forward = tmp_path / "no_relation" / "forward"
    forward.mkdir(parents=True)
    (forward / "001_unrelated.sql").write_text(
        "CREATE TABLE public.unrelated (id INT PRIMARY KEY);\n", encoding="utf-8"
    )
    (forward / "fenced-node-migrations.yaml").write_text(
        "fenced_node_migrations: []\n", encoding="utf-8"
    )
    (forward / "grandfathered-force-rls-migrations.yaml").write_text(
        "grandfathered_force_rls_migrations: []\n", encoding="utf-8"
    )
    return forward


def _run_runner(
    runner: Path, target: PgTarget, migrations: Path
) -> subprocess.CompletedProcess[str]:
    psql_dir = str(Path(_find_pg_binary("psql") or "psql").parent)
    env = {
        **os.environ,
        "PATH": f"{psql_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "POSTGRES_USER": target.user,
        "POSTGRES_PASSWORD": target.password,
        "POSTGRES_HOST": target.host,
        "POSTGRES_PORT": str(target.port),
        "POSTGRES_DB": target.dbname,
        "MIGRATIONS_DIR": str(migrations),
        "NODE_MIGRATIONS_DIR": str(migrations / "nodes"),
        "MIGRATION_LOCK_WAIT_SECONDS": "60",
        # Section 0 mints the LOGIN for the principal whose grants section 3b
        # then asserts. Without it the role is absent and the grant seam takes
        # its "role not provisioned" skip, which would make every assertion
        # below vacuous.
        "CHAIN_CANARY_READER_PASSWORD": FIXTURE_ROLE_PASSWORD,
    }
    return subprocess.run(
        ["/bin/sh", str(runner)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _column_privilege(target: PgTarget, column: str) -> bool:
    return (
        _psql(
            target,
            f"SELECT has_column_privilege('{ROLE}', '{RELATION}', "
            f"'{column}', 'SELECT')",
        )
        == "t"
    )


def _superuser_column_privilege(target: PgTarget, column: str) -> bool:
    return (
        _psql(
            target,
            f"SELECT has_column_privilege('{target.user}', '{RELATION}', "
            f"'{column}', 'SELECT')",
        )
        == "t"
    )


def _role_exists(target: PgTarget) -> bool:
    # S608: every interpolated value in this module is a module-level literal
    # naming a role, relation or column of this repo's own schema — there is no
    # caller-supplied input anywhere in the file.
    query = f"SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = '{ROLE}'"  # noqa: S608
    return _psql(target, query) == "1"


@pytest.mark.integration
def test_runner_without_the_grant_seam_leaves_the_reader_unable_to_read(
    pg_target: PgTarget,
    migrations_dir: Path,
    tmp_path: Path,
) -> None:
    """RED control: strip the seam and link 2 stays unreadable.

    This is the state that shipped with omnibase_infra#3345 and produced the
    canary's link-2 ``InvalidPasswordError`` / permission failure: the role is
    provisioned, the relation exists, and the reader holds nothing on it.
    """
    stripped = tmp_path / "runner_without_grant_seam.sh"
    stripped.write_text(strip_grant_seam(_runner_text()), encoding="utf-8")
    stripped.chmod(0o755)

    result = _run_runner(stripped, pg_target, migrations_dir)
    assert result.returncode == 0, result.stdout

    assert _role_exists(pg_target), (
        "the credential seam must still provision the role in the RED control — "
        "otherwise this test proves nothing about the GRANT half"
    )
    for column in GRANTED_COLUMNS:
        assert not _column_privilege(pg_target, column), (
            f"{ROLE} can already SELECT {column} without the OMN-18060 seam; "
            "the GREEN assertion below would be vacuous"
        )


@pytest.mark.integration
def test_runner_grants_the_reader_exactly_its_two_columns(
    pg_target: PgTarget,
    migrations_dir: Path,
) -> None:
    """GREEN + negative + positive control, in one cluster.

    The negative control is the point of the whole ticket: a relation-wide
    ``GRANT SELECT`` would satisfy the two GREEN assertions identically while
    handing a scheduled CI probe every tenant's delegation payloads.
    """
    result = _run_runner(RUNNER, pg_target, migrations_dir)
    assert result.returncode == 0, result.stdout

    for column in GRANTED_COLUMNS:
        assert _column_privilege(pg_target, column), (
            f"{ROLE} cannot SELECT {column} after the runner ran; the chain "
            f"canary's link-2 readback fails. Runner output:\n{result.stdout}"
        )

    for column in WITHHELD_COLUMNS:
        assert not _column_privilege(pg_target, column), (
            f"{ROLE} can SELECT {column} — the grant is not column-scoped, so a "
            "scheduled probe holds material it has no need for"
        )
        assert _superuser_column_privilege(pg_target, column), (
            f"positive control failed: the superuser cannot SELECT {column} "
            "either, so the negative control above proves nothing about "
            f"{ROLE}'s privileges"
        )

    assert (
        _psql(
            pg_target,
            f"SELECT has_table_privilege('{ROLE}', '{RELATION}', 'SELECT')",
        )
        == "f"
    ), (
        f"{ROLE} holds RELATION-level SELECT; column scoping is defeated and "
        "every column of the relation is readable"
    )
    assert (
        _psql(
            pg_target,
            f"SELECT has_database_privilege('{ROLE}', current_database(), 'CONNECT')",
        )
        == "t"
    )


@pytest.mark.integration
def test_reader_holds_no_write_privilege_of_any_kind(
    pg_target: PgTarget,
    migrations_dir: Path,
) -> None:
    """An instrument that can write is not a read."""
    result = _run_runner(RUNNER, pg_target, migrations_dir)
    assert result.returncode == 0, result.stdout

    for privilege in ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES"):
        assert (
            _psql(
                pg_target,
                f"SELECT has_table_privilege('{ROLE}', '{RELATION}', '{privilege}')",
            )
            == "f"
        ), f"{ROLE} holds {privilege} on {RELATION}"
    # The role's OWN acl entry on schema public, not its effective privilege.
    # `has_schema_privilege` folds in whatever PUBLIC holds, and PUBLIC holds
    # CREATE on schema public by default on PostgreSQL 14 and earlier (removed
    # in 15; the lanes run postgres:16-alpine, the scratch cluster here is
    # whatever the host's initdb is -- 14.17 when this was written). Asserting
    # the effective privilege would therefore assert the cluster's default
    # rather than what this seam issued, and would pass or fail by host.
    # No interpolation: the whole acl array comes back and the role's entry is
    # picked out in Python, so this query carries no constructed SQL at all.
    schema_acl_rows = _psql(
        pg_target,
        "SELECT unnest(coalesce(nspacl, '{}')::text[]) "
        "FROM pg_namespace WHERE nspname = 'public'",
    )
    schema_acl = next(
        (
            row.split("=", 1)[1]
            for row in schema_acl_rows.splitlines()
            if row.startswith(f"{ROLE}=")
        ),
        "none",
    )
    assert schema_acl != "none", (
        f"positive control: {ROLE} has no acl entry at all on schema public, so "
        "the USAGE grant did not take and the assertion below is vacuous"
    )
    assert "C" not in schema_acl.split("/", 1)[0], (
        f"the seam granted {ROLE} CREATE on schema public (acl {schema_acl!r}), "
        "so it can own a table — and a table's owner is exempt from that "
        "table's row-level security unconditionally, which is exactly what the "
        "canary's pg_roles probe exists to refuse"
    )
    assert schema_acl.split("/", 1)[0] == "U", (
        f"{ROLE}'s acl on schema public is {schema_acl!r}; this seam declares "
        "USAGE and nothing else"
    )


@pytest.mark.integration
def test_absent_relation_is_a_named_skip_and_the_next_run_converges(
    pg_target: PgTarget,
    empty_migrations_dir: Path,
    migrations_dir: Path,
) -> None:
    """The gate: no relation yet is a legitimate lane state, not a failure.

    First run — the flat stream has not created the relation. The seam must
    skip with a reason naming the relation, and must NOT fail the run (a failed
    run leaves ``migrations_complete`` FALSE and the migration gate UNHEALTHY,
    which would refuse to start the runtime over a grant that is simply early).

    Second run — the relation now exists. The same seam must converge without
    any operator action, which is the whole reason it is re-asserted on every
    compose up rather than applied once.
    """
    first = _run_runner(RUNNER, pg_target, empty_migrations_dir)
    assert first.returncode == 0, first.stdout
    assert RELATION in first.stdout and "does not exist yet" in first.stdout, (
        "the skip must name the relation it is waiting on; an unexplained skip "
        f"is indistinguishable from a seam that never ran. Output:\n{first.stdout}"
    )
    assert _role_exists(pg_target), "the credential seam must still have run"

    second = _run_runner(RUNNER, pg_target, migrations_dir)
    assert second.returncode == 0, second.stdout
    for column in GRANTED_COLUMNS:
        assert _column_privilege(pg_target, column), (
            f"{ROLE} still cannot SELECT {column} on the run after the relation "
            f"appeared; the seam does not converge. Output:\n{second.stdout}"
        )


@pytest.mark.integration
def test_grants_are_idempotent_across_repeated_runs(
    pg_target: PgTarget,
    migrations_dir: Path,
) -> None:
    """Every compose up re-asserts. The second one must not fail or widen."""
    assert _run_runner(RUNNER, pg_target, migrations_dir).returncode == 0
    second = _run_runner(RUNNER, pg_target, migrations_dir)
    assert second.returncode == 0, second.stdout

    for column in GRANTED_COLUMNS:
        assert _column_privilege(pg_target, column)
    for column in WITHHELD_COLUMNS:
        assert not _column_privilege(pg_target, column), (
            f"a re-assert widened {ROLE} to {column}"
        )


@pytest.mark.integration
def test_relation_wide_select_is_refused_rather_than_tolerated(
    pg_target: PgTarget,
    migrations_dir: Path,
) -> None:
    """A pre-existing table-wide SELECT defeats the column scoping silently.

    ``has_column_privilege`` returns true for a relation-level grant, so a lane
    where somebody granted the whole relation would pass every column assertion
    above while the reader held ``payload``. The seam refuses that state loudly
    instead of re-asserting a narrower grant on top of a wider one and calling
    it least privilege.
    """
    assert _run_runner(RUNNER, pg_target, migrations_dir).returncode == 0
    _psql(pg_target, f'GRANT SELECT ON {RELATION} TO "{ROLE}"')
    assert _column_privilege(pg_target, "payload"), (
        "positive control: the widening GRANT must actually have taken effect"
    )

    widened = _run_runner(RUNNER, pg_target, migrations_dir)
    assert widened.returncode != 0, (
        "the runner accepted a relation-wide SELECT on the reader; the column "
        f"scoping is unenforced. Output:\n{widened.stdout}"
    )
    assert "RELATION-WIDE SELECT" in widened.stdout + widened.stderr


@pytest.mark.integration
def test_escalated_reader_is_refused_rather_than_granted(
    pg_target: PgTarget,
    migrations_dir: Path,
) -> None:
    """A column-scoped grant to a BYPASSRLS role is a grant that constrains nothing.

    This is the half the parked ``_blocked`` migration carried as a correcting
    ``ALTER ROLE``. Delivered here as detection + refusal instead: correcting it
    would demand role-administration privileges this seam deliberately does not
    hold, and the escalation belongs to whoever made it. The canary makes the
    same check at connect time, so both ends of the contract are asserted.
    """
    assert _run_runner(RUNNER, pg_target, migrations_dir).returncode == 0
    _psql(pg_target, f'ALTER ROLE "{ROLE}" BYPASSRLS')
    assert (
        _psql(
            pg_target, "SELECT rolbypassrls FROM pg_roles WHERE rolname = current_user"
        )
        is not None
    )

    escalated = _run_runner(RUNNER, pg_target, migrations_dir)
    assert escalated.returncode != 0, (
        "the runner granted a column-scoped SELECT to a BYPASSRLS role; the "
        f"scoping constrains nothing. Output:\n{escalated.stdout}"
    )
    assert "SUPERUSER or BYPASSRLS" in escalated.stdout + escalated.stderr

    # Positive control on the refusal: remove the escalation and the same runner
    # converges, so the failure above is the attribute and not a wedged cluster.
    _psql(pg_target, f'ALTER ROLE "{ROLE}" NOBYPASSRLS')
    recovered = _run_runner(RUNNER, pg_target, migrations_dir)
    assert recovered.returncode == 0, recovered.stdout
    for column in GRANTED_COLUMNS:
        assert _column_privilege(pg_target, column)
