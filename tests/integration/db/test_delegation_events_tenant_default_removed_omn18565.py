# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18565: the vendored 0042 removes the house-tenant DEFAULT, and the row-level
security refusal it makes possible is reproduced against a real PostgreSQL.

WHAT THIS REPO OWNS AND WHY THE PROOF LIVES HERE. `delegation_events` is written
by omnimarket, but the SQL that reaches a lane is the copy vendored under
`docker/migrations/forward/nodes/`, applied by this repo's forward-migration
runner. A migration that is correct in its source repo and wrong in the vendored
tree is exactly the drift the vendor-parity gate exists to catch, so the applied
behaviour is asserted against the vendored bytes -- the file the runner will
actually execute -- rather than against omnimarket's copy of it.

THE DEFECT THE MIGRATION CLOSES. Rows for one correlation are written by TWO
independent Kafka subscriptions inside the same deployment: the quality-gate
verdict and the delegation terminal. Both UPSERT on `correlation_id`, so
whichever lands first CREATES the row. The verdict carried no tenant, and with a
house-tenant column DEFAULT in place a write that said NOTHING about its tenant
became a write that ASSERTED one -- authored by the schema, invisible to the
writer that appeared to have made it, and indistinguishable to a reader from a
deliberate attribution. The terminal then upserted the same correlation under
the real submitting tenant, and because the relation carries FORCE ROW LEVEL
SECURITY and `tenant_isolation` is FOR ALL, PostgreSQL evaluated the policy's
USING half against the PRE-EXISTING house row on the ON CONFLICT DO UPDATE path
and refused:

    new row violates row-level security policy (USING expression)
    for table "delegation_events"

The terminal write never landed and the staging business proof failed on
`quality_gate`. When the terminal won the race instead, the row was created
attributed and the proof passed -- roughly three passes in sixteen proof runs
over 24 hours on 2026-09-17.

WHY A REAL DATABASE, AND WHY A LEAST-PRIVILEGE ROLE. The defect is a property of
how PostgreSQL evaluates a policy across the two arms of an UPSERT. No mock has
a policy at all, and a SUPERUSER connection is exempt from one even under FORCE
ROW LEVEL SECURITY, so a superuser fixture would pass vacuously. This module
brings up an ephemeral cluster and drives the writes as a `NOSUPERUSER
NOBYPASSRLS` LOGIN role, and asserts both of those properties before asserting
any behaviour.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

psycopg2 = pytest.importorskip(
    "psycopg2", reason="psycopg2 required for the OMN-18565 row-level-security proof"
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.slow]

_OWNER_ROLE = "postgres"
_WRITER_ROLE = "omn18565_writer"
_WRITER_SECRET = "omn18565"  # pragma: allowlist secret
_DATABASE = "omnidash_analytics"
_TABLE = "delegation_events"

#: The house tenant the OMN-16831 writer ruling stamps for an unattributed row,
#: and -- until this migration -- the column DEFAULT.
_HOUSE_TENANT = "820272f9-4aaf-5add-a2df-0af942852ab2"
#: The submitting tenant of the staging business proof, as measured.
_REAL_TENANT = "91c74442-1233-4c97-b191-911a10346fdf"

#: The VENDORED file, not omnimarket's copy. This is the one the forward runner
#: executes, and asserting against it is what makes this a proof about what a
#: lane will do rather than about what another repo intends.
_VENDORED_MIGRATION = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_delegation"
    / "0042_delegation_events_drop_house_tenant_default.sql"
)

#: The onex-dev PRE-STATE, reduced to the columns this proof needs: a UUID
#: tenant column carrying the house DEFAULT, NOT NULL, under an enabled and
#: FORCED `tenant_isolation` policy. The migration must find a default here to
#: remove, or the assertions below would pass against a schema nobody changed.
_PRE_STATE_SQL = f"""
CREATE TABLE {_TABLE} (
    correlation_id UUID PRIMARY KEY,
    task_type      TEXT NOT NULL DEFAULT '',
    tenant_id      UUID NOT NULL DEFAULT '{_HOUSE_TENANT}'::uuid
);
ALTER TABLE {_TABLE} ENABLE ROW LEVEL SECURITY;
ALTER TABLE {_TABLE} FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON {_TABLE}
  FOR ALL
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
  WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);
"""


def _pg_bin(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for prefix in sorted(Path("/opt/homebrew/opt").glob("postgresql@*"), reverse=True):
        candidate = prefix / "bin" / name
        if candidate.exists():
            return str(candidate)
    return None


_INITDB = _pg_bin("initdb")
_PG_CTL = _pg_bin("pg_ctl")
if not _INITDB or not _PG_CTL:  # pragma: no cover - environment dependent
    pytest.skip(
        "initdb/pg_ctl not available -- cannot bring up ephemeral PostgreSQL",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def pg_socket_dir() -> Iterator[str]:
    root = Path(tempfile.mkdtemp(prefix="omn18565-pg-"))
    data_dir = root / "data"
    socket_dir = root / "socket"
    socket_dir.mkdir()
    env = {**os.environ, "LANG": "C", "LC_ALL": "C", "LC_CTYPE": "C"}
    subprocess.run(
        [
            str(_INITDB),
            "-D",
            str(data_dir),
            "-U",
            _OWNER_ROLE,
            "--auth-local=trust",
            "--auth-host=trust",
            "-E",
            "UTF8",
        ],
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(
        [
            str(_PG_CTL),
            "-D",
            str(data_dir),
            "-l",
            str(root / "postgres.log"),
            "-o",
            f"-k {socket_dir} -h ''",
            "-w",
            "start",
        ],
        check=True,
        capture_output=True,
        env=env,
    )
    try:
        yield str(socket_dir)
    finally:
        subprocess.run(
            [str(_PG_CTL), "-D", str(data_dir), "-m", "immediate", "-w", "stop"],
            check=False,
            capture_output=True,
            env=env,
        )
        time.sleep(0.2)
        shutil.rmtree(root, ignore_errors=True)


def _connect(socket_dir: str, role: str) -> object:
    conn = psycopg2.connect(f"host={socket_dir} dbname={_DATABASE} user={role}")
    conn.autocommit = True
    return conn


@pytest.fixture(scope="module")
def lane(pg_socket_dir: str) -> Iterator[tuple[str, object]]:
    """Yield ``(socket_dir, owner_conn)`` with the pre-state built, the writer
    role provisioned, and the VENDORED migration applied on top."""
    bootstrap = psycopg2.connect(
        f"host={pg_socket_dir} dbname=postgres user={_OWNER_ROLE}"
    )
    bootstrap.autocommit = True
    with bootstrap.cursor() as cur:
        cur.execute(f"CREATE DATABASE {_DATABASE}")
    bootstrap.close()

    owner = _connect(pg_socket_dir, _OWNER_ROLE)
    with owner.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(_PRE_STATE_SQL)
        cur.execute(
            f"CREATE ROLE {_WRITER_ROLE} LOGIN PASSWORD %s "
            "NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION",
            (_WRITER_SECRET,),
        )
        cur.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {_TABLE} TO {_WRITER_ROLE}"
        )
        # The pre-state is asserted here, not in a test, because every test in
        # this module depends on the fixture really having installed the thing
        # the migration is supposed to remove.
        cur.execute(
            "SELECT column_default FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s "
            "AND column_name = 'tenant_id'",
            (_TABLE,),
        )
        record = cur.fetchone()
        assert record is not None and record[0] is not None
        assert _HOUSE_TENANT in str(record[0]), (
            "the fixture did not install the house-tenant DEFAULT, so the "
            "migration below would have nothing to remove"
        )
        cur.execute(_VENDORED_MIGRATION.read_text(encoding="utf-8"))
    try:
        yield pg_socket_dir, owner
    finally:
        owner.close()  # type: ignore[attr-defined]


@pytest.mark.integration
def test_the_fixture_really_binds_row_level_security(
    lane: tuple[str, object],
) -> None:
    """Positive control. Without it every assertion below could pass for the
    wrong reason on a connection the policy never applied to."""
    _socket_dir, owner = lane
    with owner.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = %s",
            (_WRITER_ROLE,),
        )
        role = cur.fetchone()
        cur.execute(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
            "WHERE oid = %s::regclass",
            (_TABLE,),
        )
        relation = cur.fetchone()
    assert role is not None and role[0] is False and role[1] is False
    assert relation is not None and relation[0] is True and relation[1] is True


@pytest.mark.integration
def test_the_vendored_migration_removes_the_default_and_keeps_not_null(
    lane: tuple[str, object],
) -> None:
    _socket_dir, owner = lane
    with owner.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(
            "SELECT column_default, is_nullable FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s "
            "AND column_name = 'tenant_id'",
            (_TABLE,),
        )
        record = cur.fetchone()
    assert record is not None
    assert record[0] is None, (
        "the vendored 0042 left a column DEFAULT in place; an unattributed "
        "write is still silently house-attributed by the schema"
    )
    assert record[1] == "NO", (
        "dropping the DEFAULT must not relax NOT NULL: an unattributed write "
        "has to fail, not store a NULL tenant that every tenant-scoped reader "
        "treats as absent"
    )


@pytest.mark.integration
def test_applying_the_migration_twice_is_a_no_op(lane: tuple[str, object]) -> None:
    """A forward migration is re-applied on redeploys and on a lane that never
    carried the default. Failing the second time would wedge those lanes."""
    _socket_dir, owner = lane
    with owner.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(_VENDORED_MIGRATION.read_text(encoding="utf-8"))
        cur.execute(
            "SELECT column_default FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s "
            "AND column_name = 'tenant_id'",
            (_TABLE,),
        )
        record = cur.fetchone()
    assert record is not None and record[0] is None


@pytest.mark.integration
def test_a_write_that_names_no_tenant_is_refused(lane: tuple[str, object]) -> None:
    """The behaviour the DDL change buys, stated as behaviour.

    WHICH refusal is PostgreSQL's choice of the first constraint it reaches, and
    both are correct: under the policy, WITH CHECK is evaluated against the
    proposed row where ``tenant_id`` is now NULL, so the comparison is NULL and
    the write is denied as a policy violation before NOT NULL is consulted.
    Asserting one SQLSTATE would pin an evaluation-order detail, so both are
    accepted and the assertion that matters is that no row exists afterwards.
    """
    socket_dir, owner = lane
    correlation_id = str(uuid4())
    writer = _connect(socket_dir, _WRITER_ROLE)
    try:
        writer.autocommit = False  # type: ignore[attr-defined]
        with writer.cursor() as cur:  # type: ignore[attr-defined]
            cur.execute(
                "SELECT set_config('app.tenant_id', %s, true)", (_HOUSE_TENANT,)
            )
            with pytest.raises(
                (
                    psycopg2.errors.NotNullViolation,
                    psycopg2.errors.InsufficientPrivilege,
                )
            ):
                cur.execute(
                    "INSERT INTO delegation_events (correlation_id) VALUES (%s)",
                    (correlation_id,),
                )
        writer.rollback()  # type: ignore[attr-defined]
    finally:
        writer.close()  # type: ignore[attr-defined]

    with owner.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(
            "SELECT count(*) FROM delegation_events WHERE correlation_id = %s",
            (correlation_id,),
        )
        count = cur.fetchone()
    assert count is not None and int(count[0]) == 0


@pytest.mark.integration
def test_a_write_that_names_its_tenant_still_lands(lane: tuple[str, object]) -> None:
    """Negative control on the refusal above.

    Without it, a refusal caused by anything else -- a missing grant, a broken
    policy -- would read exactly like a proof that the DEFAULT's removal worked.
    """
    socket_dir, owner = lane
    correlation_id = str(uuid4())
    writer = _connect(socket_dir, _WRITER_ROLE)
    try:
        writer.autocommit = False  # type: ignore[attr-defined]
        with writer.cursor() as cur:  # type: ignore[attr-defined]
            cur.execute("SELECT set_config('app.tenant_id', %s, true)", (_REAL_TENANT,))
            cur.execute(
                "INSERT INTO delegation_events (correlation_id, tenant_id) "
                "VALUES (%s, %s::uuid)",
                (correlation_id, _REAL_TENANT),
            )
        writer.commit()  # type: ignore[attr-defined]
    finally:
        writer.close()  # type: ignore[attr-defined]

    with owner.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(
            "SELECT tenant_id FROM delegation_events WHERE correlation_id = %s",
            (correlation_id,),
        )
        record = cur.fetchone()
    assert record is not None and str(record[0]) == _REAL_TENANT


@pytest.mark.integration
def test_the_cross_tenant_conflict_update_refusal_is_pinned(
    lane: tuple[str, object],
) -> None:
    """AC3. The refusal the whole ticket rests on, reproduced against a real
    policy so a later reader can tell a fixed writer from a weakened policy.

    This is characterisation and it must keep raising. The ``(USING
    expression)`` suffix is the discriminator: a plain INSERT refusal does not
    carry it, only the conflict-update path does, because only that path
    evaluates the policy against a PRE-EXISTING row.
    """
    socket_dir, owner = lane
    correlation_id = str(uuid4())
    with owner.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(
            "INSERT INTO delegation_events (correlation_id, tenant_id) "
            "VALUES (%s, %s::uuid)",
            (correlation_id, _HOUSE_TENANT),
        )
    writer = _connect(socket_dir, _WRITER_ROLE)
    try:
        writer.autocommit = False  # type: ignore[attr-defined]
        with writer.cursor() as cur:  # type: ignore[attr-defined]
            cur.execute("SELECT set_config('app.tenant_id', %s, true)", (_REAL_TENANT,))
            with pytest.raises(psycopg2.errors.InsufficientPrivilege) as excinfo:
                cur.execute(
                    "INSERT INTO delegation_events (correlation_id, tenant_id) "
                    "VALUES (%s, %s::uuid) "
                    "ON CONFLICT (correlation_id) DO UPDATE "
                    "SET tenant_id = EXCLUDED.tenant_id",
                    (correlation_id, _REAL_TENANT),
                )
        writer.rollback()  # type: ignore[attr-defined]
    finally:
        writer.close()  # type: ignore[attr-defined]
    message = str(excinfo.value)
    assert "row-level security policy" in message
    assert "(USING expression)" in message
