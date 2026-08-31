# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17139 — an in-place rewrite of an applied migration must have a way out.

``run-forward-migrations.sh`` records a ``content_sha256`` for every node
migration it applies, into ``platform_catalog.schema_migrations``. When the file
is later edited in place, ``migration_is_applied()`` finds a recorded hash that
no longer matches and exits 1 -- on that lane, on every subsequent run, forever.

That is correct and fail-closed. What was missing is the other half: a way to
*resolve* it without reverting the bytes. The three adoption relations that
already exist (OMN-15857, OMN-16915, OMN-16919) are all read by ``bootstrap.sql``
and all answer questions about an IMPORT source -- ``public.schema_migrations``,
``public.omnimarket_schema_migrations``. None of them is reachable from
``migration_is_applied()``, which reads the canonical ledger the runner writes
itself. So on 2026-08-30 the dev lane was simply un-deployable at dev tip, with
no sanctioned repair.

These proofs drive **the artifact that actually runs** -- the shipped
``scripts/run-forward-migrations.sh`` -- against a real Postgres, through the
exact sequence that produced the incident: apply, rewrite in place, re-run.

``test_inplace_rewrite_without_a_declaration_still_aborts`` is the RED control.
Without it, the GREEN proof below would also pass against a runner that had
simply stopped checking checksums, which is the one outcome worse than the bug.

Database selection matches ``test_forward_migration_advisory_lock.py``:
an ephemeral local cluster via ``initdb``/``pg_ctl``, else skip -- unless
``REQUIRE_MIGRATION_LOCK_DB`` is set, in which case fail. A check that silently
skips is a check that does not exist.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
REAL_LEDGER = REPO_ROOT / "docker" / "migrations" / "forward" / "_ledger"

NODE = "node_probe_events"
FILENAME = "0001_create_probe_events.sql"
ARTIFACT = f"nodes/{NODE}/{FILENAME}"
VERSION = f"node:{NODE}:{FILENAME}"
STREAM = f"node:{NODE}"
DOMAIN = "omninode_internal"
TICKET = "OMN-17139"
RECEIPT_SHA = "a" * 64

# Deliberately trivial and idempotent: this proof is about the ledger, not about
# what the SQL does. It lives in omninode_internal because that is the domain the
# manifest validator accepts for a node migration, and the schema is created here
# rather than assumed.
MIGRATION_V1 = """\
-- probe migration, revision 1
CREATE SCHEMA IF NOT EXISTS omninode_internal;
CREATE TABLE IF NOT EXISTS omninode_internal.probe_events (
  id BIGSERIAL PRIMARY KEY,
  label TEXT NOT NULL
);
"""

# Byte-different, program-identical: the only change is a comment. This is the
# exact shape of the omnibase_infra#3019 rewrite that caused OMN-17139.
MIGRATION_V2 = MIGRATION_V1.replace(
    "-- probe migration, revision 1",
    "-- probe migration, revision 2 (comment-only rewrite; no executable change)",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PgTarget:
    host: str
    port: int
    user: str
    password: str
    dbname: str

    def env(self) -> dict[str, str]:
        return {**os.environ, "PGPASSWORD": self.password}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _find_pg_binary(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for root, pattern in (
        ("/opt/homebrew/opt", "postgresql@*/bin"),
        ("/usr/lib/postgresql", "*/bin"),
    ):
        base = Path(root)
        if not base.exists():
            continue
        for candidate in sorted(base.glob(pattern), reverse=True):
            binary = candidate / name
            if binary.exists():
                return str(binary)
    return None


def _unavailable(reason: str) -> None:
    if os.environ.get("REQUIRE_MIGRATION_LOCK_DB"):
        pytest.fail(
            f"REQUIRE_MIGRATION_LOCK_DB is set but this proof cannot run: {reason}"
        )
    pytest.skip(reason)


NODE_DB = "omnidash_analytics"

DB_METADATA_SQL = (
    "CREATE TABLE public.db_metadata ("
    "  id BOOLEAN PRIMARY KEY DEFAULT TRUE,"
    "  migrations_complete BOOLEAN NOT NULL DEFAULT FALSE,"
    "  runner_completed_at TIMESTAMPTZ,"
    "  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());"
    " INSERT INTO public.db_metadata (id) VALUES (TRUE);"
)


def _psql(target: PgTarget, sql: str, *, dbname: str | None = None) -> str:
    result = subprocess.run(
        [
            _find_pg_binary("psql") or "psql",
            "-h",
            target.host,
            "-p",
            str(target.port),
            "-U",
            target.user,
            "-d",
            dbname or target.dbname,
            "-v",
            "ON_ERROR_STOP=1",
            "-tAc",
            sql,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=target.env(),
    )
    return result.stdout.strip()


@pytest.fixture
def pg_target() -> Iterator[PgTarget]:
    if not _find_pg_binary("psql"):
        _unavailable("psql client not available")
    initdb = _find_pg_binary("initdb")
    pg_ctl = _find_pg_binary("pg_ctl")
    if not initdb or not pg_ctl:
        _unavailable("no local initdb/pg_ctl to start an ephemeral cluster")
        return

    with tempfile.TemporaryDirectory(dir="/tmp", prefix="omn17139-") as base:
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
            # Two databases, exactly as every lane runs: the service ledger lives
            # in POSTGRES_DB and the node ledger in NODE_POSTGRES_DB. bootstrap.sql
            # refuses to select a service-owned ledger for the application
            # database, so collapsing them would not exercise the real shape.
            _psql(target, f'CREATE DATABASE "{NODE_DB}"')
            # The migration-gate sentinel the runner flips on success. Created
            # by the compose bootstrap on a real lane; created here so the run
            # reaches its own completion path rather than dying after the work.
            for database in (target.dbname, NODE_DB):
                _psql(target, DB_METADATA_SQL, dbname=database)
            yield target
        finally:
            subprocess.run(
                [pg_ctl, "-D", str(datadir), "-m", "immediate", "-w", "stop"],
                check=False,
                capture_output=True,
            )


def _write_migration(forward: Path, body: str) -> str:
    """Write the node migration and re-declare it in the fixture manifest."""
    node_dir = forward / "nodes" / NODE
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / FILENAME).write_text(body, encoding="utf-8")
    checksum = _sha256(body)
    (forward / "_ledger" / "application-migrations.tsv").write_text(
        "\t".join((ARTIFACT, STREAM, STREAM, DOMAIN, VERSION, checksum)) + "\n",
        encoding="utf-8",
    )
    return checksum


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    forward = tmp_path / "migrations" / "forward"
    ledger = forward / "_ledger"
    ledger.mkdir(parents=True)
    # The real bootstrap: this proof must exercise the shipped canonical-ledger
    # convergence, not a stand-in for it.
    shutil.copy(REAL_LEDGER / "bootstrap.sql", ledger / "bootstrap.sql")
    for name in (
        "application-migration-blocks.tsv",
        "legacy-node-migrations.tsv",
        "verified-checksum-adoptions.tsv",
        "verified-divergent-adoptions.tsv",
        "verified-cross-source-adoptions.tsv",
        "verified-canonical-adoptions.tsv",
        "cloud-migration-aliases.tsv",
    ):
        (ledger / name).write_text("", encoding="utf-8")
    (forward / "fenced-node-migrations.yaml").write_text(
        "fenced_node_migrations: []\n", encoding="utf-8"
    )
    (forward / "grandfathered-force-rls-migrations.yaml").write_text(
        "grandfathered_force_rls_migrations: []\n", encoding="utf-8"
    )
    _write_migration(forward, MIGRATION_V1)
    return forward


def _run(target: PgTarget, migrations: Path) -> subprocess.CompletedProcess[str]:
    psql_dir = str(Path(_find_pg_binary("psql") or "psql").parent)
    env = {
        **os.environ,
        "PATH": f"{psql_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "POSTGRES_USER": target.user,
        "POSTGRES_PASSWORD": target.password,
        "POSTGRES_HOST": target.host,
        "POSTGRES_PORT": str(target.port),
        "POSTGRES_DB": target.dbname,
        "NODE_POSTGRES_DB": NODE_DB,
        "MIGRATIONS_DIR": str(migrations),
        "NODE_MIGRATIONS_DIR": str(migrations / "nodes"),
    }
    return subprocess.run(
        ["/bin/sh", str(RUNNER)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def _write_adoption(
    migrations: Path, *, source_checksum: str, manifest_checksum: str
) -> None:
    (migrations / "_ledger" / "verified-canonical-adoptions.tsv").write_text(
        "\t".join(
            (
                VERSION,
                source_checksum,
                manifest_checksum,
                TICKET,
                RECEIPT_SHA,
                "2026-08-30",
            )
        )
        + "\n",
        encoding="utf-8",
    )


# ruff: noqa: S608 -- VERSION is a literal defined at the top of this file; there
# is no untrusted input anywhere in this module.


def _ledger_row(target: PgTarget) -> tuple[str, str]:
    row = _psql(
        target,
        "SELECT checksum || '|' || provenance FROM platform_catalog.schema_migrations "
        f"WHERE version = '{VERSION}'",
        dbname=NODE_DB,
    )
    checksum, _, provenance = row.partition("|")
    return checksum, provenance


def _apply_first_revision(target: PgTarget, migrations: Path) -> str:
    """Apply revision 1 through the real runner and return its recorded checksum."""
    first = _run(target, migrations)
    assert first.returncode == 0, f"{first.stdout}\n{first.stderr}"
    recorded, provenance = _ledger_row(target)
    assert recorded == _sha256(MIGRATION_V1), (
        "the runner must record the applied revision's own content hash; without "
        f"that this whole class of failure cannot arise: {recorded}"
    )
    assert provenance.startswith("file:"), provenance
    return recorded


def test_inplace_rewrite_without_a_declaration_still_aborts(
    pg_target: PgTarget, migrations_dir: Path
) -> None:
    """RED control: the gate must still fire when nothing has been proven.

    If this passed, the GREEN proof below would be satisfied by a runner that had
    simply stopped comparing checksums -- an adoption path that admits everything
    is not an adoption path.
    """
    _apply_first_revision(pg_target, migrations_dir)
    _write_migration(migrations_dir, MIGRATION_V2)

    second = _run(pg_target, migrations_dir)
    assert second.returncode != 0, (
        "an in-place rewrite of an applied migration must abort the run: "
        f"{second.stdout}\n{second.stderr}"
    )
    assert "conflicting migration checksum" in second.stderr, second.stderr
    recorded, _ = _ledger_row(pg_target)
    assert recorded == _sha256(MIGRATION_V1), (
        "a refused run must not have touched the recorded checksum"
    )


def test_declaration_naming_a_different_revision_is_refused(
    pg_target: PgTarget, migrations_dir: Path
) -> None:
    """A declaration covers ONE revision, not any revision.

    The source checksum below is well-formed and the version is right; only the
    revision it names is wrong. That must not admit the row, or a stale
    declaration would outlive the fact it attested to.
    """
    _apply_first_revision(pg_target, migrations_dir)
    manifest_checksum = _write_migration(migrations_dir, MIGRATION_V2)
    _write_adoption(
        migrations_dir, source_checksum="b" * 64, manifest_checksum=manifest_checksum
    )

    result = _run(pg_target, migrations_dir)
    assert result.returncode != 0, f"{result.stdout}\n{result.stderr}"
    assert "conflicting migration checksum" in result.stderr, result.stderr


def test_declaration_proven_against_other_bytes_is_refused(
    pg_target: PgTarget, migrations_dir: Path
) -> None:
    """Rewriting the file again after the proof re-opens the question."""
    recorded = _apply_first_revision(pg_target, migrations_dir)
    _write_migration(migrations_dir, MIGRATION_V2)
    _write_adoption(
        migrations_dir, source_checksum=recorded, manifest_checksum="c" * 64
    )

    result = _run(pg_target, migrations_dir)
    assert result.returncode != 0, f"{result.stdout}\n{result.stderr}"
    assert "conflicting migration checksum" in result.stderr, result.stderr


def test_declared_revision_is_adopted_and_the_question_survives(
    pg_target: PgTarget, migrations_dir: Path
) -> None:
    """GREEN: a declared, proven revision converges the lane instead of bricking it."""
    recorded = _apply_first_revision(pg_target, migrations_dir)
    manifest_checksum = _write_migration(migrations_dir, MIGRATION_V2)
    _write_adoption(
        migrations_dir, source_checksum=recorded, manifest_checksum=manifest_checksum
    )

    result = _run(pg_target, migrations_dir)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    checksum, provenance = _ledger_row(pg_target)
    assert checksum == manifest_checksum, (
        "adoption records the MANIFEST hash: the proof says the live schema is "
        "what the current bytes produce, so that is the honest content claim"
    )
    # The adoption records the question and its answer; it does not erase the
    # question. Everything needed to re-audit the decision stays on the row.
    assert provenance.startswith("verified-canonical-adoption:"), provenance
    assert f"raw-checksum={recorded}" in provenance, provenance
    assert f"ticket={TICKET}" in provenance, provenance
    assert f"receipt={RECEIPT_SHA}" in provenance, provenance


def test_adoption_is_idempotent(pg_target: PgTarget, migrations_dir: Path) -> None:
    """A second run finds the checksums already equal and adopts nothing again."""
    recorded = _apply_first_revision(pg_target, migrations_dir)
    manifest_checksum = _write_migration(migrations_dir, MIGRATION_V2)
    _write_adoption(
        migrations_dir, source_checksum=recorded, manifest_checksum=manifest_checksum
    )
    assert _run(pg_target, migrations_dir).returncode == 0

    again = _run(pg_target, migrations_dir)
    assert again.returncode == 0, f"{again.stdout}\n{again.stderr}"
    checksum, provenance = _ledger_row(pg_target)
    assert checksum == manifest_checksum
    # The second run finds the checksums already equal, so the adoption UPDATE
    # matches nothing and the provenance written by the FIRST run stands -- with
    # the raw recorded hash still on it. A re-adoption would have overwritten
    # raw-checksum with the already-converged value and quietly destroyed the
    # only record that a question was ever asked.
    assert f"raw-checksum={recorded}" in provenance, provenance
