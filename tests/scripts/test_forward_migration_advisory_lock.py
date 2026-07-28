# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15291 — the forward-migration runner must serialize concurrent runs.

Defect (``docs/deep-dives/JULY_27_2026_DEEP_DIVE.md``, "Remaining friction"):
``scripts/run-forward-migrations.sh`` applied every migration through an
unsynchronized check-then-act — ``SELECT`` from ``schema_migrations``, then
``psql -f``, then ``INSERT ... ON CONFLICT DO NOTHING`` — with **no advisory
lock of any kind**.  Two concurrent runners both read "not applied" and both
executed the same file; non-idempotent DDL then errored in the loser, and the
``ON CONFLICT`` hid the double-apply so the tracking table still looked clean.

OMN-15254 fixed the sibling defect in ``omninode_infra``'s k8s Job runners.
This is the same canonical single-session lock ported to this runner in POSIX
``sh``.

These tests drive **the artifact that actually runs** — the shipped
``scripts/run-forward-migrations.sh``, executed twice concurrently against one
real Postgres with a deliberately slow, non-idempotent migration.
``test_lock_free_runner_does_not_serialize`` is the RED control: the identical
harness against a copy of the runner with the lock block mechanically stripped,
asserting the critical sections DO overlap.  Without that control the
serialization assertion would be unfalsifiable — it would also pass against a
harness that never runs anything concurrently.

Database selection, in order:
  1. ``MIGRATION_LOCK_TEST_HOST``/``_PORT``/``_USER``/``_PASSWORD``/``_DB``.
  2. An ephemeral local cluster via ``initdb``/``pg_ctl`` if those binaries exist.
  3. Skip — unless ``REQUIRE_MIGRATION_LOCK_DB`` is set, in which case fail.
     A check that silently skips is a check that does not exist.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"

BEGIN_MARKER = "# ---- BEGIN canonical forward-migration advisory lock (OMN-15291) ----"
END_MARKER = "# ---- END canonical forward-migration advisory lock (OMN-15291) ----"

# The migration applied by the live proofs. The 3s pg_sleep is the critical
# section: long enough that a genuinely concurrent second runner is
# observably inside it too. It is deliberately NOT idempotent:
# a second application raises "relation already exists", which is exactly the
# production failure the lock prevents. The probe rows bracket the window in
# which a runner is inside the apply.
RACE_MIGRATION_SQL = """\
INSERT INTO public.apply_probe (label, phase)
  VALUES (current_setting('application_name'), 'start');
SELECT pg_sleep(3);
CREATE TABLE public.t_race (id INT PRIMARY KEY);
INSERT INTO public.apply_probe (label, phase)
  VALUES (current_setting('application_name'), 'end');
"""

SETUP_SQL = """\
CREATE TABLE public.db_metadata (
  id BOOLEAN PRIMARY KEY DEFAULT TRUE,
  migrations_complete BOOLEAN NOT NULL DEFAULT FALSE,
  runner_completed_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO public.db_metadata (id) VALUES (TRUE);
CREATE TABLE public.apply_probe (
  label TEXT NOT NULL,
  phase TEXT NOT NULL,
  at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
-- Pre-created so the runner's own `CREATE TABLE IF NOT EXISTS` bootstrap is a
-- no-op. Two lock-free runners executing that statement concurrently lose to a
-- duplicate-key error on pg_type_typname_nsp_index before either reaches the
-- apply loop -- a real symptom of the same defect, but it would pre-empt the
-- apply race these proofs are about and make the RED control host-load
-- dependent.
CREATE TABLE public.schema_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    checksum     TEXT NOT NULL,
    source_set   TEXT NOT NULL
);
"""


def _runner_text() -> str:
    return RUNNER.read_text()


def extract_lock_block(text: str | None = None) -> str:
    """Return the canonical lock block, markers stripped.

    Raises with a specific message if the markers are missing or duplicated —
    that is how a silent removal of the lock surfaces as a test failure rather
    than as a vacuous pass.
    """
    lines = (text if text is not None else _runner_text()).splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == BEGIN_MARKER]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == END_MARKER]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        msg = (
            "scripts/run-forward-migrations.sh: expected exactly one canonical "
            "advisory-lock block delimited by the OMN-15291 markers (found "
            f"{len(starts)} begin / {len(ends)} end markers)"
        )
        raise AssertionError(msg)
    return "\n".join(lines[starts[0] + 1 : ends[0]]) + "\n"


def strip_lock_block(text: str) -> str:
    """The pre-OMN-15291 runner: byte-identical minus the lock block.

    Derived from the shipped artifact rather than pinned as a copy, so the RED
    control can never drift away from the thing it is the control for.
    """
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.strip() == BEGIN_MARKER)
    end = next(i for i, ln in enumerate(lines) if ln.strip() == END_MARKER)
    remaining = lines[:start] + lines[end + 1 :]
    # The lock-free runner has no lock to assert on either.
    return "".join(
        ln for ln in remaining if ln.strip() != "assert_migration_lock_still_held"
    )


# --------------------------------------------------------------------------
# Static assertions — no database required, so they gate every PR.
# --------------------------------------------------------------------------


def test_runner_carries_the_canonical_lock_block() -> None:
    block = extract_lock_block()
    assert "pg_advisory_lock(" in block, (
        "the OMN-15291 block must actually acquire an advisory lock"
    )


def test_runner_never_uses_a_cross_session_unlock() -> None:
    """The original defect's signature: an unlock from a session that never held it."""
    offenders = [
        ln
        for ln in _runner_text().splitlines()
        if "pg_advisory_unlock" in ln and not ln.strip().startswith("#")
    ]
    assert not offenders, (
        "pg_advisory_unlock() releases only locks held by the CALLING session; "
        "this runner holds its lock in a dedicated session that is released by "
        f"disconnecting (OMN-15291). Offending lines: {offenders}"
    )


def test_lock_is_never_acquired_through_a_one_shot_psql() -> None:
    """A ``psql -c`` acquisition drops the lock as soon as that psql exits."""
    for line in _runner_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "pg_advisory_lock" not in stripped:
            continue
        assert not stripped.startswith("psql "), (
            f"`{stripped}` acquires the advisory lock in a one-shot psql session "
            "that exits immediately, releasing it (OMN-15291)"
        )


def test_runner_asserts_the_lock_survived_the_whole_run() -> None:
    calls = [
        ln
        for ln in _runner_text().splitlines()
        if ln.strip() == "assert_migration_lock_still_held"
    ]
    assert len(calls) == 1, (
        "expected exactly one final `assert_migration_lock_still_held` call so a "
        "holder session that died mid-run fails the runner instead of flipping "
        f"the migration-gate sentinel HEALTHY (found {len(calls)})"
    )


def test_held_ness_check_is_bound_to_our_own_session() -> None:
    """ "Somebody holds it" is not proof that WE hold it."""
    block = extract_lock_block()
    assert "application_name = '${MIGRATION_LOCK_TAG}'" in block, (
        "the held-ness probe must match our own application_name tag, otherwise "
        "another runner's lock reads as ours (OMN-15291)"
    )


def test_acquisition_failure_is_fatal_not_advisory() -> None:
    block = extract_lock_block()
    assert block.count("exit 1") >= 2, (
        "both acquisition failure paths (holder died, deadline exceeded) must "
        "exit nonzero — proceeding unserialized is the defect"
    )


def test_runner_is_posix_sh() -> None:
    """The container shell is busybox ash; bashisms would fail only in prod."""
    assert _runner_text().splitlines()[0] == "#!/bin/sh"
    block = extract_lock_block()
    assert not re.search(r"^\s*local\s", block, re.MULTILINE), (
        "`local` is not POSIX and is unavailable in every /bin/sh this runner "
        "may execute under"
    )


# --------------------------------------------------------------------------
# Live concurrency proof.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PgTarget:
    """Connection coordinates for the scratch Postgres under test."""

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
    for pattern in ("/opt/homebrew/opt/postgresql@*/bin", "/usr/lib/postgresql/*/bin"):
        root = Path(pattern).parent
        if not root.exists():
            continue
        for candidate in sorted(root.glob(Path(pattern).name)):
            binary = candidate / name
            if binary.exists():
                return str(binary)
    return None


def _unavailable(reason: str) -> None:
    if os.environ.get("REQUIRE_MIGRATION_LOCK_DB"):
        pytest.fail(
            "REQUIRE_MIGRATION_LOCK_DB is set but the advisory-lock concurrency "
            f"proof cannot run: {reason}"
        )
    pytest.skip(reason)


def _psql(target: PgTarget, sql: str) -> str:
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
            target.dbname,
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
    """A scratch Postgres: external if configured, else an ephemeral cluster.

    Function-scoped on purpose — each proof needs a virgin database (the racing
    migration creates a table that must not already exist).
    """
    if not _find_pg_binary("psql"):
        _unavailable("psql client not available")

    host = os.environ.get("MIGRATION_LOCK_TEST_HOST")
    if host:
        target = PgTarget(
            host=host,
            port=int(os.environ.get("MIGRATION_LOCK_TEST_PORT", "5432")),
            user=os.environ.get("MIGRATION_LOCK_TEST_USER", "postgres"),
            password=os.environ.get("MIGRATION_LOCK_TEST_PASSWORD", "postgres"),
            dbname=os.environ.get("MIGRATION_LOCK_TEST_DB", "postgres"),
        )
        scratch = f"omn15291_{int(time.time() * 1000) % 100_000_000}"
        _psql(target, f'CREATE DATABASE "{scratch}"')
        scoped = PgTarget(
            host=target.host,
            port=target.port,
            user=target.user,
            password=target.password,
            dbname=scratch,
        )
        try:
            _psql(scoped, SETUP_SQL)
            yield scoped
        finally:
            _psql(target, f'DROP DATABASE IF EXISTS "{scratch}" WITH (FORCE)')
        return

    initdb = _find_pg_binary("initdb")
    pg_ctl = _find_pg_binary("pg_ctl")
    if not initdb or not pg_ctl:
        _unavailable(
            "no MIGRATION_LOCK_TEST_HOST and no local initdb/pg_ctl to start an "
            "ephemeral cluster"
        )
        return

    # Short base dir: the unix socket path has a ~100 char limit.
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="omn15291-") as base:
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
            _psql(target, SETUP_SQL)
            yield target
        finally:
            subprocess.run(
                [pg_ctl, "-D", str(datadir), "-m", "immediate", "-w", "stop"],
                check=False,
                capture_output=True,
            )


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    forward = tmp_path / "migrations" / "forward"
    forward.mkdir(parents=True)
    (forward / "001_race_target.sql").write_text(RACE_MIGRATION_SQL)
    return forward


def _runner_env(
    target: PgTarget, migrations: Path, label: str, wait: int
) -> dict[str, str]:
    psql_dir = str(Path(_find_pg_binary("psql") or "psql").parent)
    return {
        **os.environ,
        "PATH": f"{psql_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "PGAPPNAME": label,
        "POSTGRES_USER": target.user,
        "POSTGRES_PASSWORD": target.password,
        "POSTGRES_HOST": target.host,
        "POSTGRES_PORT": str(target.port),
        "POSTGRES_DB": target.dbname,
        "MIGRATIONS_DIR": str(migrations),
        "NODE_MIGRATIONS_DIR": str(migrations / "nodes"),
        "MIGRATION_LOCK_WAIT_SECONDS": str(wait),
    }


def _spawn(
    runner: Path, target: PgTarget, migrations: Path, label: str, wait: int = 60
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["/bin/sh", str(runner)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_runner_env(target, migrations, label, wait),
    )


def _intervals(target: PgTarget) -> dict[str, tuple[float, float | None]]:
    rows = _psql(
        target,
        "SELECT label, phase, extract(epoch from at) FROM public.apply_probe "
        "ORDER BY at",
    )
    start: dict[str, float] = {}
    end: dict[str, float] = {}
    for line in filter(None, rows.splitlines()):
        label, phase, at = line.split("|")
        (start if phase == "start" else end)[label] = float(at)
    return {label: (at, end.get(label)) for label, at in start.items()}


def _overlaps(intervals: dict[str, tuple[float, float | None]]) -> bool:
    labels = sorted(intervals)
    for i, first in enumerate(labels):
        for second in labels[i + 1 :]:
            a_start, a_end = intervals[first]
            b_start, b_end = intervals[second]
            if a_start < (b_end if b_end is not None else float("inf")) and b_start < (
                a_end if a_end is not None else float("inf")
            ):
                return True
    return False


@pytest.mark.integration
def test_lock_free_runner_does_not_serialize(
    pg_target: PgTarget, migrations_dir: Path, tmp_path: Path
) -> None:
    """RED control: strip the lock block and concurrent runs stop being safe.

    The safe signature the fixed runner produces is exact and narrow: both runs
    exit 0, one applies, one observes the migration as already applied, and the
    critical sections never overlap. This asserts the lock-free runner does NOT
    produce it. Asserting only "the sections overlap" would be host-load
    dependent -- which unserialized step the loser dies on varies -- and a RED
    control that flakes is worse than none.
    """
    legacy = tmp_path / "run-forward-migrations.legacy.sh"
    legacy.write_text(strip_lock_block(_runner_text()))

    procs = [_spawn(legacy, pg_target, migrations_dir, label) for label in ("a", "b")]
    outputs = [proc.communicate(timeout=120) for proc in procs]
    codes = [proc.returncode for proc in procs]
    intervals = _intervals(pg_target)

    overlapped = _overlaps(intervals)
    failed = any(code != 0 for code in codes)
    assert overlapped or failed, (
        "the lock-free runner was expected to be unsafe under concurrency (the "
        f"OMN-15291 defect) but both runs completed cleanly and serialized: "
        f"intervals={intervals} codes={codes}\n{outputs}"
    )

    applied = [out for out, _ in outputs if "apply 001_race_target.sql" in out]
    skipped = [out for out, _ in outputs if "already applied" in out]
    safe_signature = (
        codes == [0, 0] and len(applied) == 1 and len(skipped) == 1 and not overlapped
    )
    assert not safe_signature, (
        "the lock-free runner produced the SAME safe outcome as the locked one, "
        "so the GREEN proof below cannot distinguish them and is vacuous: "
        f"intervals={intervals} codes={codes}"
    )


@pytest.mark.integration
def test_shipped_runner_serializes_concurrent_runs(
    pg_target: PgTarget, migrations_dir: Path
) -> None:
    """GREEN: the shipped runner admits exactly one run at a time."""
    procs = [_spawn(RUNNER, pg_target, migrations_dir, label) for label in ("a", "b")]
    outputs = [proc.communicate(timeout=180) for proc in procs]

    for proc, (out, _) in zip(procs, outputs, strict=True):
        assert proc.returncode == 0, out

    intervals = _intervals(pg_target)
    assert not _overlaps(intervals), (
        "concurrent forward-migration runners entered the apply at the same time "
        f"— the advisory lock is not serializing: {intervals}"
    )
    # Exactly one runner applied it; the other observed it as already applied.
    assert len(intervals) == 1, (
        f"expected one applying runner, got {sorted(intervals)} — the migration "
        "was applied twice"
    )
    applied = [out for out, _ in outputs if "apply 001_race_target.sql" in out]
    skipped = [out for out, _ in outputs if "already applied" in out]
    assert len(applied) == 1 and len(skipped) == 1, (
        f"expected one apply and one skip across the two runs: {outputs}"
    )


@pytest.mark.integration
def test_contended_lock_fails_loud_instead_of_proceeding(
    pg_target: PgTarget, migrations_dir: Path
) -> None:
    """A lock we cannot get must abort with a message, not hang and not proceed."""
    lock_id = os.environ.get("FORWARD_MIGRATION_LOCK_ID", "100010")
    holder = subprocess.Popen(
        [
            _find_pg_binary("psql") or "psql",
            "-h",
            pg_target.host,
            "-p",
            str(pg_target.port),
            "-U",
            pg_target.user,
            "-d",
            pg_target.dbname,
            "-v",
            "ON_ERROR_STOP=1",
            "-q",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=pg_target.env(),
    )
    assert holder.stdin is not None
    holder.stdin.write(f"SELECT pg_advisory_lock({lock_id});\n")
    holder.stdin.flush()
    # Wait until the foreign holder is actually granted the lock.
    deadline = time.time() + 30
    while time.time() < deadline:
        granted = _psql(
            pg_target,
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND granted "  # noqa: S608 - lock_id is a numeric runner constant, not user input
            f"AND classid::bigint * 4294967296 + objid::bigint = {lock_id}",
        )
        if granted != "0":
            break
        time.sleep(0.2)
    else:  # pragma: no cover - environment failure, not a product defect
        holder.kill()
        pytest.fail("could not establish the contending advisory-lock holder")

    try:
        proc = _spawn(RUNNER, pg_target, migrations_dir, "waiter", wait=2)
        out, _ = proc.communicate(timeout=120)
    finally:
        holder.stdin.close()
        holder.kill()
        holder.wait(timeout=30)

    assert proc.returncode != 0, (
        f"a runner that could not acquire the lock exited 0 and proceeded: {out}"
    )
    assert "could not acquire advisory lock" in out, (
        f"expected a specific fail-loud message naming the lock: {out}"
    )
    assert "apply 001_race_target.sql" not in out, (
        f"the runner applied migrations without holding the lock: {out}"
    )
