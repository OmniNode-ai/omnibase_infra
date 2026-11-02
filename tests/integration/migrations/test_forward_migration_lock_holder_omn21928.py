# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The forward-migration advisory-lock holder is checked while the run goes, not only at its end.

Measured on the first slot bring-up of 2026-11-02 (OMN-21928): the runner applied
every migration, then reported ``advisory lock 100010 was NOT held for the whole
run -- the holder session died mid-run`` and the lane refused with exit 9. The
check ran once, as the last act, about seven minutes after the lock was taken,
so the run neither said when the holder went nor stopped applying files while it
was gone.

Two behaviours are pinned against a real PostgreSQL 16 cluster and the real
runner (no stand-in for the database, no stand-in for the runner):

* a holder session killed mid-run stops the run at the next file, names the file
  it would have applied, and applies nothing after it;
* a session of ANOTHER database that holds the same advisory lock id under the
  same application name (two runners in two containers both have ``$$`` = 1, so
  both tag themselves ``forward-migration-lock-100010-1``) is not counted as a
  second holder of this run's lock.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.integration.migrations.test_application_migration_ledger_omn15413 import (
    RUNNER,
    Pg16Cluster,
    _synthetic_migration_tree,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

LOCK_ID = 100010
SLOW_SECONDS = 12


def _scalar(
    pg16: Pg16Cluster, sql: str, database: str = "postgres", **variables: str
) -> str:
    """One value from ``sql``; variables reach it as psql ``:'name'`` literals, never by interpolation."""
    arguments = ["-At"]
    for name, value in variables.items():
        arguments += ["-v", f"{name}={value}"]
    return pg16.command(database, *arguments, input_text=sql).stdout.strip()


def _tree(tmp_path: Path) -> Path:
    migrations_dir, _version, _checksum = _synthetic_migration_tree(tmp_path)
    (migrations_dir / "001_slow.sql").write_text(
        f"SELECT pg_sleep({SLOW_SECONDS});\n", encoding="utf-8"
    )
    (migrations_dir / "002_after.sql").write_text(
        "CREATE TABLE public.omn21928_after (id INTEGER PRIMARY KEY);\n",
        encoding="utf-8",
    )
    return migrations_dir


def _start_runner(
    pg16: Pg16Cluster, migrations_dir: Path, databases: tuple[str, str, str], out: Path
) -> subprocess.Popen[str]:
    service_database, application_database, cloud_database = databases
    for database in databases:
        pg16.create_database(database)
    with out.open("w", encoding="utf-8") as sink:
        return subprocess.Popen(
            ["sh", str(RUNNER)],
            stdout=sink,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                "PATH": (
                    f"{pg16.bin_dir}:/opt/homebrew/bin:/usr/local/bin:"
                    "/usr/bin:/bin:/usr/sbin:/sbin"
                ),
                "POSTGRES_HOST": "127.0.0.1",
                "POSTGRES_PORT": str(pg16.port),
                "POSTGRES_USER": "postgres",
                "POSTGRES_PASSWORD": "",
                "POSTGRES_DB": service_database,
                "NODE_POSTGRES_DB": application_database,
                "OMNINODE_CLOUD_HISTORY_DB": cloud_database,
                "MIGRATIONS_DIR": str(migrations_dir),
                "PG_WAIT_RETRIES": "5",
                "MIGRATION_LOCK_WAIT_SECONDS": "10",
            },
        )


def _wait_for(out: Path, needle: str, proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if needle in out.read_text(encoding="utf-8"):
            return
        assert proc.poll() is None, (
            f"runner exited before {needle!r}:\n{out.read_text()}"
        )
        time.sleep(0.2)
    raise AssertionError(f"no {needle!r} in 60s:\n{out.read_text()}")


@pytest.fixture
def reaper() -> Iterator[list[subprocess.Popen[Any]]]:
    procs: list[subprocess.Popen[Any]] = []
    yield procs
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_a_holder_killed_mid_run_stops_the_run_at_the_next_file(
    pg16: Pg16Cluster, tmp_path: Path, reaper: list[subprocess.Popen[Any]]
) -> None:
    databases = ("omn21928_kill_service", "omn21928_kill_app", "omn21928_kill_cloud")
    out = tmp_path / "runner.out"
    proc = _start_runner(pg16, _tree(tmp_path), databases, out)
    reaper.append(proc)
    _wait_for(out, "apply 001_slow.sql", proc)

    killed = _scalar(
        pg16,
        "SELECT count(pg_terminate_backend(pid)) FROM pg_stat_activity "
        "WHERE datname = :'db' AND application_name LIKE :'tag';",
        db=databases[0],
        tag=f"forward-migration-lock-{LOCK_ID}-%",
    )
    assert killed == "1", "positive control: exactly one holder session was killed"

    assert proc.wait(timeout=90) != 0
    text = out.read_text(encoding="utf-8")
    assert f"advisory lock {LOCK_ID} was lost before applying 002_after.sql" in text
    assert "apply 002_after.sql" not in text
    assert (
        _scalar(pg16, "SELECT to_regclass('public.omn21928_after');", databases[0])
        == ""
    )


def test_an_unharmed_run_applies_every_file_and_sets_the_sentinel(
    pg16: Pg16Cluster, tmp_path: Path, reaper: list[subprocess.Popen[Any]]
) -> None:
    """Positive control for the pins around it: the same tree, nobody touches the holder."""
    databases = ("omn21928_ok_service", "omn21928_ok_app", "omn21928_ok_cloud")
    out = tmp_path / "runner.out"
    proc = _start_runner(pg16, _tree(tmp_path), databases, out)
    reaper.append(proc)

    assert proc.wait(timeout=90) == 0, out.read_text(encoding="utf-8")
    text = out.read_text(encoding="utf-8")
    assert "apply 002_after.sql" in text
    assert "Sentinel set. Migration gate will report HEALTHY." in text


def test_a_same_named_lock_holder_in_another_database_is_not_a_second_holder(
    pg16: Pg16Cluster, tmp_path: Path, reaper: list[subprocess.Popen[Any]]
) -> None:
    databases = ("omn21928_dup_service", "omn21928_dup_app", "omn21928_dup_cloud")
    foreign_database = "omn21928_dup_foreign"
    pg16.create_database(foreign_database)
    out = tmp_path / "runner.out"
    proc = _start_runner(pg16, _tree(tmp_path), databases, out)
    reaper.append(proc)
    _wait_for(out, "apply 001_slow.sql", proc)

    # What a second runner in another container looks like to the server: the
    # same lock id, in its own database, under the tag `$$` = 1 gives both. The
    # foreign session borrows this runner's tag, which is `$$`-based.
    foreign = subprocess.Popen(
        [
            pg16.psql,
            "-X",
            "-q",
            "-h",
            "127.0.0.1",
            "-p",
            str(pg16.port),
            "-U",
            "postgres",
            "-d",
            foreign_database,
            "-c",
            f"SELECT pg_advisory_lock({LOCK_ID}), pg_sleep(120)",
        ],
        env={
            **os.environ,
            "PGPASSWORD": "",
            "PGAPPNAME": f"forward-migration-lock-{LOCK_ID}-{proc.pid}",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    reaper.append(foreign)
    deadline = time.monotonic() + 15
    while True:
        visible = _scalar(
            pg16,
            "SELECT count(*) FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid "
            "WHERE l.locktype = 'advisory' AND l.granted "
            "AND a.application_name = :'tag';",
            tag=f"forward-migration-lock-{LOCK_ID}-{proc.pid}",
        )
        if visible == "2" or time.monotonic() > deadline:
            break
        time.sleep(0.2)
    assert visible == "2", "positive control: the server sees two granted holders"

    assert proc.wait(timeout=90) == 0, out.read_text(encoding="utf-8")
    assert "Sentinel set. Migration gate will report HEALTHY." in out.read_text(
        encoding="utf-8"
    )


def test_a_foreign_advisory_lock_with_a_wide_key_does_not_fail_the_run(
    pg16: Pg16Cluster, tmp_path: Path, reaper: list[subprocess.Popen[Any]]
) -> None:
    """Measured on slot 1 (OMN-21928): the probe raised ``bigint out of range``.

    ``classid * 4294967296`` was evaluated for every advisory lock on the server,
    and another service's lock with a key whose high word is 2**31 or more (any
    negative bigint key) overflows it, so the probe read "not held" while the
    lock was held. ``pg_advisory_lock(-1)`` is such a key.
    """
    databases = ("omn21928_wide_service", "omn21928_wide_app", "omn21928_wide_cloud")
    foreign_database = "omn21928_wide_foreign"
    pg16.create_database(foreign_database)
    foreign = subprocess.Popen(
        [
            pg16.psql,
            "-X",
            "-q",
            "-h",
            "127.0.0.1",
            "-p",
            str(pg16.port),
            "-U",
            "postgres",
            "-d",
            foreign_database,
            "-c",
            "SELECT pg_advisory_lock(-1), pg_sleep(120)",
        ],
        env={**os.environ, "PGPASSWORD": ""},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    reaper.append(foreign)
    deadline = time.monotonic() + 15
    while True:
        wide = _scalar(
            pg16,
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
            "AND classid >= 2147483648;",
        )
        if wide == "1" or time.monotonic() > deadline:
            break
        time.sleep(0.2)
    assert wide == "1", "positive control: the server holds one wide-key lock"

    out = tmp_path / "runner.out"
    proc = _start_runner(pg16, _tree(tmp_path), databases, out)
    reaper.append(proc)

    assert proc.wait(timeout=90) == 0, out.read_text(encoding="utf-8")
    assert "Sentinel set. Migration gate will report HEALTHY." in out.read_text(
        encoding="utf-8"
    )
