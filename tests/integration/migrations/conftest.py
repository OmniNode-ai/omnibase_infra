# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Throwaway Postgres cluster shared by the migration live-apply proofs.

Extracted from ``test_094_app_dashboard_role.py`` under OMN-15297, which needed
the same cluster to prove the app_dashboard *grant* chain rather than the role
migration alone. One copy, not two: a second hand-maintained copy of a fixture
whose whole job is to make security proofs honest is exactly how the two copies
drift apart and one of them quietly stops proving anything.

Why an ephemeral cluster and not the shared local Postgres: these tests create,
reshape and drop cluster-wide ROLES and revoke database-level CONNECT. Doing
that against a shared lane would collide with anything else running there, and
doing it against a cloud/RDS instance is not something a test may do at all.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import psycopg2
import pytest

# initdb/pg_ctl/psql are the real production apply path's tools. When they are
# absent the live proofs SKIP rather than silently degrade to string matching —
# a skipped proof is visible, a downgraded one is not.
PG_TOOLS_MISSING = any(
    shutil.which(tool) is None for tool in ("initdb", "pg_ctl", "psql")
)


def _free_port() -> int:
    """Reserve a port number for the unix socket file name.

    ``listen_addresses=''`` below means the cluster never binds TCP, so this
    number only ever names the socket file. It is still probed rather than
    hardcoded so two modules (or two pytest-xdist workers) cannot collide on
    the same socket path.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class EphemeralPostgres:
    """A throwaway, superuser-owned Postgres cluster for one test."""

    def __init__(self, socket_dir: str, port: int) -> None:
        self.socket_dir = socket_dir
        self.port = port

    def connect(
        self,
        *,
        user: str = "postgres",
        password: str | None = None,
        dbname: str = "postgres",
    ) -> psycopg2.extensions.connection:
        return psycopg2.connect(
            host=self.socket_dir,
            port=self.port,
            user=user,
            password=password,
            dbname=dbname,
        )

    def psql(
        self,
        *args: str,
        user: str = "postgres",
        dbname: str = "postgres",
    ) -> subprocess.CompletedProcess[str]:
        """Apply SQL the same way the real migration runner does.

        ``run-forward-migrations.sh`` invokes each file as
        ``psql -v ON_ERROR_STOP=1 -f <file>`` — matching that invocation
        (rather than executing the SQL text through a driver call) is what
        makes this an honest reproduction of the production apply path, and is
        what makes ``\\connect`` directives inside a migration behave the way
        they do in production.
        """
        return subprocess.run(
            [
                "psql",
                "-h",
                self.socket_dir,
                "-p",
                str(self.port),
                "-U",
                user,
                "-d",
                dbname,
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
        )


@pytest.fixture
def ephemeral_postgres() -> Iterator[EphemeralPostgres]:
    if PG_TOOLS_MISSING:
        pytest.skip(
            "initdb/pg_ctl/psql not on PATH — cannot spin up an ephemeral "
            "Postgres cluster for the live-apply proof"
        )

    scratch = tempfile.mkdtemp(prefix="onexpg_")
    data_dir = Path(scratch) / "data"
    log_file = Path(scratch) / "server.log"
    port = _free_port()

    init = subprocess.run(
        ["initdb", "-D", str(data_dir), "-U", "postgres", "--auth=trust", "--no-sync"],
        capture_output=True,
        text=True,
        check=False,
    )
    if init.returncode != 0:
        shutil.rmtree(scratch, ignore_errors=True)
        pytest.fail(f"initdb failed for the ephemeral test cluster: {init.stderr}")

    start = subprocess.run(
        [
            "pg_ctl",
            "-D",
            str(data_dir),
            "-o",
            f"-k {scratch} -p {port} -c listen_addresses=",
            "-l",
            str(log_file),
            "-w",
            "-t",
            "30",
            "start",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if start.returncode != 0:
        log_text = log_file.read_text() if log_file.exists() else ""
        shutil.rmtree(scratch, ignore_errors=True)
        pytest.fail(
            f"pg_ctl start failed for the ephemeral test cluster: "
            f"{start.stderr}\n{log_text}"
        )

    try:
        yield EphemeralPostgres(socket_dir=scratch, port=port)
    finally:
        subprocess.run(
            ["pg_ctl", "-D", str(data_dir), "-m", "fast", "stop"],
            capture_output=True,
            text=True,
            check=False,
        )
        shutil.rmtree(scratch, ignore_errors=True)
