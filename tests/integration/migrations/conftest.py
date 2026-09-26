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

import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
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

#: ``psql`` is the one tool no backend can substitute: every live-apply proof
#: drives the migration through ``psql -v ON_ERROR_STOP=1 -f <file>`` because
#: that is the invocation ``run-forward-migrations.sh`` uses in production.
PSQL_MISSING = shutil.which("psql") is None

#: Same major version as the ``services.postgres`` image the CI jobs provision,
#: so the container backend and the hosted jobs agree on server behaviour.
_DOCKER_IMAGE = os.environ.get("ONEX_EPHEMERAL_PG_IMAGE", "postgres:16-alpine")

#: Set to "1" to turn a SKIP into a hard failure. With a working backend in
#: hand a skipped security proof is a vacuous green, and this suite exists
#: precisely because enforcement once shipped ahead of provisioning unnoticed.
_REQUIRE_ENV = "ONEX_MIGRATION_PROOF_REQUIRE_PG"


def _docker_usable() -> bool:
    """Whether a throwaway Postgres CONTAINER can stand in for ``initdb``.

    Checked by running ``docker info`` rather than by the binary's presence: a
    Docker CLI with no reachable daemon is the ordinary state on a runner, and
    it answers ``which`` exactly as well as a working one.
    """
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


#: A cluster can be built natively OR in a container. Modules that gate on
#: availability must read THIS, not PG_TOOLS_MISSING: the native tools being
#: absent stopped meaning "no cluster is possible" once the container backend
#: existed, and a module that still gates on the old name skips a proof it
#: could have run.
EPHEMERAL_POSTGRES_UNAVAILABLE = PSQL_MISSING or (
    PG_TOOLS_MISSING and not _docker_usable()
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


def _unavailable(reason: str) -> None:
    """Skip, or fail when the caller declared a backend must be present."""
    if os.environ.get(_REQUIRE_ENV) == "1":
        pytest.fail(f"{_REQUIRE_ENV}=1 but no Postgres backend is usable: {reason}")
    pytest.skip(reason)


def _docker_cluster() -> Iterator[EphemeralPostgres]:
    """A throwaway cluster in a container, for hosts where ``initdb`` cannot run.

    Two hosts need this and neither is exotic. A CI container installs
    ``postgresql-client`` (psql) but not the server package, so ``initdb`` is
    absent; and on macOS the SysV shared-memory ceiling (``kern.sysv.shmall``
    defaults to 1024 pages = 4 MiB, shared with every other process) makes
    ``initdb`` fail at ``shmget`` even with the binaries installed. Both
    previously SKIPPED every live-apply proof in this directory, which is how
    eight security proofs for OMN-15425 came to execute nowhere at all while
    the suite still reported green.

    One container per test, torn down after, so the per-test virgin-cluster
    property the native backend provides is preserved exactly. That property is
    load-bearing here: these proofs create, escalate and drop CLUSTER-WIDE
    roles and revoke database-level CONNECT, so a reused cluster would let one
    test's role state decide another test's verdict.

    ``POSTGRES_HOST_AUTH_METHOD=trust`` mirrors the native backend's
    ``initdb --auth=trust``, so no password has to be threaded through
    ``psql()`` and the two backends stay behaviourally identical.
    """
    name = f"onexpg-{uuid.uuid4().hex[:12]}"
    port = _free_port()
    start = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--detach",
            "--name",
            name,
            "--env",
            "POSTGRES_HOST_AUTH_METHOD=trust",
            "--env",
            "POSTGRES_PASSWORD=",
            "--publish",
            f"127.0.0.1:{port}:5432",
            _DOCKER_IMAGE,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if start.returncode != 0:
        _unavailable(
            f"could not start the {_DOCKER_IMAGE} container for the "
            f"live-apply proof: {start.stderr.strip()}"
        )

    try:
        deadline = time.monotonic() + 60.0
        last = ""
        while time.monotonic() < deadline:
            # pg_isready INSIDE the container answers for the server; the
            # connect below answers for the published port. Both are needed:
            # the server accepts connections several hundred ms before the
            # host-side mapping is reliably reachable.
            ready = subprocess.run(
                ["docker", "exec", name, "pg_isready", "-U", "postgres", "-q"],
                capture_output=True,
                text=True,
                check=False,
            )
            if ready.returncode == 0:
                try:
                    psycopg2.connect(
                        host="127.0.0.1", port=port, user="postgres", dbname="postgres"
                    ).close()
                    break
                except psycopg2.Error as exc:  # pragma: no cover - timing only
                    last = str(exc)
            time.sleep(0.25)
        else:
            pytest.fail(
                f"the {_DOCKER_IMAGE} container never became reachable on "
                f"127.0.0.1:{port} within 60s: {last}"
            )
        yield EphemeralPostgres(socket_dir="127.0.0.1", port=port)
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name],
            capture_output=True,
            text=True,
            check=False,
        )


@pytest.fixture
def ephemeral_postgres() -> Iterator[EphemeralPostgres]:
    if PSQL_MISSING:
        _unavailable(
            "psql not on PATH — the live-apply proofs drive migrations through "
            "the same psql invocation production uses and cannot substitute a "
            "driver call for it"
        )
    if PG_TOOLS_MISSING:
        yield from _docker_cluster()
        return

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
        # Present-but-unusable is a real host state, not a defect in the proof:
        # macOS refuses the SysV segment initdb asks for once kern.sysv.shmall
        # (4 MiB by default, shared machine-wide) is consumed. Falling through
        # to the container backend keeps the proof running instead of turning a
        # host limit into a red suite.
        if _docker_usable():
            yield from _docker_cluster()
            return
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


# ---------------------------------------------------------------------------
# OMN-15857: shared PostgreSQL 16 cluster for the migration-ledger proofs
# ---------------------------------------------------------------------------
#
# ``test_application_migration_ledger_omn15413.py`` owns that harness. The
# OMN-15857 files reuse it, and re-exporting the fixture here rather than
# importing it into each module keeps the name out of those modules' namespaces,
# where it would shadow every ``pg16`` test parameter.
from tests.integration.migrations.test_application_migration_ledger_omn15413 import (
    pg16,
)
