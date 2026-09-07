# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18012 -- a hermetic Redpanda broker that REQUIRES authentication.

Why this exists
---------------
Every local proof lane in this org runs a no-auth broker: the dev lane on
.201, ``docker/docker-compose.e2e.yml`` here, and every unit test's mock.
Broker authentication is therefore a boundary that exists only in staging and
prod, and a client that silently opens PLAINTEXT against an auth-required
listener passes every local check and fails in front of a customer. That is
escape 1 of the 2026-09-06 set (omnimarket ``_build_event_bus`` without
``apply_environment_overrides`` -> PLAINTEXT to the IAM-only MSK listener ->
every BYOK registration 503).

RESIDUAL, stated plainly and repeated in every module that uses this harness:
staging and prod are MSK with ``AWS_MSK_IAM`` + ``SASL_SSL``. This harness is
SCRAM-SHA-256 over ``SASL_PLAINTEXT``. What it proves is

    "this client did not silently open PLAINTEXT against an auth-required
    listener"

and NOT "this client speaks IAM". The mechanism differs; the *failure mode*
being pinned -- a client constructed with no credentials at all -- does not.

RUNNER TOPOLOGY (OMN-18012, 2026-09-07)
--------------------------------------
The broker port is published into the **Docker host's** network namespace.
On a GitHub-hosted runner the test process is on that host and ``localhost``
is correct. On this org's self-hosted ``omnibase-ci`` runners it is not: those
runners are containers that reach the daemon through a mounted
``/var/run/docker.sock`` (``docker/docker-compose.runners.yml``, no
``networks:`` block, no ``network_mode: host``), so the published port is
ECONNREFUSED at ``localhost`` and reachable only at the Docker host address --
the container's own default route. 7/7 fleet runs of the required job failed
that way while 6/6 hosted runs passed. Same precedent, same fix as OMN-15567
in ``.github/workflows/reusable-runtime-boot.yml``.

Two consequences are baked in below and must stay:

* the resolved address is used for BOTH ``--advertise-kafka-addr`` and
  ``bootstrap``. Advertised metadata redirects every client, so fixing only
  the bootstrap string sends the client straight back to ``localhost``.
* readiness is proven on the CLIENT path (a TCP connect from THIS process),
  not only by ``rpk`` inside the broker container. The in-container probe
  answers from the one namespace where ``localhost`` is always right, which is
  why an unreachable broker previously reported ready and surfaced as an
  opaque aiokafka error inside three unrelated tests.

Every credential in this module is a synthetic test constant. No real
credential appears in any test, fixture or compose file.
"""

from __future__ import annotations

import re
import socket
import struct
import subprocess
import time
import uuid
from dataclasses import dataclass

# Pinned by digest, not by a floating tag: the same job on two runners must
# resolve the same image. This is the multi-arch manifest-list digest of
# redpandadata/redpanda:v24.2.7 -- `.github/workflows/ci.yml` pre-pulls this
# exact reference and tests/unit/docker/test_omn18012_harness_runner_topology.py
# asserts the two never drift apart.
REDPANDA_IMAGE = (
    "redpandadata/redpanda:v24.2.7@sha256:"
    "82a69763bef8d8b55ea5a520fa1b38f993908ef68946819ca1aed43541824c48"
)

# A second, container-local listener. `rpk` runs with `docker exec` INSIDE the
# broker, so it must not be sent out to the host address and back in: that
# would make every topic/produce call depend on NAT hairpinning. This port is
# never published and lives only in the broker's own namespace.
INTERNAL_PORT = 9092

# Applied to every container this harness starts. Concurrent CI jobs share one
# Docker host; a leaked container has to be attributable.
HARNESS_LABEL = "com.omninode.omn18012-harness"

# docker's own wording when a host port is taken. `_free_port()` binds in the
# TEST PROCESS's namespace, which on a containerized runner is not the
# namespace the port is published into, so a collision is possible and is a
# deterministic, self-reported condition with a specific remedy: pick another
# port. Nothing else is retried.
_PORT_TAKEN = ("port is already allocated", "address already in use")
_MAX_PORT_ATTEMPTS = 5

# Synthetic test credentials. Not a secret; never used outside a throwaway
# container that is destroyed at the end of the test session.
SASL_USERNAME = "omn18012-harness"
SASL_PASSWORD = "omn18012-synthetic-not-a-real-secret"
SASL_MECHANISM = "SCRAM-SHA-256"
SECURITY_PROTOCOL = "SASL_PLAINTEXT"


class HarnessError(RuntimeError):
    """The harness could not be brought up. Never swallowed into a skip."""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_cgroup() -> str:
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def _self_container_id() -> str | None:
    """The container this process runs in, when it runs in one at all.

    Mirrors "Detect runner network topology" in
    ``.github/workflows/reusable-runtime-boot.yml``: try every 64-hex id in
    the cgroup file, then the hostname (cgroup v2 on these runners exposes no
    id, and the hostname IS the 12-hex container id -- the live failing logs
    show ``Machine name: '8f635a7e3c47'``), and let the daemon adjudicate.
    """
    candidates: list[str] = list(
        dict.fromkeys(re.findall(r"[0-9a-f]{64}", _read_cgroup()))
    )
    try:
        candidates.append(socket.gethostname())
    except OSError:
        pass
    for candidate in candidates:
        if not candidate:
            continue
        try:
            proc = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--type",
                    "container",
                    "--format",
                    "{{.Id}}",
                    candidate,
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    return None


def _default_route_gateway() -> str | None:
    """This container's default route -- the Docker host side of its bridge."""
    try:
        with open("/proc/net/route", encoding="utf-8") as route_file:
            next(route_file)
            for line in route_file:
                fields = line.split()
                if len(fields) >= 3 and fields[1] == "00000000":
                    return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except (OSError, StopIteration, ValueError):
        return None
    return None


def resolve_docker_host_address() -> str:
    """An address for the Docker host's published ports, from THIS process.

    Deterministic and evaluated once: a bare runner is the Docker host and
    keeps ``localhost``; a containerized runner gets its default route. When
    detection finds a container but no route to read, the answer stays
    ``localhost`` and the client-path readiness probe below turns that into a
    loud, topology-naming failure -- the harness never gropes for an address
    through a chain of broker restarts.
    """
    if _self_container_id() is None:
        return "localhost"
    return _default_route_gateway() or "localhost"


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """Can THIS process open a TCP connection to the broker's external port?"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def docker_available() -> bool:
    """True when a docker daemon is reachable."""
    try:
        return (
            subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                timeout=30,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


@dataclass(frozen=True)
class RedpandaSasl:
    """A running, auth-required Redpanda broker."""

    container: str
    port: int
    host: str = "localhost"

    @property
    def bootstrap(self) -> str:
        """The address the TEST PROCESS reaches the external listener at."""
        return f"{self.host}:{self.port}"

    # -- rpk -------------------------------------------------------------
    def rpk(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Run authenticated ``rpk`` inside the broker container."""
        cmd = [
            "docker",
            "exec",
            self.container,
            "rpk",
            *args,
            # In-container: the internal listener, which never leaves this
            # container's namespace. Never self.bootstrap -- that is the
            # runner-side address.
            "-X",
            f"brokers=localhost:{INTERNAL_PORT}",
            "-X",
            f"user={SASL_USERNAME}",
            "-X",
            f"pass={SASL_PASSWORD}",
            "-X",
            f"sasl.mechanism={SASL_MECHANISM}",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False
        )
        if check and proc.returncode != 0:
            raise HarnessError(
                f"rpk {' '.join(args)} failed rc={proc.returncode}: "
                f"{proc.stdout}\n{proc.stderr}"
            )
        return proc

    def rpk_unauthenticated(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run ``rpk`` with NO credentials -- the positive control.

        A harness whose listener is not actually enforcing auth would make
        every 'the client authenticated' assertion in this suite vacuous.
        """
        return subprocess.run(
            [
                "docker",
                "exec",
                self.container,
                "rpk",
                *args,
                "-X",
                f"brokers=localhost:{INTERNAL_PORT}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def create_topic(self, name: str, *, partitions: int = 1) -> None:
        self.rpk("topic", "create", name, "-p", str(partitions), "-r", "1")

    def produce(self, topic: str, payloads: list[str]) -> None:
        """Produce one record per payload (newline-delimited)."""
        body = "".join(f"{p}\n" for p in payloads)
        proc = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                self.container,
                "rpk",
                "topic",
                "produce",
                topic,
                "-X",
                f"brokers=localhost:{INTERNAL_PORT}",
                "-X",
                f"user={SASL_USERNAME}",
                "-X",
                f"pass={SASL_PASSWORD}",
                "-X",
                f"sasl.mechanism={SASL_MECHANISM}",
            ],
            input=body,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if proc.returncode != 0:
            raise HarnessError(
                f"produce to {topic} failed rc={proc.returncode}: "
                f"{proc.stdout}\n{proc.stderr}"
            )

    def offsets(self, topic: str, partition: int = 0) -> tuple[int, int]:
        """Return ``(log_start, high_watermark)`` for one partition via rpk.

        Parsed from the ``rpk topic describe -p`` table; ``rpk`` in
        v24.2.7 has no ``--format json`` for this subcommand. The tests
        additionally re-assert the log start through ``aiokafka`` itself,
        because the client library's view is the one that matters.
        """
        proc = self.rpk("topic", "describe", topic, "-p")
        for line in proc.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 6 and fields[0].isdigit():
                if int(fields[0]) == partition:
                    return int(fields[4]), int(fields[5])
        raise HarnessError(
            f"partition {partition} absent from rpk describe of {topic}: {proc.stdout}"
        )

    def trim_prefix(self, topic: str, offset: int, partition: int = 0) -> None:
        """Advance the partition's log start. Verified, never assumed."""
        self.rpk(
            "topic",
            "trim-prefix",
            topic,
            "--offset",
            str(offset),
            "--partitions",
            str(partition),
            "--no-confirm",
        )
        log_start, _high = self.offsets(topic, partition)
        if log_start < offset:
            raise HarnessError(
                f"trim-prefix did not take on {topic}/{partition}: "
                f"log start is {log_start}, expected >= {offset}"
            )

    # -- env ---------------------------------------------------------------
    def env(self) -> dict[str, str]:
        """The KAFKA_* environment a client needs to reach this broker."""
        return {
            "KAFKA_BOOTSTRAP_SERVERS": self.bootstrap,
            "KAFKA_SECURITY_PROTOCOL": SECURITY_PROTOCOL,
            "KAFKA_SASL_MECHANISM": SASL_MECHANISM,
            "KAFKA_SASL_USERNAME": SASL_USERNAME,
            "KAFKA_SASL_PASSWORD": SASL_PASSWORD,
        }


def _docker_run(
    container: str, port: int, host: str
) -> subprocess.CompletedProcess[str]:
    """One `docker run` of the auth-required broker on a chosen host port."""
    return subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container,
            "--label",
            f"{HARNESS_LABEL}=1",
            "-p",
            f"{port}:{port}",
            "-e",
            f"RP_BOOTSTRAP_USER={SASL_USERNAME}:{SASL_PASSWORD}:{SASL_MECHANISM}",
            REDPANDA_IMAGE,
            "redpanda",
            "start",
            # Two listeners, deliberately. `internal` is container-local and is
            # what in-container `rpk` uses. `external` is the published port,
            # and it must ADVERTISE the runner-reachable host: advertised
            # metadata is what redirects every client, so advertising
            # "localhost" here is exactly the OMN-18012 failure regardless of
            # what bootstrap string the caller passes.
            f"--kafka-addr=internal://0.0.0.0:{INTERNAL_PORT},external://0.0.0.0:{port}",
            f"--advertise-kafka-addr=internal://localhost:{INTERNAL_PORT},external://{host}:{port}",
            "--mode=dev-container",
            "--smp=1",
            "--memory=1G",
            "--reserve-memory=0M",
            "--reactor-backend=epoll",
            "--default-log-level=warn",
            "--set",
            "redpanda.enable_sasl=true",
            "--set",
            f"redpanda.superusers=['{SASL_USERNAME}']",
            # Small segments so a trim-prefix has something to reclaim and a
            # retention-truncated partition is reachable in a test.
            "--set",
            "redpanda.log_segment_size=1048576",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def start_redpanda_sasl(
    *, boot_timeout_s: float = 120.0, host: str | None = None
) -> RedpandaSasl:
    """Start an auth-required Redpanda and return it once a CLIENT can reach it.

    Raises rather than skips: a harness that cannot come up is a red test,
    not an absent one. A silently-skipped boundary test is exactly how the
    2026-09-06 escapes stayed invisible.
    """
    resolved_host = host if host is not None else resolve_docker_host_address()

    container = ""
    proc: subprocess.CompletedProcess[str] | None = None
    port = 0
    for _attempt in range(_MAX_PORT_ATTEMPTS):
        port = _free_port()
        container = f"omn18012-rp-{uuid.uuid4().hex[:10]}"
        proc = _docker_run(container, port, resolved_host)
        if proc.returncode == 0:
            break
        combined = f"{proc.stdout}\n{proc.stderr}".lower()
        if not any(marker in combined for marker in _PORT_TAKEN):
            # Not a port collision. Bounded, cause-specific reselection only --
            # a blind retry loop would convert a real defect into a slow flake.
            break
        # _free_port() picked in this process's namespace; the port is
        # published into the Docker HOST's, where it was already taken.
        # Drop the half-created container and reselect.
        subprocess.run(
            ["docker", "rm", "-f", container],
            capture_output=True,
            timeout=120,
            check=False,
        )

    if proc is None or proc.returncode != 0:
        detail = (
            "no attempt was made" if proc is None else f"{proc.stdout}\n{proc.stderr}"
        )
        raise HarnessError(f"docker run failed: {detail}")

    broker = RedpandaSasl(container=container, port=port, host=resolved_host)
    try:
        return _await_ready(broker, boot_timeout_s)
    except BaseException:
        # Never leak a container on a shared Docker host, on any failure path.
        stop_redpanda(broker)
        raise


def _await_ready(broker: RedpandaSasl, boot_timeout_s: float) -> RedpandaSasl:
    """Ready means BOTH: the broker answers, and this process can connect.

    The `rpk` half runs inside the broker container, which is the one
    namespace where `localhost:<port>` is always right -- on its own it
    reported a broker no client could reach as ready (OMN-18012). The TCP half
    is the client path the tests actually use. Neither substitutes for the
    other: a TCP accept from a half-started broker is not readiness either.
    """
    deadline = time.monotonic() + boot_timeout_s
    last = ""
    broker_answered = False
    while True:
        probe = broker.rpk("cluster", "info", check=False)
        if probe.returncode == 0:
            broker_answered = True
            if _tcp_reachable(broker.host, broker.port):
                return broker
        else:
            last = f"{probe.stdout}\n{probe.stderr}"
        if time.monotonic() >= deadline:
            break
        time.sleep(2)

    logs = subprocess.run(
        ["docker", "logs", "--tail", "40", broker.container],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if broker_answered:
        raise HarnessError(
            f"the broker answered `rpk cluster info` inside its own container, "
            f"but {broker.bootstrap} is not reachable from this test process. "
            f"The broker port is published into the Docker HOST's network "
            f"namespace; this process is in a different network namespace (a "
            f"containerized CI runner reaching the daemon through a mounted "
            f"docker.sock), so the published port is unreachable at that "
            f"address. Resolved Docker host address: {broker.host!r} "
            f"(container id of self: {_self_container_id()!r}, default route: "
            f"{_default_route_gateway()!r}). See OMN-18012, and the same fix "
            f"for the nightly stack in OMN-15567.\n"
            f"container logs: {logs.stdout}\n{logs.stderr}"
        )
    raise HarnessError(
        f"redpanda did not become ready in {boot_timeout_s}s. "
        f"last rpk: {last}\ncontainer logs: {logs.stdout}\n{logs.stderr}"
    )


def stop_redpanda(broker: RedpandaSasl) -> None:
    subprocess.run(
        ["docker", "rm", "-f", broker.container],
        capture_output=True,
        timeout=120,
        check=False,
    )
