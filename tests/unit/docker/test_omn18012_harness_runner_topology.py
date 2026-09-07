# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18012: the customer-path harness must be reachable from the RUNNER.

What went wrong
---------------
``tests/integration/customer_path/redpanda_sasl_harness.py`` published the
broker port with ``docker run -p PORT:PORT`` -- into the **Docker host's**
network namespace -- and advertised ``external://localhost:PORT``. That is
correct on a GitHub-hosted runner, which *is* the Docker host. It is wrong on
this org's self-hosted ``omnibase-ci`` runners, which are **containers** that
reach the daemon through a mounted ``/var/run/docker.sock`` and do not share
the host's network namespace (``docker/docker-compose.runners.yml`` declares
no ``networks:`` block and no ``network_mode: host``).

Live census of every run of the required job ``Customer Path Boundary
(OMN-18012)`` on 2026-09-07, 13 runs, zero exceptions: 6/6 GitHub-hosted runs
succeeded, 7/7 ``omninode-runner-*`` runs failed. Same-PR control on
``omnibase_infra#3283``: run 34116039416 PASSED at 11:40:58Z on ``GitHub
Actions 1002898605``; run 34122261538 on the same branch FAILED at 12:51:11Z
on ``omninode-runner-51``. The failure was always::

    aiokafka.errors.KafkaConnectionError: KafkaConnectionError: Unable to
    bootstrap from [('localhost', 52567, <AddressFamily.AF_UNSPEC: 0>)]
    ERROR aiokafka:client.py:206 Unable connect to "localhost:52567":
    [Errno 111] Connect call failed ('127.0.0.1', 52567)

ECONNREFUSED on both address families -- nothing bound in the caller's
namespace. Four wedged PRs, none of which touched this suite.

Why the harness could not see it
--------------------------------
``start_redpanda_sasl`` gated readiness on ``rpk cluster info`` executed with
``docker exec`` **inside the broker container**. That probe answers from the
one namespace where ``localhost:PORT`` is always right, so a broker that no
client could reach still reported ready and the failure surfaced as an opaque
aiokafka error inside three unrelated tests.

Why a bootstrap-string-only fix is not enough
---------------------------------------------
Measured locally 2026-09-07 (Docker 29.6.2, port 52275, the harness's exact
``docker run``): a sibling container connected fine to the broker container's
own address, and ``rpk cluster info -X brokers=172.17.0.6:52275`` then
returned ``HOST localhost PORT 52275`` -- the broker's *advertised* metadata
sends every client straight back to the unreachable name. The advertised
address has to change too, which is what an ``internal://``/``external://``
listener pair gives us: in-container ``rpk`` keeps a localhost listener that
never leaves the container, and the test process gets an external listener
advertised at an address it can actually reach.

Precedent
---------
This repo already solved exactly this under OMN-15567 for the nightly e2e
stack (``.github/workflows/reusable-runtime-boot.yml``, "Detect runner network
topology"), pinned by ``test_nightly_e2e_runner_connectivity.py``. The new
harness did not adopt it. This module is the adoption, in Python, at the
harness.
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from tests.integration.customer_path import redpanda_sasl_harness as harness

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]

_GATEWAY = "172.18.0.1"
_RUNNER_CONTAINER_ID = (
    "8f635a7e3c47000000000000000000000000000000000000000000000000dead"
)


# ---------------------------------------------------------------------------
# The address the test process must use
# ---------------------------------------------------------------------------


def test_bootstrap_is_the_resolved_host_not_a_hardcoded_localhost() -> None:
    """Every test reaches the broker through ``.bootstrap`` / ``.env()``.

    Both must carry the resolved host, or the resolution is decorative.
    """
    broker = harness.RedpandaSasl(container="c", port=52567, host=_GATEWAY)
    assert broker.bootstrap == f"{_GATEWAY}:52567"
    assert broker.env()["KAFKA_BOOTSTRAP_SERVERS"] == f"{_GATEWAY}:52567"


def test_resolve_docker_host_address_is_localhost_off_a_bare_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GitHub-hosted runner IS the Docker host -- localhost stays correct.

    This is the branch that keeps the 6/6 hosted successes green.
    """
    monkeypatch.setattr(harness, "_self_container_id", lambda: None)
    assert harness.resolve_docker_host_address() == "localhost"


def test_resolve_docker_host_address_is_the_docker_host_gateway_in_a_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A containerized ``omnibase-ci`` runner reaches host-published ports at
    its own default route -- the host side of the bridge it is attached to."""
    monkeypatch.setattr(harness, "_self_container_id", lambda: _RUNNER_CONTAINER_ID)
    monkeypatch.setattr(harness, "_default_route_gateway", lambda: _GATEWAY)
    assert harness.resolve_docker_host_address() == _GATEWAY


def test_resolve_docker_host_address_stays_deterministic_with_no_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No default route to read -> localhost, and the CLIENT-PATH readiness
    probe below is what turns that into a loud, named failure. Resolution
    never guesses through a chain of broker restarts."""
    monkeypatch.setattr(harness, "_self_container_id", lambda: _RUNNER_CONTAINER_ID)
    monkeypatch.setattr(harness, "_default_route_gateway", lambda: None)
    assert harness.resolve_docker_host_address() == "localhost"


def test_self_container_id_falls_back_to_the_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgroup v2 in these runners exposes no 64-hex id; the 12-hex hostname is
    the id that resolves. The live failing logs show exactly that at line 4:
    ``Machine name: '8f635a7e3c47'``."""
    monkeypatch.setattr(harness, "_read_cgroup", lambda: "0::/\n")
    monkeypatch.setattr(socket, "gethostname", lambda: "8f635a7e3c47")
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, f"{_RUNNER_CONTAINER_ID}\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert harness._self_container_id() == _RUNNER_CONTAINER_ID
    assert seen and "8f635a7e3c47" in seen[0]


# ---------------------------------------------------------------------------
# The address the BROKER advertises
# ---------------------------------------------------------------------------


class _FakeDocker:
    """Records ``docker`` argv and answers a scripted boot sequence."""

    def __init__(
        self,
        *,
        run_results: list[tuple[int, str]] | None = None,
        rpk_rc: int = 0,
    ) -> None:
        self.calls: list[list[str]] = []
        self.run_results = list(run_results or [(0, "")])
        self.rpk_rc = rpk_rc

    def __call__(self, cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if cmd[:2] == ["docker", "run"]:
            code, err = self.run_results.pop(0) if self.run_results else (0, "")
            return subprocess.CompletedProcess(cmd, code, "container-id\n", err)
        if cmd[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(cmd, self.rpk_rc, "CLUSTER\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def run_argv(self) -> list[list[str]]:
        return [c for c in self.calls if c[:2] == ["docker", "run"]]


def _install(
    monkeypatch: pytest.MonkeyPatch,
    docker: _FakeDocker,
    *,
    reachable: bool,
    host: str = _GATEWAY,
) -> None:
    monkeypatch.setattr(subprocess, "run", docker)
    monkeypatch.setattr(harness, "resolve_docker_host_address", lambda: host)
    monkeypatch.setattr(harness, "_tcp_reachable", lambda *_a, **_k: reachable)
    monkeypatch.setattr(time, "sleep", lambda _s: None)


def _flag(argv: list[str], prefix: str) -> str:
    return next(a for a in argv if a.startswith(prefix))


def test_start_advertises_the_resolved_host_on_the_external_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The advertised metadata is what every client is redirected to."""
    docker = _FakeDocker()
    _install(monkeypatch, docker, reachable=True)

    broker = harness.start_redpanda_sasl()

    argv = docker.run_argv()[0]
    advertise = _flag(argv, "--advertise-kafka-addr=")
    assert f"external://{_GATEWAY}:{broker.port}" in advertise, advertise
    assert "external://localhost:" not in advertise, advertise
    assert broker.host == _GATEWAY


def test_start_keeps_an_internal_listener_for_in_container_rpk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``rpk`` runs with ``docker exec`` INSIDE the broker. Sending it out to
    the host gateway and back in would make every topic/produce call depend on
    NAT hairpinning. A separate internal listener keeps that traffic in the
    container namespace and never touches the host."""
    docker = _FakeDocker()
    _install(monkeypatch, docker, reachable=True)

    broker = harness.start_redpanda_sasl()

    argv = docker.run_argv()[0]
    kafka_addr = _flag(argv, "--kafka-addr=")
    advertise = _flag(argv, "--advertise-kafka-addr=")
    assert f"internal://0.0.0.0:{harness.INTERNAL_PORT}" in kafka_addr, kafka_addr
    assert f"external://0.0.0.0:{broker.port}" in kafka_addr, kafka_addr
    assert f"internal://localhost:{harness.INTERNAL_PORT}" in advertise, advertise

    exec_argv = next(c for c in docker.calls if c[:2] == ["docker", "exec"])
    assert f"brokers=localhost:{harness.INTERNAL_PORT}" in exec_argv, exec_argv


def test_the_broker_container_is_labelled_and_uniquely_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent jobs share one Docker host. Names must not collide and a
    leaked container must be attributable to this harness."""
    docker = _FakeDocker()
    _install(monkeypatch, docker, reachable=True)

    first = harness.start_redpanda_sasl()
    second = harness.start_redpanda_sasl()

    assert first.container != second.container
    argv = docker.run_argv()[0]
    assert "--label" in argv
    assert f"{harness.HARNESS_LABEL}=1" in argv, argv


def test_the_image_is_pinned_by_digest() -> None:
    """A floating tag is not a hermetic input: the same job on two runners can
    resolve two different images. ``REDPANDA_IMAGE`` must carry an ``@sha256:``
    digest, and ci.yml's pre-pull must name the identical reference."""
    assert "@sha256:" in harness.REDPANDA_IMAGE, harness.REDPANDA_IMAGE
    workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert harness.REDPANDA_IMAGE in workflow.read_text(encoding="utf-8"), (
        "ci.yml pre-pulls a different reference than the harness runs"
    )


# ---------------------------------------------------------------------------
# Readiness must be proven on the CLIENT path, and fail loudly
# ---------------------------------------------------------------------------


def test_readiness_fails_named_when_the_client_path_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression, stated as a test: ``rpk`` inside the container is happy
    and the test process still cannot connect. That must be a HarnessError
    that NAMES the topology -- not an opaque aiokafka bootstrap error thrown
    later from three unrelated tests."""
    docker = _FakeDocker(rpk_rc=0)
    _install(monkeypatch, docker, reachable=False)

    with pytest.raises(harness.HarnessError) as failure:
        harness.start_redpanda_sasl(boot_timeout_s=0.01)

    message = str(failure.value)
    assert "network namespace" in message, message
    assert _GATEWAY in message, message
    assert "not reachable from this test process" in message, message
    assert any(c[:3] == ["docker", "rm", "-f"] for c in docker.calls), (
        "an unreachable broker must still be removed, not leaked on a shared "
        f"Docker host: {docker.calls}"
    )


def test_readiness_still_requires_the_broker_itself_to_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client-path probe is added to the rpk probe, never substituted for
    it: a TCP accept from a half-started broker is not readiness."""
    docker = _FakeDocker(rpk_rc=1)
    _install(monkeypatch, docker, reachable=True)

    with pytest.raises(harness.HarnessError) as failure:
        harness.start_redpanda_sasl(boot_timeout_s=0.01)
    assert "did not become ready" in str(failure.value)


def test_a_docker_host_port_collision_selects_a_fresh_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_free_port`` binds in the TEST PROCESS's namespace; on a containerized
    runner the port is published into the HOST's, where it may already be
    taken. That is a deterministic, self-reported docker error with a distinct
    fix (pick another port) -- not a flake to be papered over by re-running a
    failed boot."""
    docker = _FakeDocker(
        run_results=[
            (125, "Error: port is already allocated"),
            (0, ""),
        ]
    )
    _install(monkeypatch, docker, reachable=True)

    broker = harness.start_redpanda_sasl()

    runs = docker.run_argv()
    assert len(runs) == 2, runs
    first_port = _flag(runs[0], "--kafka-addr=")
    second_port = _flag(runs[1], "--kafka-addr=")
    assert first_port != second_port
    assert str(broker.port) in second_port


def test_an_unrelated_docker_run_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded, cause-specific reselection -- never a blind retry loop."""
    docker = _FakeDocker(run_results=[(125, "Error: no such image")])
    _install(monkeypatch, docker, reachable=True)

    with pytest.raises(harness.HarnessError) as failure:
        harness.start_redpanda_sasl()
    assert "no such image" in str(failure.value)
    assert len(docker.run_argv()) == 1
