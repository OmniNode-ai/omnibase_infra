# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15567: the nightly e2e stack must be Healthy AND reachable.

`.github/workflows/nightly-integration.yml` had concluded `failure` on every
visible scheduled run (2026-07-22 through 2026-07-31): the `Spin up e2e stack`
step died with `omnibase-infra-redpanda exited (132)` (SIGILL) before pytest
ever ran. That symptom turned out to be a confound of OMN-15565 (the e2e
redpanda was being recreated inside the lab lane's compose project, on the
lab lane's data volume, with lane-incompatible startup flags) -- once
OMN-15565's isolated-namespace fix landed, redpanda started Healthy in ~5s on
both a `workflow_dispatch` run (30681782952) and the next scheduled run
(30685774570).

pytest then ran to completion on both of those runs (2471 passed / 444
skipped / 1 xfailed / 6 failed / 13 errors in 848.45s) and surfaced the
*second*, independent defect this ticket targets: the self-hosted
`omnibase-ci` runner executes inside a container that does not share a
network namespace with the Docker host (the same topology
`reusable-runtime-boot.yml:250-270` already documents and handles for the
Tier-1/Tier-2 smoke workflows). 13 of those errors were the runner's own
connection to `localhost:<host-published-port>` failing outright
(`aiokafka.errors.KafkaConnectionError: Unable to bootstrap from
[('localhost', 40335, ...)]`, `asyncpg` `OSError: ... Connect call failed`)
even when the port was the freshly-derived, correctly-read dynamic port --
proving this was never a mismatched-port bug, it was the host string itself
being unreachable from the runner process.

This module proves the connectivity-detection/resolution steps this ticket
adds actually execute the fallback chain (Docker DNS -> localhost -> Docker
host gateway) and fail closed with diagnostics -- never silently -- when none
of the three are reachable, mirroring the precedent in
`reusable-runtime-boot.yml`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "nightly-integration.yml"


def _workflow_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = workflow["jobs"]["integration-tests"]["steps"]
    return steps


def _step(name: str) -> dict[str, Any]:
    return next(step for step in _workflow_steps() if step.get("name") == name)


def _step_index(name: str) -> int:
    names = [step.get("name") for step in _workflow_steps()]
    return names.index(name)


# ---------------------------------------------------------------------------
# Step ordering — the redpanda advertise host must be known before spin-up;
# the reachable host must be resolved after the stack is Healthy but before
# pytest connects to it.
# ---------------------------------------------------------------------------


def test_detect_topology_runs_before_spin_up_e2e_stack() -> None:
    """E2E_REDPANDA_ADVERTISE_HOST must exist before `docker compose up` reads it."""
    assert _step_index("Detect runner network topology") < _step_index(
        "Spin up e2e stack"
    )


def test_resolve_connectivity_runs_between_health_checks_and_pytest() -> None:
    """The probe needs a Healthy stack, and pytest needs the probe's env vars."""
    assert (
        _step_index("Wait for health checks")
        < _step_index("Resolve reachable e2e connectivity host")
        < _step_index("Run integration tests")
    )


def test_spin_up_e2e_stack_does_not_hardcode_advertise_host() -> None:
    """Compose must inherit E2E_REDPANDA_ADVERTISE_HOST from the environment.

    docker-compose.e2e.yml's redpanda service already reads
    `${E2E_REDPANDA_ADVERTISE_HOST:-localhost}` for --advertise-kafka-addr;
    nightly-integration.yml never set that variable pre-fix, so it always
    silently defaulted to "localhost" regardless of runner topology.
    """
    compose_text = (REPO_ROOT / "docker" / "docker-compose.e2e.yml").read_text(
        encoding="utf-8"
    )
    assert "E2E_REDPANDA_ADVERTISE_HOST" in compose_text
    detect_run = _step("Detect runner network topology")["run"]
    assert "E2E_REDPANDA_ADVERTISE_HOST=" in detect_run


# ---------------------------------------------------------------------------
# Execute the real step shell (stubbed docker/python3) — not a structural
# grep-match. Mirrors test_e2e_compose_lane_isolation.py's stub pattern.
# ---------------------------------------------------------------------------


def _write_stub(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    path.chmod(0o700)


def _run_step(
    step_name: str, tmp_path: Path, extra_env: dict[str, str], stubs: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Execute a workflow step's `run:` shell with stub binaries on PATH."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for stub_name, body in stubs.items():
        _write_stub(fake_bin / stub_name, body)

    github_env = tmp_path / "github-env"
    github_env.write_text("", encoding="utf-8")
    env = (
        os.environ
        | extra_env
        | {
            "GITHUB_ENV": str(github_env),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        }
    )
    result = subprocess.run(
        ["bash", "-c", _step(step_name)["run"]],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, github_env


def _read_github_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return dict(line.split("=", 1) for line in lines)


# --- "Detect runner network topology" ---------------------------------------

_DOCKER_INSPECT_ALWAYS_FAILS = "exit 1\n"


def test_detect_topology_defaults_to_localhost_off_a_bare_runner(
    tmp_path: Path,
) -> None:
    """A non-containerized runner (every `docker inspect` candidate fails) stays localhost."""
    result, github_env = _run_step(
        "Detect runner network topology",
        tmp_path,
        extra_env={},
        stubs={"docker": _DOCKER_INSPECT_ALWAYS_FAILS},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    values = _read_github_env(github_env)
    assert values["E2E_REDPANDA_ADVERTISE_HOST"] == "localhost"
    assert values["OMNIBASE_INFRA_RUNNER_CONTAINER_ID"] == ""


# --- "Resolve reachable e2e connectivity host" -------------------------------

_KAFKA_PORT = "40335"
_POSTGRES_PORT = "45937"
_GATEWAY = "172.17.0.1"

_BASE_RESOLVE_ENV = {
    "INTEGRATION_POSTGRES_PASSWORD": "test-password",
    "KAFKA_PORT": _KAFKA_PORT,
    "POSTGRES_PORT": _POSTGRES_PORT,
    "OMNIBASE_INFRA_NETWORK": "omnibase-infra-e2e-test-network",
    "OMNIBASE_INFRA_POSTGRES_CONTAINER": "omnibase-infra-e2e-test-postgres",
    "OMNIBASE_INFRA_REDPANDA_CONTAINER": "omnibase-infra-e2e-test-redpanda",
    # No OMNIBASE_INFRA_RUNNER_CONTAINER_ID -> runner_on_compose_network stays
    # false, so every case below exercises the localhost/gateway branches
    # without needing a `docker network connect`/`docker inspect` stub.
}


def _python3_can_connect_stub(reachable_host_port_pairs: str) -> str:
    """A python3 stub that answers can_connect() for an allowlist of host:port pairs.

    `can_connect` invokes `python3 - "$host" "$port" <<'PY' ...`, so from the
    stub's perspective argv is [python3, "-", host, port]; stdin (the real
    heredoc script) is drained and ignored.
    """
    return (
        "cat >/dev/null\n"
        f'allow="{reachable_host_port_pairs}"\n'
        'host="$2"\n'
        'port="$3"\n'
        'case "$allow" in\n'
        '  *"$host:$port"*) exit 0 ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n"
    )


def test_resolve_connectivity_fails_closed_when_nothing_is_reachable(
    tmp_path: Path,
) -> None:
    """No Docker DNS, no localhost, no gateway reachable -> exit 1, no env emitted."""
    result, github_env = _run_step(
        "Resolve reachable e2e connectivity host",
        tmp_path,
        extra_env={**_BASE_RESOLVE_ENV, "E2E_REDPANDA_ADVERTISE_HOST": _GATEWAY},
        stubs={"python3": _python3_can_connect_stub("")},
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "unreachable from the runner" in (result.stdout + result.stderr)
    assert _read_github_env(github_env) == {}


def test_resolve_connectivity_prefers_localhost_when_reachable(
    tmp_path: Path,
) -> None:
    """Host-published ports reachable at localhost -> localhost wins, dynamic ports kept."""
    result, github_env = _run_step(
        "Resolve reachable e2e connectivity host",
        tmp_path,
        extra_env={**_BASE_RESOLVE_ENV, "E2E_REDPANDA_ADVERTISE_HOST": "localhost"},
        stubs={
            "python3": _python3_can_connect_stub(
                f"localhost:{_POSTGRES_PORT} localhost:{_KAFKA_PORT}"
            )
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    values = _read_github_env(github_env)
    assert values["KAFKA_BOOTSTRAP_SERVERS"] == f"localhost:{_KAFKA_PORT}"
    assert values["INTEGRATION_POSTGRES_HOST"] == "localhost"
    assert values["INTEGRATION_POSTGRES_PORT"] == _POSTGRES_PORT
    assert values["OMNIBASE_INFRA_DB_URL"] == (
        f"postgresql://postgres:test-password@localhost:{_POSTGRES_PORT}/omnibase_infra"
    )


def test_resolve_connectivity_falls_back_to_docker_host_gateway(
    tmp_path: Path,
) -> None:
    """localhost unreachable (the confirmed live failure mode) -> gateway wins.

    This is the exact scenario observed live on runs 30681782952/30685774570:
    the runner could not reach "localhost:<host-published-port>" at all. This
    test proves the fallback this ticket adds actually resolves that case
    instead of letting pytest fail on an opaque connection-refused error.
    """
    result, github_env = _run_step(
        "Resolve reachable e2e connectivity host",
        tmp_path,
        extra_env={**_BASE_RESOLVE_ENV, "E2E_REDPANDA_ADVERTISE_HOST": _GATEWAY},
        stubs={
            "python3": _python3_can_connect_stub(
                f"{_GATEWAY}:{_POSTGRES_PORT} {_GATEWAY}:{_KAFKA_PORT}"
            )
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    values = _read_github_env(github_env)
    assert values["KAFKA_BOOTSTRAP_SERVERS"] == f"{_GATEWAY}:{_KAFKA_PORT}"
    assert values["INTEGRATION_POSTGRES_HOST"] == _GATEWAY
    assert values["REDPANDA_ADVERTISE_HOST"] == _GATEWAY
    assert values["OMNIBASE_INFRA_DB_URL"] == (
        f"postgresql://postgres:test-password@{_GATEWAY}:{_POSTGRES_PORT}/omnibase_infra"
    )


def test_resolve_connectivity_refuses_to_resolve_to_live_201_host(
    tmp_path: Path,
) -> None:
    """Even a reachable gateway is refused if it is the live .201 instance."""
    live_host = "192.168.86.201"
    result, github_env = _run_step(
        "Resolve reachable e2e connectivity host",
        tmp_path,
        extra_env={**_BASE_RESOLVE_ENV, "E2E_REDPANDA_ADVERTISE_HOST": live_host},
        stubs={
            "python3": _python3_can_connect_stub(
                f"{live_host}:{_POSTGRES_PORT} {live_host}:{_KAFKA_PORT}"
            )
        },
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "live .201 instance" in (result.stdout + result.stderr)
    values = _read_github_env(github_env)
    assert live_host not in values.get("KAFKA_BOOTSTRAP_SERVERS", "")
    assert live_host not in values.get("OMNIBASE_INFRA_DB_URL", "")


# ---------------------------------------------------------------------------
# Test-suite side: the two hardcoded-bootstrap files this ticket also fixes.
# aiokafka tests must read the CI-derived KAFKA_BOOTSTRAP_SERVERS, mirroring
# the pre-existing correct pattern in test_runtime_health_monitor.py, or the
# runner-topology fix above is silently defeated for those two files.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_file",
    [
        "tests/integration/test_consumer_health_pipeline.py",
        "tests/integration/test_runtime_log_bridge_pipeline.py",
    ],
)
def test_kafka_test_files_read_bootstrap_servers_from_env(test_file: str) -> None:
    text = (REPO_ROOT / test_file).read_text(encoding="utf-8")
    assert 'BOOTSTRAP_SERVERS = "localhost:19092"' not in text, (
        f"{test_file} hardcodes the docker-compose.e2e.yml default bootstrap "
        f"address, silently ignoring nightly-integration.yml's run-scoped "
        f"KAFKA_BOOTSTRAP_SERVERS (OMN-15567)."
    )
    assert 'os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")' in text


def test_python_can_import_the_fixed_kafka_test_modules() -> None:
    """Import (not just grep) both fixed files to prove the edit is syntactically real."""
    for module in (
        "tests.integration.test_consumer_health_pipeline",
        "tests.integration.test_runtime_log_bridge_pipeline",
    ):
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
