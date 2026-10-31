# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17232: the HP DL560 Gen8 pool is arch-neutral CI on an opt-in label, with its own slice and git mirror.

Operator RULING 2026-10-29T14:00:26Z: make before break. The pool carries the opt-in label
``omnibase-ci-multiarch`` and never ``omnibase-ci`` (this CPU has no AVX2, so a job that imports an AVX2 wheel must
stay on the ``omnibase-ci`` pools), and its class ``action-multiarch`` is the one no ``omnibase-ci`` spread counts.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFIX = "hp051-ci-runner"
COMPOSE = REPO_ROOT / "docker" / f"docker-compose.runners-{PREFIX}.yml"
POOL_DIR = REPO_ROOT / "docker" / "runners" / "systemd" / "pools" / PREFIX
SLICE = POOL_DIR / "omnirunners.slice"
MIRROR_DAEMON = POOL_DIR / "omninode-git-mirror-daemon.service"
MIRROR_REFRESH = POOL_DIR / "omninode-git-mirror-refresh.service"
MIRROR_TIMER = POOL_DIR / "omninode-git-mirror-refresh.timer"
FLEET = REPO_ROOT / "config" / "runner_fleet.yaml"
OPT_IN_LABEL = "omnibase-ci-multiarch"
# The labels that put a runner in a required-context pool. The opt-in label is the only one this pool carries.
REQUIRED_POOL_LABELS = {"omnibase-ci", "omnibase-verify"}
MIRROR_ROOT = "/data/scratch/git-mirror"


def _compose() -> dict[str, Any]:
    return cast("dict[str, Any]", yaml.safe_load(COMPOSE.read_text(encoding="utf-8")))


def _directives(unit: str) -> dict[str, str]:
    """Uncommented ``Key=value`` lines of a unit file, last value winning."""
    pairs = (
        line.split("=", 1)
        for line in unit.replace("\\\n", " ").splitlines()
        if "=" in line and not line.lstrip().startswith(("#", "["))
    )
    return {key.strip(): value.strip() for key, value in pairs}


def _values(unit: str, key: str) -> list[str]:
    """Every value of a directive that systemd lets repeat (ExecStartPre), in file order."""
    return [
        value.strip()
        for name, value in (
            line.split("=", 1)
            for line in unit.replace("\\\n", " ").splitlines()
            if "=" in line and not line.lstrip().startswith(("#", "["))
        )
        if name.strip() == key
    ]


def test_every_runner_carries_the_opt_in_label_and_no_required_pool_label() -> None:
    services = _compose()["services"]
    assert len(services) == 4
    for name, service in services.items():
        labels = set(str(service["environment"]["RUNNER_LABELS"]).split(","))
        assert OPT_IN_LABEL in labels, f"{name} lacks the opt-in label"
        assert not labels & REQUIRED_POOL_LABELS, f"{name} would join a required pool"
        assert "arch-amd64" in labels
        assert labels == {
            "self-hosted",
            OPT_IN_LABEL,
            "hp-canary",
            "linux",
            "x64",
            "arch-amd64",
        }
        assert service["environment"]["RUNNER_GROUP"] == "omnibase-ci"


def test_no_repository_workflow_routes_jobs_to_the_canary_pin() -> None:
    for workflow in (REPO_ROOT / ".github" / "workflows").glob("*.y*ml"):
        document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        # Include matrix values as well as literal runs-on lists.
        assert "hp-canary" not in str(document), workflow.name


def test_compose_cli_and_yaml_loader_agree_on_inherited_service_settings() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI is not installed")
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE),
            "config",
            "--no-interpolate",
            "--format",
            "json",
        ],
        env={
            **os.environ,
            "COMPOSE_PROJECT_NAME": os.environ.get(
                "COMPOSE_PROJECT_NAME", "hp051-config-test"
            ),
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rendered = json.loads(result.stdout)["services"]
    declared = _compose()["services"]
    assert set(rendered) == set(declared)
    for name, service in rendered.items():
        assert service["environment"] == declared[name]["environment"]
        assert service["cgroup_parent"] == declared[name]["cgroup_parent"]
        assert _bytes(service["mem_limit"]) == _bytes(declared[name]["mem_limit"])
        assert _bytes(service["memswap_limit"]) == _bytes(
            declared[name]["memswap_limit"]
        )


def _bytes(value: str | int) -> int:
    if isinstance(value, int):
        return value
    assert value[-1].lower() == "g", value
    return int(value[:-1]) * 1024**3


def test_the_pool_is_its_own_cgroup_slice_with_declared_limits() -> None:
    services = _compose()["services"]
    assert {s["cgroup_parent"] for s in services.values()} == {"omnirunners.slice"}
    limits = _directives(SLICE.read_text(encoding="utf-8"))
    assert limits["CPUWeight"] == "25"
    assert limits["MemoryHigh"] == "96G"
    assert limits["MemoryMax"] == "112G"
    assert limits["MemorySwapMax"] == "2G"
    ram = sum(_bytes(s["mem_limit"]) for s in services.values())
    assert ram <= _bytes(limits["MemoryHigh"]) <= _bytes(limits["MemoryMax"])
    swap = sum(
        _bytes(s["memswap_limit"]) - _bytes(s["mem_limit"]) for s in services.values()
    )
    # Child allowances are ceilings; the parent imposes the tighter aggregate swap budget.
    assert 0 < _bytes(limits["MemorySwapMax"]) <= swap


def test_the_fleet_declares_the_pool_with_the_multiarch_class_only() -> None:
    document = yaml.safe_load(FLEET.read_text(encoding="utf-8"))
    rows = [row for row in document["hosts"] if row["host"] == "h051"]
    assert len(rows) == 1
    assert rows[0]["runner_name_prefix"] == PREFIX
    assert rows[0]["expected_count"] == 4
    assert rows[0]["arch"] == "amd64"
    assert rows[0]["classes"] == ["action-multiarch"]
    config = load_runner_fleet_config(FLEET)
    host = config.host_of_runner(f"{PREFIX}-1")
    assert host is not None
    assert host.declared_count("action") == 0
    assert host.declared_count("verify") == 0
    assert host.declared_count("action-multiarch") == 4
    # The pool adds nothing to the omnibase-ci spread: the total is what the other hosts declare, whatever that is.
    others = sum(h.declared_count("action") for h in config.hosts if h.host != "h051")
    assert config.declared_total("action") == others


def test_the_runner_names_match_the_declared_count() -> None:
    names = sorted(_compose()["services"])
    assert names == [f"{PREFIX}-{n}" for n in range(1, 5)]


def test_the_pool_has_its_own_git_mirror_on_the_data_volume() -> None:
    """The pre-seed reads its default address; the three units serve it from the data volume."""
    daemon = MIRROR_DAEMON.read_text(encoding="utf-8")
    # The address is host configuration: a required file, so the unit refuses to start without it.
    assert "EnvironmentFile=/etc/default/omninode-git-mirror\n" in daemon
    assert "--listen=${OMNI_GIT_MIRROR_LISTEN}" in daemon
    # Parsed, not grepped: the env file is required (no leading '-'), and ExecStart is a git daemon with the flags below.
    directives = _directives(daemon)
    assert directives["EnvironmentFile"] == "/etc/default/omninode-git-mirror"
    assert directives["ExecStart"].split()[0] == "/usr/lib/git-core/git-daemon"
    assert shlex.split(directives["ExecStart"]) == [
        "/usr/lib/git-core/git-daemon",
        "--verbose",
        "--reuseaddr",
        "--export-all",
        "--disable=receive-pack",
        "--forbid-override=receive-pack",
        "--disable=upload-archive",
        "--forbid-override=upload-archive",
        f"--base-path={MIRROR_ROOT}",
        "--listen=${OMNI_GIT_MIRROR_LISTEN}",
        "--port=9418",
        "--timeout=60",
        "--max-connections=24",
        MIRROR_ROOT,
    ]
    refresh = MIRROR_REFRESH.read_text(encoding="utf-8")
    assert f"Environment=OMNI_GIT_MIRROR_ROOT={MIRROR_ROOT}" in refresh
    assert "Type=oneshot" in refresh
    # A failing refresh must surface as a failed unit; exit 1 is never declared a success.
    assert "SuccessExitStatus" not in _directives(refresh)
    assert _directives(refresh)["ExecStart"].endswith("/git-mirror-refresh.sh")
    timer = MIRROR_TIMER.read_text(encoding="utf-8")
    assert "Unit=omninode-git-mirror-refresh.service" in timer


def test_daemon_preflight_waits_for_the_compose_bridge_without_a_start_limit() -> None:
    daemon = MIRROR_DAEMON.read_text(encoding="utf-8")
    directives = _directives(daemon)
    assert directives["Requires"] == "docker.service"
    assert "docker.service" in directives["After"].split()
    assert directives["StartLimitIntervalSec"] == "0"
    assert directives["Restart"] == "always"
    assert directives["RestartSec"] == "5"
    preflight = [shlex.split(value) for value in _values(daemon, "ExecStartPre")]
    assert len(preflight) == 2
    assert preflight[0][:2] == ["/bin/sh", "-ec"]
    assert preflight[1][:2] == ["/usr/bin/python3", "-c"]


@pytest.mark.parametrize(
    ("gateway", "status", "expected"),
    [("", "1", 1), ("", "0", 1), ("127.0.0.2", "0", 1), ("127.0.0.1", "0", 0)],
)
def test_daemon_network_preflight_requires_the_configured_compose_gateway(
    tmp_path: Path,
    gateway: str,
    status: str,
    expected: int,
) -> None:
    docker = tmp_path / "docker"
    docker.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" > "$PROBE_ARGUMENTS"\nprintf "%s\\n" "$PROBE_GATEWAY"\nexit "$PROBE_STATUS"\n'
    )
    docker.chmod(0o755)
    record = tmp_path / "arguments"
    first = _values(MIRROR_DAEMON.read_text(), "ExecStartPre")[0]
    # systemd turns $$ into a literal $ before passing the command to the shell.
    command = [argument.replace("$$", "$") for argument in shlex.split(first)]
    result = subprocess.run(
        command,
        env={
            **os.environ,
            "PATH": str(tmp_path),
            "OMNI_GIT_MIRROR_LISTEN": "127.0.0.1",
            "PROBE_ARGUMENTS": str(record),
            "PROBE_GATEWAY": gateway,
            "PROBE_STATUS": status,
        },
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == expected, result.stderr
    assert (
        record.read_text().strip()
        == "network inspect hp051-ci-runner_default --format {{(index .IPAM.Config 0).Gateway}}"
    )


@pytest.mark.parametrize(
    ("address", "diagnostic"),
    [
        (None, "KeyError"),
        ("", "ValueError"),
        pytest.param(str(IPv4Address(0)), "must be unicast", id="ipv4-wildcard"),
        ("::", "must be unicast"),
        ("[::]", "ValueError"),
        ("224.0.0.1", "must be unicast"),
        ("ff02::1", "must be unicast"),
        ("not-an-ip", "ValueError"),
        ("192.0.2.1", "OSError"),
    ],
)
def test_daemon_preflight_refuses_unsafe_or_unassigned_listen_addresses(
    address: str | None, diagnostic: str
) -> None:
    """Each refusal comes from the check meant to make it, so a broken one-liner cannot pass every case."""
    command = shlex.split(_values(MIRROR_DAEMON.read_text(), "ExecStartPre")[1])
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("OMNI_GIT_MIRROR_LISTEN", None)
    if address is not None:
        environment["OMNI_GIT_MIRROR_LISTEN"] = address
    result = subprocess.run(
        [sys.executable, *command[1:]],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode != 0, result
    assert "SyntaxError" not in result.stderr, result.stderr
    assert diagnostic in result.stderr, result.stderr


def test_daemon_preflight_accepts_an_assigned_unicast_address() -> None:
    command = shlex.split(_values(MIRROR_DAEMON.read_text(), "ExecStartPre")[1])
    result = subprocess.run(
        [sys.executable, *command[1:]],
        env={
            **os.environ,
            "OMNI_GIT_MIRROR_LISTEN": "127.0.0.1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_git_daemon_serves_reads_but_refuses_repository_write_and_archive_overrides(
    tmp_path: Path,
) -> None:
    mirror = tmp_path / "probe.git"
    subprocess.run(
        ["git", "init", "--bare", str(mirror)],
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )
    for option in ("daemon.receivepack", "daemon.uploadarch"):
        subprocess.run(
            ["git", "-C", str(mirror), "config", option, "true"],
            check=True,
            env=scrub_git_location_env(os.environ),
        )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    arguments = shlex.split(_directives(MIRROR_DAEMON.read_text())["ExecStart"])[1:]
    replacements = {
        f"--base-path={MIRROR_ROOT}": f"--base-path={tmp_path}",
        "--listen=${OMNI_GIT_MIRROR_LISTEN}": "--listen=127.0.0.1",
        "--port=9418": f"--port={port}",
        MIRROR_ROOT: str(tmp_path),
    }
    process = subprocess.Popen(
        ["git", "daemon", *(replacements.get(arg, arg) for arg in arguments)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=scrub_git_location_env(os.environ),
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                assert process.poll() is None, process.communicate()[1]
                assert time.monotonic() < deadline, "git daemon did not listen"
                time.sleep(0.02)
        remote = f"git://127.0.0.1:{port}/probe.git"
        read = subprocess.run(
            ["git", "ls-remote", remote],
            capture_output=True,
            check=False,
            timeout=10,
            env=scrub_git_location_env(os.environ),
        )
        assert read.returncode == 0, read.stderr
        # Speak Git's initial service request directly so no local ref or archive failure can mask an allowed service.
        for service in ("git-receive-pack", "git-upload-archive"):
            payload = f"{service} /probe.git\0host=127.0.0.1\0".encode()
            with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
                client.sendall(f"{len(payload) + 4:04x}".encode() + payload)
                response = client.recv(4096)
            assert b"ERR access denied" in response, (service, response)
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=5)


def test_no_runner_overrides_the_mirror_address() -> None:
    """The default gateway is right on this host; an override would silently point the pre-seed elsewhere."""
    for name, service in _compose()["services"].items():
        env = service["environment"]
        assert "OMNI_GIT_MIRROR_HOST" not in env, name
        assert "OMNI_GIT_MIRROR_DISABLE" not in env, name
