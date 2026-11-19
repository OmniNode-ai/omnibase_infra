# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25674: the CI pool that takes over the second amd64 host's capacity is declared on h051, sized from measurement.

Measured on the second amd64 host on 2026-11-19T17:04Z with 16 runners busy: the pool's slice sat at its 20 GiB MemoryHigh (memory.current
21.45 GB, `high` events 545,431,572), memory pressure `full` avg10 94.8 percent, host load 135, and the required `ci` jobs of
two repositories reached their 10 minute step limit there. The pool below moves that capacity to h051 (the general
`omnibase-ci` pool, no AVX2), with job directories on the host's NVMe.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pytest
import yaml

from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FLEET = REPO_ROOT / "config" / "runner_fleet.yaml"
SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"
POOLS = REPO_ROOT / "docker" / "runners" / "systemd" / "pools"
GIB = 1024**3

# (host, pool prefix, runner count, NVMe mount, labels the pool must carry, labels it must not carry, classes)
CASES = [
    (
        "h051",
        "hp051-omnibase-ci-runner",
        16,
        "/mnt/h051-nvme",
        {"omnibase-ci", "omnibase-ci-multiarch", "arch-amd64"},
        {"hp-canary", "omnibase-ci-mq"},
        ("action", "action-multiarch"),
    ),
]


def _compose(prefix: str) -> dict[str, Any]:
    path = REPO_ROOT / "docker" / f"docker-compose.runners-{prefix}.yml"
    return cast("dict[str, Any]", yaml.safe_load(path.read_text(encoding="utf-8")))


def _limits(path: Path) -> dict[str, str]:
    pairs = (
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith(("#", "["))
    )
    return {key.strip(): value.strip() for key, value in pairs}


def _gib(value: str) -> int:
    assert value[-1] in "gG", value
    return int(value[:-1])


@pytest.mark.parametrize(
    ("host", "prefix", "count", "nvme", "labels", "forbidden", "classes"), CASES
)
def test_pool_is_declared_with_its_compose_and_labels(
    host: str,
    prefix: str,
    count: int,
    nvme: str,
    labels: set[str],
    forbidden: set[str],
    classes: tuple[str, ...],
) -> None:
    config = load_runner_fleet_config(FLEET)
    declared = {
        pool.runner_name_prefix: pool
        for row in config.hosts
        if row.host == host
        for pool in row.pools
    }
    assert declared[prefix].expected_count == count
    assert declared[prefix].classes == classes

    services = _compose(prefix)["services"]
    assert sorted(services) == sorted(f"{prefix}-{n}" for n in range(1, count + 1))
    for name, service in services.items():
        assert service["container_name"] == name
        assert service["volumes"][-2] == (
            f"{name}-creds:${{RUNNER_USER_HOME:-/home/runner}}/.runner-creds"
        )
    base_env = _compose(prefix)["x-runner-base"]["environment"]
    carried = set(base_env["RUNNER_LABELS"].split(","))
    assert labels <= carried, f"{prefix} lacks {sorted(labels - carried)}"
    assert not (forbidden & carried), f"{prefix} carries {sorted(forbidden & carried)}"
    assert _compose(prefix)["x-runner-base"]["cgroup_parent"] == "omnirunners.slice"


@pytest.mark.parametrize(
    ("host", "prefix", "count", "nvme", "labels", "forbidden", "classes"), CASES
)
def test_job_directories_are_nvme_binds_not_tmpfs(
    host: str,
    prefix: str,
    count: int,
    nvme: str,
    labels: set[str],
    forbidden: set[str],
    classes: tuple[str, ...],
) -> None:
    del host, labels, forbidden, classes
    for number in range(1, count + 1):
        service = _compose(prefix)["services"][f"{prefix}-{number}"]
        assert "tmpfs" not in service
        bound = [
            volume
            for volume in service["volumes"]
            if isinstance(volume, str) and volume.startswith(f"{nvme}/")
        ]
        targets = {volume.split(":", 1)[1] for volume in bound}
        home = "${RUNNER_USER_HOME:-/home/runner}"
        assert f"{home}/actions-runner/_work" in targets
        assert str(PurePosixPath(home, "tmp")) in targets
        assert len(bound) == 8
        # one directory per runner, never shared between two
        assert all(
            volume.startswith(f"{nvme}/runners/{prefix}/{number}/") for volume in bound
        )


def test_the_pool_slice_is_the_hosts_aggregate_and_holds_the_measured_peak() -> None:
    # h051 shared unit: all three pools install the same limits.
    h051 = [
        _limits(POOLS / name / "omnirunners.slice")
        for name in ("hp051-ci-runner", "hp051-omnibase-ci-runner")
    ]
    assert {k: v for k, v in h051[0].items() if k != "Description"} == {
        k: v for k, v in h051[1].items() if k != "Description"
    }
    assert (h051[0]["MemoryHigh"], h051[0]["MemoryMax"]) == ("128G", "144G")
    # Measured peak of the shared slice before this pool (2026-11-19T17:15Z) was 50.1 GB with eight runners; the
    # sixteen new ones are budgeted 3 GiB a job (the the second amd64 host pool measured about 2.1 GiB).
    assert _gib(h051[0]["MemoryHigh"]) >= 51 + 16 * 3


def test_the_action_fleet_does_not_drop_when_the_second_host_pool_is_drained() -> None:
    config = load_runner_fleet_config(FLEET)
    # the pool being drained is the action pool of the host that is neither HP host
    drained = sum(
        pool.expected_count
        for row in config.hosts
        if row.host not in {"h051", "h053"}
        for pool in row.pools
        if "action" in pool.classes
    )
    assert drained == 16
    # make before break: the replacement is declared before the second amd64 host's pool is removed, and the total after the removal is
    # the 21 the router's degraded floor was sized on.
    assert config.declared_total("action") - drained == 21


def test_the_declared_installer_resolves_the_pool() -> None:
    host = "h051"
    prefix = "hp051-omnibase-ci-runner"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            f"--host={host}",
            f"--pool={prefix}",
            "--dry-run",
            "--skip-build",
            "--limit=1",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert f"Targets: {prefix}-1" in result.stdout
    assert f"pools/{prefix}/omnirunners.slice" in result.stdout
    # the installer's own readback list carries the limit in bytes: 128 GiB
    assert re.search(rf"MemoryHigh={128 * GIB}\b", result.stdout)
