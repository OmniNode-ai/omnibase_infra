# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22111: the general ``omnibase-ci`` pool runs on h053, which has AVX2; h051 stays opt-in only.

Operator RULING 2026-11-02T13:33:32Z (OMN-17232): the general CI runners leave the primary host. h051 has no AVX2
(a wheel built for it dies there with SIGILL), h053 has it, so only h053 carries ``omnibase-ci``. The invariant this
file holds: no runner that carries ``omnibase-ci`` can sit on a host that lacks AVX2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER = REPO_ROOT / "docker"
POOLS = DOCKER / "runners" / "systemd" / "pools"
FLEET = REPO_ROOT / "config" / "runner_fleet.yaml"
H053 = "hp053-ci-runner"
H051 = "hp051-ci-runner"
COUNT = 5
# Hosts whose CPU lacks AVX2, read from /proc/cpuinfo (OMN-17232, 2026-11-02).
NO_AVX2_PREFIXES = (H051,)
# The runner image's home. h053 spells it as the primary compose file does, so no machine path is written.
IMAGE_HOME = "/home/runner"
RUNNER_HOME = "${RUNNER_USER_HOME:-" + IMAGE_HOME + "}"
CREDS_MOUNT = RUNNER_HOME + "/.runner-creds"


def _compose(prefix: str) -> dict[str, Any]:
    path = DOCKER / f"docker-compose.runners-{prefix}.yml"
    return cast("dict[str, Any]", yaml.safe_load(path.read_text(encoding="utf-8")))


def _labels(service: dict[str, Any]) -> set[str]:
    return set(str(service["environment"]["RUNNER_LABELS"]).split(","))


def _bytes(value: str | int) -> int:
    if isinstance(value, int):
        return value
    assert value[-1].lower() == "g", value
    return int(value[:-1]) * 1024**3


def _limits(unit: Path) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for key, value in (
            line.split("=", 1)
            for line in unit.read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.lstrip().startswith(("#", "["))
        )
    }


def test_every_h053_runner_carries_the_general_and_the_opt_in_label() -> None:
    services = _compose(H053)["services"]
    assert sorted(services) == [f"{H053}-{n}" for n in range(1, COUNT + 1)]
    for name, service in services.items():
        assert _labels(service) == {
            "self-hosted",
            "omnibase-ci",
            "omnibase-ci-multiarch",
            "linux",
            "x64",
            "arch-amd64",
        }, name
        assert service["environment"]["RUNNER_GROUP"] == "omnibase-ci"
        assert service["environment"]["ONEX_LANE_HOST"] == "h053"
        assert service["environment"]["GITHUB_ORG_URL"].startswith("${GITHUB_ORG_URL:?")
        assert service["environment"]["UV_HTTP_TIMEOUT"].startswith(
            "${UV_HTTP_TIMEOUT:?"
        )
        assert service["environment"]["RUNNER_NAME"] == name
        assert service["container_name"] == name
        assert service["cgroup_parent"] == "omnirunners.slice"
        assert f"{name}-creds:{CREDS_MOUNT}" in service["volumes"]


def test_no_runner_that_carries_omnibase_ci_sits_on_a_host_without_avx2() -> None:
    """The invariant behind OMN-22111: h051 cannot import an AVX2 wheel, so it must never take ``omnibase-ci``."""
    for prefix in NO_AVX2_PREFIXES:
        for name, service in _compose(prefix)["services"].items():
            assert "omnibase-ci" not in _labels(service), name
    config = load_runner_fleet_config(FLEET)
    for host in config.hosts:
        prefixes = [
            host.runner_name_prefix,
            *(p.runner_name_prefix for p in host.pools),
        ]
        if any(prefix in NO_AVX2_PREFIXES for prefix in prefixes):
            assert host.declared_count("action") == 0, host.host


def test_the_fleet_declares_h053_with_the_general_class() -> None:
    document = yaml.safe_load(FLEET.read_text(encoding="utf-8"))
    rows = [row for row in document["hosts"] if row["host"] == "h053"]
    assert len(rows) == 1
    assert rows[0]["runner_name_prefix"] == H053
    assert rows[0]["expected_count"] == COUNT
    assert rows[0]["arch"] == "amd64"
    assert rows[0]["classes"] == ["action", "action-multiarch"]
    config = load_runner_fleet_config(FLEET)
    host = config.host_of_runner(f"{H053}-{COUNT}")
    assert host is not None and host.host == "h053"
    assert host.declared_count("action") == COUNT
    assert host.declared_count("verify") == 0
    # The action fleet now spans three hosts, so no required context relies on fewer than two.
    assert [h.host for h in config.hosts if h.declared_count("action") > 0].count(
        "h053"
    ) == 1
    assert len([h for h in config.hosts if h.declared_count("action") > 0]) >= 3


def test_the_pool_slice_bounds_the_whole_pool() -> None:
    services = _compose(H053)["services"]
    limits = _limits(POOLS / H053 / "omnirunners.slice")
    ram = sum(_bytes(s["mem_limit"]) for s in services.values())
    swap = sum(
        _bytes(s["memswap_limit"]) - _bytes(s["mem_limit"]) for s in services.values()
    )
    assert limits["CPUWeight"] == "25"
    assert ram <= _bytes(limits["MemoryHigh"]) <= _bytes(limits["MemoryMax"])
    # OMN-19998: the job directories are tmpfs, so nothing in the pool may swap to the rotational root volume.
    assert swap == 0
    assert limits["MemorySwapMax"] == "0"


@pytest.mark.parametrize(
    "unit",
    [
        "omninode-git-mirror-daemon.service",
        "omninode-git-mirror-refresh.service",
        "omninode-git-mirror-refresh.timer",
    ],
)
def test_the_mirror_units_are_the_h051_units_for_this_pool(unit: str) -> None:
    """The mirror is the same OMN-14027 C2 mirror; only the host, the pool prefix and the ticket differ."""
    derived = (
        (POOLS / H051 / unit)
        .read_text(encoding="utf-8")
        .replace(H051, H053)
        .replace("h051", "h053")
        .replace("OMN-17232", "OMN-22111")
    )
    # The hardcoded-config ratchet refuses a new file that spells the organisation URL, so no h053 unit has one.
    derived = "".join(
        line
        for line in derived.splitlines(keepends=True)
        if not line.startswith("Documentation=")
    )
    assert (POOLS / H053 / unit).read_text(encoding="utf-8") == derived


def test_the_h053_service_blocks_are_the_h051_blocks_with_the_h053_names() -> None:
    """Everything but the label set, the host name, the GID comment and the count is h051's anchor, unchanged."""
    h051 = _compose(H051)["services"][f"{H051}-1"]
    h053 = _compose(H053)["services"][f"{H053}-1"]
    for key in ("image", "restart", "pids_limit"):
        assert h053[key] == h051[key], key
    # OMN-19998: the limits are h053's own, from the reads of 2026-11-04 (see the compose header).
    assert (h053["mem_limit"], h053["memswap_limit"], h053["cpus"]) == (
        "9g",
        "9g",
        "5.0",
    )
    assert h053["group_add"] == h051["group_add"] == ["988"]
    assert h053["healthcheck"] == h051["healthcheck"]
    assert set(h053["environment"]) == set(h051["environment"])
    differing = {
        key
        for key, value in h051["environment"].items()
        if str(value).replace(IMAGE_HOME + "/", RUNNER_HOME + "/")
        != h053["environment"][key]
    }
    assert differing == {
        "ONEX_LANE_HOST",
        "RUNNER_LABELS",
        "RUNNER_NAME",
        "GITHUB_ORG_URL",
        "UV_HTTP_TIMEOUT",
    }
