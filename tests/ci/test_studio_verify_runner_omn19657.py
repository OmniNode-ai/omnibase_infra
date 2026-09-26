# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19657 -- the .200 verify runner is where the dev-200 verify job lands.

Mirrors ``test_omnipc2_verify_runner_omn19507.py``: the dev-200 instance (once
its ``verify:`` block lands, OMN-19543/OMN-19507) sends its per-merge verify
job to ``[self-hosted, omnibase-verify, host-200]``. That job reads the
dev-200 lane's docker daemon and, through ``host.docker.internal``, the ports
the lane publishes on the docker bridge gateway (loopback-only on Docker
Desktop). So the runner that satisfies those labels must:

* carry every one of those labels, and be the only declared runner carrying
  ``host-200``;
* never carry ``omnibase-ci`` (the action fleet) or ``omnibase-deploy``-only
  duties that are not this lane's, so no other job class lands on .200;
* mount the docker socket and carry the host-gateway alias, and nothing that
  hands a job a write surface or a credential (no operator env, no clone
  tree, no lab kubeconfig -- this runner is lane-pinned, not a general
  cloud-touching verify host).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.ci

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
COMPOSE = DOCKER / "docker-compose.runners-omninode-studio-runner.yml"
SERVICE = "omninode-studio-runner-1"
DEV_200_VERIFY_LABELS = {"self-hosted", "omnibase-verify", "host-200"}
FORBIDDEN_LABELS = {"omnibase-ci", "omnibase-deploy", "omnipc2-customer"}


def _services(path: Path) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("services") or {}


def _labels(service: dict[str, Any]) -> set[str]:
    env = service.get("environment") or {}
    return set(str(env.get("RUNNER_LABELS", "")).split(",")) - {""}


def _runner() -> dict[str, Any]:
    services = _services(COMPOSE)
    assert SERVICE in services, sorted(services)
    return services[SERVICE]


def test_studio_verify_runner_carries_the_dev_200_verify_labels() -> None:
    labels = _labels(_runner())
    assert labels >= DEV_200_VERIFY_LABELS, sorted(labels)
    assert not (labels & FORBIDDEN_LABELS), sorted(labels & FORBIDDEN_LABELS)


def test_studio_verify_runner_carries_its_own_architecture_labels() -> None:
    # M2 Ultra is arm64; a runner claiming the wrong architecture label hands
    # arch-pinned jobs to a CPU that cannot run them.
    labels = _labels(_runner())
    assert {"arm64", "arch-arm64"} <= labels, sorted(labels)
    assert "amd64" not in labels and "arch-amd64" not in labels, sorted(labels)


def test_studio_verify_runner_is_the_only_runner_labelled_host_200() -> None:
    carriers = []
    for path in sorted(DOCKER.glob("docker-compose.runners*.yml")):
        for name, service in _services(path).items():
            if "host-200" in _labels(service or {}):
                carriers.append(f"{path.name}:{name}")
    assert carriers == [f"{COMPOSE.name}:{SERVICE}"], (
        f"host-200 must name exactly one declared runner, found {carriers}"
    )


def test_studio_verify_runner_reads_the_lane_and_nothing_more() -> None:
    runner = _runner()
    volumes = [str(v) for v in runner.get("volumes") or []]
    assert "/var/run/docker.sock:/var/run/docker.sock" in volumes
    assert "host.docker.internal:host-gateway" in (runner.get("extra_hosts") or [])
    assert (runner.get("environment") or {}).get(
        "LANE_PROBE_HOST"
    ) == "host.docker.internal"
    joined = "\n".join(volumes)
    for absent in ("operator.env", "OMNI_HOME", "lab-credentials", ".omnibase/.env"):
        assert absent not in joined, f"{absent!r} must not be mounted: {volumes}"
    # Capped like the .202 lane-pinned runner: this host also serves the
    # always-on planner and the dev-200 lane.
    assert runner.get("cpus") == "1.0"
    assert runner.get("mem_limit") == "2g"


def test_studio_verify_runner_declares_no_group_add() -> None:
    # macOS/Docker Desktop difference from the primary (Linux) host, same as
    # the .101/.105 files: the entrypoint's root phase fixes the socket GID,
    # so a static group_add here would be wrong.
    assert "group_add" not in _runner()


def test_the_dev_200_route_labels_have_a_runner_to_land_on() -> None:
    """Positive control across the seam: once the routing table declares the
    dev-200 instance's verify block, its labels are satisfiable by this
    runner. Skips today because that block has not landed yet (OMN-19543/
    OMN-19507); this is the seam test, not the trigger to land it."""
    routing = ROOT / "config" / "deploy_lane_routing.yaml"
    if not routing.is_file():
        pytest.skip("the routing table is not on this base yet")
    table = yaml.safe_load(routing.read_text(encoding="utf-8")) or {}
    instance = (table.get("instances") or {}).get("dev-200")
    if instance is None or "verify" not in instance:
        pytest.skip("the routing table declares no dev-200 verify block yet")
    wanted = set(instance["verify"]["runner_labels"])
    assert wanted <= _labels(_runner()), (sorted(wanted), sorted(_labels(_runner())))
