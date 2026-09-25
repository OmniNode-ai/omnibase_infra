# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19507 AC3 -- the .202 verify runner is where the dev-202 verify job lands.

The dev-202 instance in ``config/deploy_lane_routing.yaml`` sends its per-merge
verify job to ``[self-hosted, omnibase-verify, host-202]``. That job reads the
dev-202 lane's docker daemon and, through ``host.docker.internal``, the ports
the lane publishes on the docker bridge gateway. So the runner that satisfies
those labels must:

* carry every one of those labels, and be the only declared runner carrying
  ``host-202``;
* never carry ``omnipc2-customer`` (the clean-machine producers' runner) nor
  ``omnibase-ci`` (the action fleet), so no other job class lands on .202;
* mount the docker socket and carry the host-gateway alias, and nothing that
  hands a job a write surface or a credential (no operator env, no clone tree).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.ci

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
COMPOSE = DOCKER / "docker-compose.runners-omnipc2-verify-runner.yml"
SERVICE = "omnipc2-verify-runner-1"
DEV_202_VERIFY_LABELS = {"self-hosted", "omnibase-verify", "host-202"}
FORBIDDEN_LABELS = {"omnipc2-customer", "omnibase-ci", "omnibase-deploy"}


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


def test_omnipc2_verify_runner_carries_the_dev_202_verify_labels() -> None:
    labels = _labels(_runner())
    assert labels >= DEV_202_VERIFY_LABELS, sorted(labels)
    assert not (labels & FORBIDDEN_LABELS), sorted(labels & FORBIDDEN_LABELS)


def test_omnipc2_verify_runner_is_the_only_runner_labelled_host_202() -> None:
    carriers = []
    for path in sorted(DOCKER.glob("docker-compose.runners*.yml")):
        for name, service in _services(path).items():
            if "host-202" in _labels(service or {}):
                carriers.append(f"{path.name}:{name}")
    assert carriers == [f"{COMPOSE.name}:{SERVICE}"], (
        f"host-202 must name exactly one declared runner, found {carriers}"
    )


def test_omnipc2_verify_runner_reads_the_lane_and_nothing_more() -> None:
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
    # Capped like omninode-verify-runner-1: this host serves the model and two
    # lanes.
    assert runner.get("cpus") == "1.0"
    assert runner.get("mem_limit") == "2g"


def test_the_dev_202_route_labels_have_a_runner_to_land_on() -> None:
    """Positive control across the seam: when the routing table declares the
    dev-202 instance, its verify labels are satisfiable by this runner."""
    routing = ROOT / "config" / "deploy_lane_routing.yaml"
    if not routing.is_file():
        pytest.skip("the routing table is not on this base yet (OMN-19506)")
    table = yaml.safe_load(routing.read_text(encoding="utf-8")) or {}
    instance = (table.get("instances") or {}).get("dev-202")
    if instance is None or "verify" not in instance:
        pytest.skip("the routing table declares no dev-202 verify block yet")
    wanted = set(instance["verify"]["runner_labels"])
    assert wanted <= _labels(_runner()), (sorted(wanted), sorted(_labels(_runner())))
