# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability follow's verification probe and the image it probes (OMN-25942).

Replays the three failed stability follow jobs of 2026-11-19 (1f7f39c6, 6ff2c3f5,
914b6b79) from ``fixtures/omn25942/stability_follow_jobs_verification_probe.json``. Each
probed ``:18085`` and ``:18086`` after a recreate and read ``Connection reset by peer``
from a runtime built as an INFRA-ONLY image, which has no ``BIFROST_BASE_CONTRACT_PATH``
and exits at boot. The verification itself probed only the stability lane's own
container; the defect is the image the stability overlay builds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "omn25942"
    / "stability_follow_jobs_verification_probe.json"
)
RECORDED = json.loads(_FIXTURE.read_text())
_REPO = Path(__file__).resolve().parents[4]
_COMPOSE_TAG = re.compile(r"(?m)(:\s+|-\s+)!(?:override|reset)\b[ \t]*")
_URL = re.compile(r"http://localhost:(\d+)/health \(([^)]+)\)")


def _stability_services() -> dict[str, dict[str, Any]]:
    config = executor_mod.lane_config_for(EnumRuntimeLane.STABILITY_TEST)
    overlay = Path(config.compose_files[-1]).name
    text = (_REPO / "docker" / overlay).read_text(encoding="utf-8")
    loaded = yaml.safe_load(_COMPOSE_TAG.sub(r"\1", text))
    return dict(loaded["services"])


def test_verification_probe_reads_only_the_stability_lanes_own_runtime() -> None:
    config = executor_mod.lane_config_for(EnumRuntimeLane.STABILITY_TEST)
    own_ports = {port for _, port in config.runtime_health_targets}
    own_containers = {name for name, _ in config.runtime_health_targets}
    runtime_container = config.runtime_health_targets[0][0]
    error = RECORDED["common"]["error"].replace(
        "{runtime_container}", runtime_container
    )
    probed = _URL.findall(error)
    assert {int(port) for port, _ in probed} == {18085, 18086}
    assert {int(port) for port, _ in probed} <= own_ports
    assert {name for _, name in probed} <= own_containers
    for lane in (EnumRuntimeLane.DEV, EnumRuntimeLane.PROD):
        other = executor_mod.lane_config_for(lane)
        assert own_ports.isdisjoint(port for _, port in other.runtime_health_targets)
        assert own_containers.isdisjoint(
            name for name, _ in other.runtime_health_targets
        )


def test_verification_probe_waits_for_a_recreated_stability_runtime_to_start() -> None:
    assert EnumRuntimeLane.STABILITY_TEST in executor_mod.VERIFY_START_WAIT_LANES


def test_verification_probe_recorded_failures_follow_infra_only_images() -> None:
    readings = RECORDED["image_readings"]
    assert (
        readings["rolled_back_image_53e880e38d5a"][
            "BIFROST_BASE_CONTRACT_PATH_in_image_env"
        ]
        == 1
    )
    for image in readings["follow_built_omninode_runtime_images"]:
        assert image["BIFROST_BASE_CONTRACT_PATH_in_image_env"] == 0
        assert image["omnimarket_vcs_ref_label"] == ""
    assert (
        readings["follow_built_runtime_worker_image"][
            "BIFROST_BASE_CONTRACT_PATH_in_image_env"
        ]
        == 0
    )
    log = " ".join(RECORDED["worker_container"]["log_lines"])
    assert "BIFROST_BASE_CONTRACT_PATH is unset or blank" in log
    assert "RETRIES EXHAUSTED" in log


@pytest.mark.parametrize("service", ["omninode-runtime", "runtime-worker"])
def test_verification_probe_stability_runtime_images_are_built_composed(
    service: str,
) -> None:
    services = _stability_services()
    build = services[service].get("build") or {}
    assert build.get("dockerfile") == "Dockerfile.runtime", (
        f"{service}: the stability overlay builds the infra-only image, which has no "
        "BIFROST_BASE_CONTRACT_PATH, so the container exits at boot"
    )
    assert build["context"].endswith("sibling-repos/omnimarket")
    context_service = build["additional_contexts"]["infra-runtime"].removeprefix(
        "service:"
    )
    assert build["additional_contexts"] == {
        "infra-runtime": f"service:{context_service}"
    }
    assert build["args"] == {"INFRA_RUNTIME_IMAGE": "infra-runtime"}
    infra = services[context_service]
    assert infra["build"]["dockerfile"] == "docker/Dockerfile.runtime"
    assert "runtime" in infra["profiles"]
    assert not any(name.endswith("-infra-runtime-image") for name in services), (
        "the stack-shape baseline for *-infra-runtime-image only shrinks"
    )
