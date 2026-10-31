# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A companion-lane failure must not roll back a healthy new dev runtime (OMN-21645).

Recorded incident, dev deploy-agent journal on the lab host, job b7df6562, 2026-10-31:
the new runtime family was up and healthy, then the gateway-forwarder step raised
``GATEWAY_DEPLOY_FAILED`` (``application not healthy after 2m0s`` at 07:07:29Z). The
boot guard treated that as a runtime failure and force-recreated the runtime on the
retained OLD image at 07:07:41Z. Thirteen of the eighteen redeploys between 19:36Z and
07:11Z ended that way, ending every in-flight delegation with ``runtime_shutdown`` and
leaving the lane behind its newest dev tag.

The gateway forwarder carries no delegation traffic, so its failure keeps the job
failed and reported, but it is not a reason to replace a runtime that is up.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from omnibase_infra.handlers import handler_dev_boot_guard as guard_mod
from omnibase_infra.models.model_dev_boot_report_config import ModelDevBootReportConfig
from omnibase_infra.models.model_dev_boot_request import ModelDevBootRequest

pytestmark = pytest.mark.unit

PROJECT = "omnibase-infra"
OLD_REVISION = "a" * 40
NEW_REVISION = "b" * 40
GATEWAY_FAILURE = (
    "GATEWAY_DEPLOY_FAILED: bash scripts/deploy-gateway.sh --execute exited 1. "
    "stderr: application not healthy after 2m0s"
)


class _Lane(Enum):
    DEV = "dev"


@contextlib.contextmanager
def _probe_file_args(lane: object, service: str) -> Iterator[list[str]]:
    yield ["-f", "compose.yml"]


def _container(
    service: str, revision: str, *, health: str = "healthy"
) -> dict[str, Any]:
    return {
        "Id": f"id-{revision[:1]}-{service}",
        "Image": f"sha256:{revision[:1]}-{service}",
        "Config": {
            "Labels": {
                "com.docker.compose.project": PROJECT,
                "com.docker.compose.service": service,
                "com.docker.compose.config-hash": f"hash-{service}",
                "org.opencontainers.image.revision": revision,
            },
            "Healthcheck": {
                "StartPeriod": 10_000_000_000,
                "Retries": 1,
                "Interval": 5_000_000_000,
                "Timeout": 5_000_000_000,
            },
        },
        "State": {"Status": "running", "Health": {"Status": health}},
    }


def _family(revision: str, *, health: str = "healthy") -> list[dict[str, Any]]:
    return [
        _container(service, revision, health=health)
        for service in guard_mod.DEV_BOOT_SERVICES
    ]


@dataclass
class _Host:
    containers: list[dict[str, Any]] = field(
        default_factory=lambda: _family(OLD_REVISION)
    )
    calls: list[list[str]] = field(default_factory=list)
    written: dict[Path, Any] = field(default_factory=dict)

    @property
    def restored(self) -> bool:
        return any("--force-recreate" in argv for argv in self.calls)

    def run(
        self, argv: list[str], timeout: int, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        del timeout, env
        self.calls.append(argv)
        verb = argv[1]
        if verb == "ps":
            ids = "\n".join(c["Id"] for c in self.containers)
            return subprocess.CompletedProcess(argv, 0, ids, "")
        if verb == "inspect":
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.containers), "")
        if verb == "logs":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "config" in argv:
            services = {s: {"image": f"img-{s}"} for s in guard_mod.DEV_BOOT_SERVICES}
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"services": services}), ""
            )
        if "up" in argv:
            # The recovery brings the retained OLD family back.
            self.containers = _family(OLD_REVISION)
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected host command: {argv}")

    def execution(self) -> ModuleType:
        module = ModuleType("fake_deploy_execution")
        module.__dict__.update(
            EnumRuntimeLane=_Lane,
            lane_config_for=lambda lane: SimpleNamespace(
                compose_project=PROJECT, disabled_services=()
            ),
            _run=self.run,
            _compose_file_args=lambda lane: ["-f", "compose.yml"],
            candidate_probe_file_args=_probe_file_args,
            _compose_env=dict,
            IMAGE_PREFLIGHT_SCRIPT="pass",
            CONFIG_HASH_TIMEOUT_SECONDS=30,
            VERIFY_RECREATE_TIMEOUT_SECONDS=30,
            _tag_image=lambda image, tag: None,
            _atomic_write_private_json=self.written.__setitem__,
        )
        return module


def _guard(host: _Host, tmp_path: Path) -> guard_mod.HandlerDevBootGuard:
    guard = guard_mod.HandlerDevBootGuard(
        tmp_path,
        config_hashes=lambda services: {s: f"hash-{s}" for s in services},
        verify_recovery=lambda update: None,
        execution=host.execution(),
        reporting=ModelDevBootReportConfig(
            registry_root=tmp_path, plugin_dir=None, ledger_via_bus=False
        ),
    )
    guard.handle(ModelDevBootRequest(operation="prepare"))
    return guard


def _incident(host: _Host, tmp_path: Path) -> dict[str, Any]:
    incidents = [v for k, v in host.written.items() if k.name.startswith("incident-")]
    assert incidents, "the guard must persist an incident record"
    incident: dict[str, Any] = incidents[-1]
    return incident


def _deploy_new_runtime_then(host: _Host, failure: Exception) -> None:
    """The candidate runtime replaced the old one and is healthy, then a step fails."""
    host.containers = _family(NEW_REVISION)
    raise failure


def test_companion_lane_failure_keeps_the_healthy_new_runtime(tmp_path: Path) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)

    with pytest.raises(guard_mod.DevBootRecoveryError, match="GATEWAY_DEPLOY_FAILED"):
        with guard.protect("job-b7df6562", "c" * 40, lambda error: None):
            _deploy_new_runtime_then(
                host, guard_mod.DevCompanionLaneFailedError(GATEWAY_FAILURE)
            )

    assert not host.restored, "a healthy new runtime must not be force-recreated"
    assert {
        c["Config"]["Labels"]["org.opencontainers.image.revision"]
        for c in host.containers
    } == {NEW_REVISION}
    assert guard.point is not None
    assert guard.point.revision == NEW_REVISION
    incident = _incident(host, tmp_path)
    assert incident["status"] == "failed"
    assert incident["recovery_verified"] is True
    assert incident["runtime_kept_revision"] == NEW_REVISION


def test_companion_lane_failure_restores_when_the_new_runtime_is_unhealthy(
    tmp_path: Path,
) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)

    def deploy_unhealthy_then_fail() -> None:
        host.containers = _family(NEW_REVISION, health="unhealthy")
        raise guard_mod.DevCompanionLaneFailedError(GATEWAY_FAILURE)

    with pytest.raises(guard_mod.DevBootRecoveryError):
        with guard.protect("job-x", "c" * 40, lambda error: None):
            deploy_unhealthy_then_fail()

    assert host.restored
    assert guard.point is not None
    assert guard.point.revision == OLD_REVISION


def test_runtime_failure_still_restores_a_healthy_new_runtime(tmp_path: Path) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)

    with pytest.raises(guard_mod.DevBootRecoveryError):
        with guard.protect("job-y", "c" * 40, lambda error: None):
            _deploy_new_runtime_then(
                host, RuntimeError("verify: runtime answered 503 to the smoke probe")
            )

    assert host.restored
    assert guard.point is not None
    assert guard.point.revision == OLD_REVISION
