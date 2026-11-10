# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed dev roll keeps Docker's probe reading on its incident (OMN-23431).

Recorded incident, dev deploy-agent job 0135046e, 2026-11-10: compose reported
"Waiting container omninode-runtime is unhealthy" at 04:10:37Z, 171 s after the new
container started under a 1800 s start period. The boot guard then recreated the
container on the retained image, which destroyed its health log, and the incident kept
only compose's stderr. Nothing on the record said which probe failed or why.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
PROBE_OUTPUT = "FAIL [http_error] health endpoint returned HTTP 503"
ROLL_FAILURE = (
    "Runtime roll stopped at omninode-runtime: container omninode-runtime is unhealthy"
)


class _Lane(Enum):
    DEV = "dev"


@contextlib.contextmanager
def _probe_file_args(lane: object, service: str) -> Iterator[list[str]]:
    yield ["-f", "compose.yml"]


def _container(service: str, revision: str) -> dict[str, Any]:
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
                "StartPeriod": 1_800_000_000_000,
                "Retries": 5,
                "Interval": 30_000_000_000,
                "Timeout": 10_000_000_000,
            },
        },
        "State": {"Status": "running", "Health": {"Status": "healthy"}},
    }


def _family(revision: str) -> list[dict[str, Any]]:
    return [_container(s, revision) for s in guard_mod.DEV_BOOT_SERVICES]


def _unhealthy_runtime_family() -> list[dict[str, Any]]:
    family = _family(NEW_REVISION)
    runtime = next(
        c
        for c in family
        if c["Config"]["Labels"]["com.docker.compose.service"] == "omninode-runtime"
    )
    runtime["RestartCount"] = 0
    runtime["State"] = {
        "Status": "running",
        "StartedAt": "2026-11-10T04:07:46Z",
        "ExitCode": 0,
        "OOMKilled": False,
        "Health": {
            "Status": "unhealthy",
            "FailingStreak": 5,
            "Log": [
                {
                    "Start": f"2026-11-10T04:{minute}Z",
                    "End": f"2026-11-10T04:{minute}Z",
                    "ExitCode": 1,
                    "Output": f"{PROBE_OUTPUT}\n",
                }
                for minute in ("08:07", "08:37", "09:07", "09:37", "10:07")
            ],
        },
    }
    return family


@dataclass
class _Host:
    containers: list[dict[str, Any]] = field(
        default_factory=lambda: _family(OLD_REVISION)
    )
    written: dict[Path, Any] = field(default_factory=dict)
    logs_fail: bool = False

    def run(
        self, argv: list[str], timeout: int, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        del timeout, env
        verb = argv[1]
        if verb == "ps":
            ids = "\n".join(c["Id"] for c in self.containers)
            return subprocess.CompletedProcess(argv, 0, ids, "")
        if verb == "inspect":
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.containers), "")
        if verb == "logs":
            return subprocess.CompletedProcess(argv, 1 if self.logs_fail else 0, "", "")
        if "config" in argv:
            services = {s: {"image": f"img-{s}"} for s in guard_mod.DEV_BOOT_SERVICES}
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"services": services}), ""
            )
        if "up" in argv:
            self.containers = _family(OLD_REVISION)
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected host command: {argv}")

    def execution(self) -> ModuleType:
        module = ModuleType("fake_deploy_execution")
        module.__dict__.update(
            EnumRuntimeLane=_Lane,
            subprocess=subprocess,
            _dev_boot_clock=lambda: 0.0,
            _dev_boot_timestamp=lambda: datetime(2026, 11, 10, tzinfo=UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            _read_private_text=lambda path: path.read_text(encoding="utf-8"),
            lane_config_for=lambda lane: SimpleNamespace(
                compose_project=PROJECT, disabled_services=()
            ),
            lane_folded_profiles=lambda lane: frozenset(),
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


def _fail_roll(host: _Host, tmp_path: Path) -> tuple[dict[str, Any], list[str]]:
    guard = _guard(host, tmp_path)
    failed: list[str] = []
    with pytest.raises(guard_mod.DevBootRecoveryError, match=r"HTTP 503"):
        with guard.protect("job-0135046e", NEW_REVISION, failed.append):
            host.containers = _unhealthy_runtime_family()
            raise RuntimeError(ROLL_FAILURE)
    incidents = [v for k, v in host.written.items() if k.name.startswith("incident-")]
    assert incidents, "the guard must persist an incident record"
    return incidents[-1], failed


def _runtime_reading(incident: dict[str, Any]) -> dict[str, Any]:
    readings = [
        r for r in incident["health_probes"] if r["service"] == "omninode-runtime"
    ]
    assert len(readings) == 1
    reading: dict[str, Any] = readings[0]
    return reading


def test_failed_roll_incident_keeps_the_runtime_probe_reading(tmp_path: Path) -> None:
    host = _Host()
    incident, failed = _fail_roll(host, tmp_path)

    reading = _runtime_reading(incident)
    assert reading["health_status"] == "unhealthy"
    assert reading["failing_streak"] == 5
    assert reading["started_at"] == "2026-11-10T04:07:46Z"
    assert reading["restart_count"] == 0
    assert reading["oom_killed"] is False
    assert reading["healthcheck"]["start_period_ns"] == 1_800_000_000_000
    assert [p["exit_code"] for p in reading["probes"]] == [1] * 5
    assert all(p["output"] == PROBE_OUTPUT for p in reading["probes"])
    assert PROBE_OUTPUT in incident["error"]
    assert "omninode-runtime health=unhealthy streak=5" in incident["error"]
    assert failed and PROBE_OUTPUT in failed[0]
    healthy = [s for s in guard_mod.DEV_BOOT_SERVICES if s != "omninode-runtime"]
    assert all(f"{s} health=" not in incident["error"] for s in healthy)


def test_unreadable_logs_still_keep_the_probe_reading(tmp_path: Path) -> None:
    host = _Host(logs_fail=True)
    incident, _ = _fail_roll(host, tmp_path)

    assert _runtime_reading(incident)["failing_streak"] == 5
    assert PROBE_OUTPUT in incident["error"]


def test_reading_of_a_container_docker_reported_nothing_for() -> None:
    reading = guard_mod.health_probe_reading(
        {"Id": "c" * 64, "Image": "img", "Config": {"Labels": {}}}
    )

    assert reading["container"] == "c" * 12
    assert reading["service"] is None
    assert reading["health_status"] is None
    assert reading["probes"] == []
    assert reading["healthcheck"] == {
        "start_period_ns": None,
        "interval_ns": None,
        "retries": None,
        "timeout_ns": None,
    }
    assert guard_mod.health_probe_summary(reading) == (
        "None health=None streak=None started=None restarts=None oom=None "
        "last_probe=none"
    )


def test_healthy_container_adds_no_summary() -> None:
    reading = guard_mod.health_probe_reading(
        _container("omninode-runtime", NEW_REVISION)
    )

    assert guard_mod.health_probe_summary(reading) is None
