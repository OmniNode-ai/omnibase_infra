# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An onex-api step failure must not roll back a healthy new dev runtime (OMN-24471).

Live incident, dev deploy-agent journal on the lab host, 2026-11-14: the onex-api
image crash-looped on an import error, so the roll stopped at the onex-api step.
The guard read that as a runtime failure and force-recreated the up-and-healthy
``omninode-runtime`` family on the retained OLD image three times, at 17:26Z,
17:47Z and 18:03Z.

onex-api is not the runtime (it is not in ``DEV_BOOT_SERVICES``), so its failure
keeps the job failed and reported, but it is not a reason to replace a runtime
that is up. The carve-out copies OMN-21645's companion-lane shape.
"""

from __future__ import annotations

import subprocess
from enum import Enum
from pathlib import Path
from types import ModuleType

import pytest

from omnibase_infra.handlers import handler_dev_boot_guard as guard_mod
from tests.unit.handlers.test_handler_dev_boot_guard_companion_lane import (
    NEW_REVISION,
    OLD_REVISION,
    _family,
    _guard,
    _Host,
    _incident,
)

pytestmark = pytest.mark.unit

ONEX_API_FAILURE = (
    "Runtime roll stopped at onex-api: Waiting container onex-api is unhealthy"
)


class _Lane(Enum):
    DEV = "dev"


def _roll_execution() -> ModuleType:
    """The execution boundary run_runtime_roll_step reads, reduced to its members."""
    module = ModuleType("fake_roll_execution")
    module.__dict__.update(
        EnumRuntimeLane=_Lane,
        ONEX_API_SERVICE="onex-api",
        runtime_health_probe_budget=lambda lane, service: 5,
        RUNTIME_HEALTH_PROBE_TIMEOUT_SECONDS=5,
        CONTAINER_VERIFY_TIMEOUT_SECONDS=5,
        subprocess=subprocess,
        _compose_env=lambda extra=None: {},
        _run=lambda argv, timeout, env=None: subprocess.CompletedProcess(
            argv, 1, "", "Waiting container onex-api is unhealthy"
        ),
        verify_containers_up=lambda services, timeout_s, lane: (True, []),
    )
    return module


def test_onex_api_step_failure_keeps_the_healthy_new_runtime(tmp_path: Path) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)

    with pytest.raises(guard_mod.DevBootRecoveryError, match="onex-api is unhealthy"):
        with guard.protect("job-onexapi", "c" * 40, lambda error: None):
            host.containers = _family(NEW_REVISION)
            raise guard_mod.DevOnexApiStepFailedError(ONEX_API_FAILURE, "onex-api")

    assert not host.restored, "a healthy new runtime must not be force-recreated"
    assert {
        c["Config"]["Labels"]["org.opencontainers.image.revision"]
        for c in host.containers
    } == {NEW_REVISION}
    assert guard.point.current is not None
    assert guard.point.current.revision == NEW_REVISION
    incident = _incident(host, tmp_path)
    assert incident["status"] == "failed"
    assert incident["recovery_verified"] is True
    assert incident["runtime_kept_revision"] == NEW_REVISION


@pytest.mark.parametrize("runtime_state", ["unhealthy", "absent"])
def test_onex_api_step_failure_restores_when_new_runtime_is_unhealthy_or_absent(
    tmp_path: Path, runtime_state: str
) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)

    with pytest.raises(guard_mod.DevBootRecoveryError, match="onex-api is unhealthy"):
        with guard.protect("job-onexapi-down", NEW_REVISION, lambda error: None):
            host.containers = _family(NEW_REVISION)
            if runtime_state == "absent":
                host.containers = []
            else:
                host.containers[0]["State"]["Health"]["Status"] = "unhealthy"
            raise guard_mod.DevOnexApiStepFailedError(ONEX_API_FAILURE, "onex-api")

    assert host.restored
    assert guard.point.current is not None
    assert guard.point.current.revision == OLD_REVISION
    incident = _incident(host, tmp_path)
    assert incident["status"] == "failed"
    assert incident["recovery_verified"] is True
    assert "runtime_kept_revision" not in incident


def test_runtime_step_failure_still_restores(tmp_path: Path) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)

    with pytest.raises(guard_mod.DevBootRecoveryError):
        with guard.protect("job-runtime", "c" * 40, lambda error: None):
            host.containers = _family(NEW_REVISION)
            raise guard_mod.DevBootFailedError(
                "Runtime roll stopped at omninode-runtime: "
                "Waiting container omninode-runtime is unhealthy"
            )

    assert host.restored, (
        "a plain runtime step failure must still restore the retained family"
    )


def test_run_runtime_roll_step_types_the_onex_api_failure() -> None:
    execution = _roll_execution()

    with pytest.raises(guard_mod.DevOnexApiStepFailedError):
        guard_mod.run_runtime_roll_step(
            ["docker", "compose", "up", "-d", "--no-deps", "onex-api"],
            30,
            execution=execution,
        )

    with pytest.raises(guard_mod.DevBootFailedError) as runtime_failure:
        guard_mod.run_runtime_roll_step(
            ["docker", "compose", "up", "-d", "--no-deps", "omninode-runtime"],
            30,
            execution=execution,
        )
    assert not isinstance(runtime_failure.value, guard_mod.DevOnexApiStepFailedError), (
        "a runtime step failure must stay a plain DevBootFailedError"
    )


def test_onex_api_step_failure_leaves_a_healthy_partial_roll_running(
    tmp_path: Path,
) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)

    with pytest.raises(guard_mod.DevBootRecoveryError, match="onex-api is unhealthy"):
        with guard.protect("job-partial", "c" * 40, lambda error: None):
            # The roll replaced the first service and stopped at onex-api, so
            # the family reads as two revisions and cannot become a recovery point.
            host.containers = [
                *_family(NEW_REVISION)[:1],
                *_family(OLD_REVISION)[1:],
            ]
            raise guard_mod.DevOnexApiStepFailedError(ONEX_API_FAILURE, "onex-api")

    assert not host.restored, "a healthy runtime must not be force-recreated"
    assert guard.point.current is not None
    assert guard.point.current.revision == OLD_REVISION
    incident = _incident(host, tmp_path)
    assert incident["status"] == "failed"
    assert incident["recovery_verified"] is True
    assert incident["runtime_left_running"] is True
    assert "runtime_kept_revision" not in incident


def test_runtime_step_failure_restores_a_partial_roll(tmp_path: Path) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)

    with pytest.raises(guard_mod.DevBootRecoveryError):
        with guard.protect("job-partial-runtime", "c" * 40, lambda error: None):
            host.containers = [
                *_family(NEW_REVISION)[:1],
                *_family(OLD_REVISION)[1:],
            ]
            raise guard_mod.DevBootFailedError(
                "Runtime roll stopped at runtime-worker: container did not reach "
                "running state"
            )

    assert host.restored
    incident = _incident(host, tmp_path)
    assert "runtime_left_running" not in incident


def test_roll_expected_drops_only_the_failed_onex_api_service() -> None:
    expected = ["omninode-runtime", "onex-api", "projection-api"]
    typed = guard_mod.DevOnexApiStepFailedError(ONEX_API_FAILURE, "onex-api")

    assert guard_mod.roll_expected(expected, typed) == [
        "omninode-runtime",
        "projection-api",
    ]
    assert (
        guard_mod.roll_expected(
            expected,
            guard_mod.DevBootFailedError("Runtime roll stopped at projection-api"),
        )
        == expected
    )
    assert guard_mod.roll_expected(expected, None) == expected
