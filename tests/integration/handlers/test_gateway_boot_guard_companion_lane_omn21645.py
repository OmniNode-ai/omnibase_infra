# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Gateway executor failures cross the boot guard's recovery boundary (OMN-21645).

The executor and guard are real. Only host commands and gateway host configuration
are substituted; no Docker daemon or declared lane is touched.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import BuildSource, EnumRuntimeLane

from omnibase_infra.handlers import handler_dev_boot_guard as guard_mod
from tests.unit.handlers.test_handler_dev_boot_guard_companion_lane import (
    NEW_REVISION,
    OLD_REVISION,
    _family,
    _guard,
    _Host,
    _incident,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("gateway_failure", ["exit", "timeout"])
@pytest.mark.parametrize("runtime_state", ["healthy", "unhealthy", "absent"])
def test_gateway_failure_keeps_or_restores_the_runtime(
    tmp_path: Path, gateway_failure: str, runtime_state: str
) -> None:
    host = _Host()
    guard = _guard(host, tmp_path)
    executor = executor_mod.DeployExecutor()
    gateway_env = tmp_path / "gateway.env"
    gateway_env.write_text("")
    script = str(Path(executor_mod.__file__).resolve().parents[2] / "deploy-gateway.sh")
    budget = SimpleNamespace(timeout_seconds=120, describe=lambda: "test host ceiling")
    failure = (
        subprocess.TimeoutExpired(["bash", script, "--execute"], 120)
        if gateway_failure == "timeout"
        else None
    )
    diagnostic = (
        "GATEWAY_DEPLOY_TIMED_OUT"
        if gateway_failure == "timeout"
        else "GATEWAY_DEPLOY_FAILED"
    )
    failed_jobs: list[str] = []

    with (
        patch.object(executor_mod, "lane_runs_phase", return_value=True),
        patch.object(executor_mod, "deploy_gateway_script", return_value=script),
        patch.object(
            executor, "_assert_gateway_lane_config", return_value=str(gateway_env)
        ),
        patch.object(executor_mod, "gateway_deploy_budget", return_value=budget),
        patch.object(executor_mod, "_restore_build_byproducts") as restore_byproducts,
        patch.object(
            executor_mod,
            "_run",
            side_effect=failure,
            return_value=subprocess.CompletedProcess(
                ["bash", script, "--execute"],
                1,
                "",
                "application not healthy after 2m0s",
            ),
        ) as gateway_command,
        pytest.raises(guard_mod.DevBootRecoveryError, match=diagnostic) as recovery,
    ):
        with guard.protect("gateway-integration", NEW_REVISION, failed_jobs.append):
            host.containers = (
                []
                if runtime_state == "absent"
                else _family(NEW_REVISION, health=runtime_state)
            )
            executor._deploy_gateway_lane(
                lambda phase, status: None,
                lane=EnumRuntimeLane.DEV,
                build_source=BuildSource.RELEASE,
                targets=["gateway-forwarder"],
                git_ref=NEW_REVISION,
            )

    # The real executor must emit the exact class the real guard handles. A
    # generic RuntimeError here would silently roll a healthy family back.
    assert isinstance(recovery.value.__cause__, guard_mod.DevCompanionLaneFailedError)
    assert failed_jobs and diagnostic in failed_jobs[0]
    gateway_command.assert_called_once()
    restore_byproducts.assert_called_once()
    assert guard.point is not None
    incident = _incident(host, tmp_path)
    assert incident["status"] == "failed"
    assert incident["recovery_verified"] is True
    if runtime_state == "healthy":
        assert not host.restored
        assert guard.point.revision == NEW_REVISION
        assert incident["runtime_kept_revision"] == NEW_REVISION
    else:
        assert host.restored
        assert guard.point.revision == OLD_REVISION
        assert "runtime_kept_revision" not in incident
    assert {
        container["Config"]["Labels"]["org.opencontainers.image.revision"]
        for container in host.containers
    } == {NEW_REVISION if runtime_state == "healthy" else OLD_REVISION}
