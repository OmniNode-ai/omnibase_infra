# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev fold retires every folded role's container and cannot widen an empty restart scope."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from deploy_agent.compose_budget import _read_compose_services
from deploy_agent.events import (
    DEV_LANE_GATEWAY_SERVICES,
    EnumRuntimeLane,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import (
    DEFAULT_DEV_INSTANCE,
    DeployExecutor,
    lane_config_for,
    lane_folded_profiles,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def _committed_fold_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read this checkout's fold overlay, not the deploy host's clone path."""
    monkeypatch.setattr(
        "deploy_agent.executor.RUNTIME_POLICY_ENV_FILE",
        REPO_ROOT / "docker/runtime-policy.env",
    )


RUNTIME_WRITERS = {
    "projection-writer-live-events": "projection-live-events-writer",
    "projection-writer-registration": "projection-registration-writer",
    "projection-writer-savings": "projection-savings-writer",
    "projection-writer-tenant-credentials": "projection-tenant-credentials-writer",
    "projection-writer-tenant-registry": "projection-tenant-registry-writer",
    "projection-writer-delegation": "projection-delegation-writer",
}
# The hook-events writer has no runtime profile of its own: main runs its node by
# default, so the policy retires it by service name (``retired_services``).
HOOK_EVENTS_WRITER = "projection-claude-hook-events-writer"


def test_primary_dev_disables_the_tenant_carrier_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deploy_agent.executor._load_runtime_policy_env",
        lambda: {
            "DEV_RUNTIME_MAIN_FOLDED_PROFILES": "tenant-projection",
            "DEV_RUNTIME_MAIN_FOLDABLE_SERVICES": json.dumps(
                {"tenant-projection": "tenant-projection-writer"}
            ),
        },
    )
    config = lane_config_for(EnumRuntimeLane.DEV)
    # The gateway project's services are off too: the shipped table declares
    # gateway-deploy off for dev-201 (OMN-21629).
    assert config.disabled_services == frozenset(
        {"tenant-projection-writer", *DEV_LANE_GATEWAY_SERVICES}
    )
    assert config.compose_files[-1].endswith("docker-compose.dev-tenant-fold.yml")
    assert "tenant-projection" in lane_folded_profiles(EnumRuntimeLane.DEV)
    monkeypatch.setattr(
        "deploy_agent.executor.active_dev_instance",
        lambda: f"{DEFAULT_DEV_INSTANCE}-other",
    )
    assert not lane_folded_profiles(EnumRuntimeLane.DEV)


def test_committed_policy_disables_every_folded_writer() -> None:
    """The committed policy folds the six in-process writers; each container is disabled."""
    policy = Path(__file__).resolve().parents[4] / "docker/runtime-policy.env"
    with patch("deploy_agent.executor.RUNTIME_POLICY_ENV_FILE", policy):
        folded = lane_folded_profiles(EnumRuntimeLane.DEV)
        config = lane_config_for(EnumRuntimeLane.DEV)
    assert set(RUNTIME_WRITERS) <= folded
    # The gateway project's services are off too (OMN-21629), as the sibling test reads.
    assert config.disabled_services == frozenset(
        {
            "tenant-projection-writer",
            *RUNTIME_WRITERS.values(),
            HOOK_EVENTS_WRITER,
            *DEV_LANE_GATEWAY_SERVICES,
        }
    )
    # The delegation writer is folded too (plan task T10, OMN-22664).
    assert "projection-delegation-writer" in config.disabled_services


def test_folded_profile_without_a_gated_service_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deploy_agent.executor._load_runtime_policy_env",
        lambda: {"DEV_RUNTIME_MAIN_FOLDED_PROFILES": "workers"},
    )
    with pytest.raises(RuntimeError, match=r"maps no service for \['workers'\]"):
        lane_config_for(EnumRuntimeLane.DEV)


@pytest.mark.parametrize("failure", [None, "ps", "stop", "rm"])
def test_retirement_is_scoped_and_keeps_volumes(failure: str | None) -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        service = next(
            (
                arg.rsplit("=", 1)[1]
                for arg in cmd
                if arg.startswith("label=com.docker.compose.service=")
            ),
            "",
        )
        return subprocess.CompletedProcess(
            cmd,
            1 if cmd[1] == failure else 0,
            stdout=f"{service}-a\n{service}-b\n" if cmd[1] == "ps" else "",
            stderr="",
        )

    profiles = frozenset({"tenant-projection", "projection-writer-savings"})
    with (
        patch(
            "deploy_agent.executor.lane_folded_profiles",
            return_value=profiles,
        ),
        patch("deploy_agent.executor._run", side_effect=run),
    ):
        if failure:
            with pytest.raises(RuntimeError):
                DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.DEV)
        else:
            DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.DEV)
    assert all(
        "label=com.docker.compose.project=omnibase-infra" in c
        for c in calls
        if c[1] == "ps"
    )
    assert all("-v" not in c and "--volumes" not in c for c in calls)
    if failure in ("ps",):
        assert len(calls) == 1
        return
    censused = [c[-1] for c in calls if c[1] == "ps"]
    assert censused == [
        f"label=com.docker.compose.service={HOOK_EVENTS_WRITER}",
        "label=com.docker.compose.service=projection-savings-writer",
        "label=com.docker.compose.service=tenant-projection-writer",
    ]
    if failure is None:
        ids = [
            f"{HOOK_EVENTS_WRITER}-a",
            f"{HOOK_EVENTS_WRITER}-b",
            "projection-savings-writer-a",
            "projection-savings-writer-b",
            "tenant-projection-writer-a",
            "tenant-projection-writer-b",
        ]
        assert calls[3] == ["docker", "stop", "--time", "30", *ids]
        assert calls[4] == ["docker", "rm", *ids]


def test_retirement_is_inert_on_other_lanes() -> None:
    with patch("deploy_agent.executor._run") as run:
        DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.STABILITY_TEST)
    run.assert_not_called()
    assert HOOK_EVENTS_WRITER not in (
        lane_config_for(EnumRuntimeLane.STABILITY_TEST).disabled_services
    )


def test_other_dev_instance_does_not_retire_the_hook_events_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: the primary instance retires it; any other instance folds nothing."""
    assert HOOK_EVENTS_WRITER in lane_config_for(EnumRuntimeLane.DEV).disabled_services
    monkeypatch.setattr(
        "deploy_agent.executor.active_dev_instance",
        lambda: f"{DEFAULT_DEV_INSTANCE}-other",
    )
    assert HOOK_EVENTS_WRITER not in (
        lane_config_for(EnumRuntimeLane.DEV).disabled_services
    )
    with patch("deploy_agent.executor._run") as run:
        DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.DEV)
    run.assert_not_called()


def test_only_folded_tenant_carrier_request_does_not_start_the_whole_profile() -> None:
    updates: list[tuple[Phase, PhaseStatus]] = []
    with (
        patch("deploy_agent.executor._requested_services_for_up", return_value=[]),
        patch("deploy_agent.executor._run") as run,
    ):
        DeployExecutor()._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            ["tenant-projection-writer"],
            lambda phase, status: updates.append((phase, status)),
            lane=EnumRuntimeLane.DEV,
        )
    run.assert_not_called()
    assert updates[-1] == (Phase.RUNTIME, PhaseStatus.SUCCESS)


def test_budget_reader_honors_service_reset(tmp_path) -> None:
    base = tmp_path / "base.yml"
    overlay = tmp_path / "fold.yml"
    base.write_text(
        "services:\n  main:\n    image: example/main\n  tenant-projection-writer:\n    image: example/tenant-projection-writer\n"
    )
    overlay.write_text("services:\n  tenant-projection-writer: !reset null\n")
    assert set(_read_compose_services([str(base), str(overlay)])) == {"main"}


def test_real_folded_deploy_retires_tenant_carrier_before_starting_main() -> None:
    calls: list[list[str]] = []

    def run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=f"{cmd[-1].rsplit('=', 1)[1]}-a\n" if cmd[1] == "ps" else "",
            stderr="",
        )

    executor = DeployExecutor()
    policy = Path(__file__).resolve().parents[4] / "docker/runtime-policy.env"
    with (
        patch("deploy_agent.executor.RUNTIME_POLICY_ENV_FILE", policy),
        patch("deploy_agent.executor._run", side_effect=run),
        patch(
            "deploy_agent.executor.runtime_compose_up_budget",
            return_value=SimpleNamespace(timeout_seconds=120, describe=lambda: "test"),
        ),
        patch("deploy_agent.executor.finish_recreate_renames"),
        patch(
            "deploy_agent.executor.verify_containers_up", return_value=(True, [])
        ) as verify,
        patch.object(executor, "_ensure_runtime_migrations_ready"),
        patch.object(executor, "rerun_changed_oneshots"),
    ):
        executor._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            [],
            lambda *args: None,
            lane=EnumRuntimeLane.DEV,
        )
    retired = {
        "tenant-projection-writer",
        *RUNTIME_WRITERS.values(),
        HOOK_EVENTS_WRITER,
    }
    census = len(retired)
    compose_calls = calls[census + 2 :]
    assert [c[1] for c in calls[: census + 2]] == [*["ps"] * census, "stop", "rm"]
    assert {c[1] for c in compose_calls} == {"compose"}
    assert "omninode-runtime" in compose_calls[0]
    for service in retired:
        assert all(service not in c for c in compose_calls)
        assert service not in verify.call_args.args[0]
        assert f"{service}-a" in calls[census]


@pytest.mark.parametrize("primary", [True, False])
def test_compose_activation_uses_policy_after_inherited_and_caller_overrides(
    monkeypatch: pytest.MonkeyPatch,
    primary: bool,
) -> None:
    from deploy_agent import executor as execution

    policy = execution._load_runtime_policy_env()
    gate = "DEV_RUNTIME_MAIN_PROJECTION_SAVINGS_WRITER_RUNTIME_PROFILE"
    monkeypatch.setenv(gate, "runtime")
    monkeypatch.setattr(
        execution,
        "active_dev_instance",
        lambda: DEFAULT_DEV_INSTANCE if primary else "other",
    )
    monkeypatch.setattr(execution, "operator_store_overlay", dict)
    monkeypatch.setattr(execution, "_lane_env_overlay", dict)
    monkeypatch.setattr(execution, "_dev_broker_policy_env", lambda env: {})
    monkeypatch.setattr(execution, "resolve_lane_overlay_dir", lambda env: None)
    env = execution._compose_env({gate: "full", "DEV_RUNTIME_MAIN_FOLDED_PROFILES": ""})
    assert env[gate] == (policy[gate] if primary else "runtime")
    assert (
        env["DEV_RUNTIME_MAIN_FOLDED_PROFILES"]
        == policy["DEV_RUNTIME_MAIN_FOLDED_PROFILES"]
    )
    if not primary:
        assert not execution._folded_dev_services(EnumRuntimeLane.DEV)


@pytest.mark.parametrize(
    "mapping",
    [
        [],
        {"projection-writer-savings": "../other"},
        {"projection-writer-savings": ["wrong"]},
    ],
)
def test_malformed_retirement_policy_is_refused_before_any_docker_call(
    monkeypatch: pytest.MonkeyPatch,
    mapping: object,
) -> None:
    monkeypatch.setattr(
        "deploy_agent.executor._load_runtime_policy_env",
        lambda: {
            "DEV_RUNTIME_MAIN_FOLDED_PROFILES": "projection-writer-savings",
            "DEV_RUNTIME_MAIN_FOLDABLE_SERVICES": json.dumps(mapping),
        },
    )
    with patch("deploy_agent.executor._run") as run:
        with pytest.raises(RuntimeError, match="invalid dev foldable service mapping"):
            DeployExecutor()._retire_folded_dev_services(EnumRuntimeLane.DEV)
    run.assert_not_called()
