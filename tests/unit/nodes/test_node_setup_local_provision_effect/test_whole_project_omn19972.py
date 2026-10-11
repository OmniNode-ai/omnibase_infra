# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Whole-project, env-file, build and volume options of the local provision node.

These are what ``onex local up|down|status`` drives over a rendered catalog
bundle (OMN-19972): the bundle carries services with no published port, so the
node acts on the whole compose project and polls only the topology's ports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import omnibase_infra.nodes.node_setup_local_provision_effect.handlers.handler_local_provision as provision_mod
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_infra.nodes.node_setup_local_provision_effect.handlers.handler_local_provision import (
    HandlerLocalProvision,
)
from omnibase_infra.nodes.node_setup_local_provision_effect.handlers.handler_local_status import (
    HandlerLocalStatus,
)
from omnibase_infra.nodes.node_setup_local_provision_effect.handlers.handler_local_teardown import (
    HandlerLocalTeardown,
)
from omnibase_infra.nodes.node_setup_local_provision_effect.models.model_local_provision_effect_input import (
    ModelLocalProvisionEffectInput,
)
from omnibase_infra.nodes.node_setup_local_provision_effect.models.model_local_provision_effect_output import (
    ModelLocalProvisionEffectOutput,
)

_EXEC = (
    "omnibase_infra.nodes.node_setup_local_provision_effect.handlers."
    "handler_local_provision.asyncio.create_subprocess_exec"
)


def _proc(stdout: bytes = b"", returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, b""))
    proc.returncode = returncode
    return proc


def _envelope(tmp_path: Path, operation: str, **fields: Any) -> dict[str, object]:
    compose = tmp_path / "compose.yml"
    compose.touch()
    model = ModelLocalProvisionEffectInput(
        topology=ModelDeploymentTopology.default_minimal(),
        compose_file_path=str(compose),
        correlation_id=uuid4(),
        operation=operation,
        **fields,
    )
    return dict(model)


def _result(output: Any) -> ModelLocalProvisionEffectOutput:
    result = output.result
    assert isinstance(result, ModelLocalProvisionEffectOutput)
    return result


@pytest.mark.unit
class TestProvisionWholeProject:
    async def test_build_then_up_with_env_files_and_no_service_args(
        self, tmp_path: Path
    ) -> None:
        calls: list[list[str]] = []

        async def fake_exec(*args: Any, **_kw: Any) -> MagicMock:
            calls.append(list(args))
            return _proc()

        env = _envelope(
            tmp_path,
            "provision_local",
            whole_project=True,
            env_files=("/p/policy.env", "/p/local.env"),
            build=True,
            build_args=("GIT_SHA=abc",),
            max_wait_seconds=1,
        )
        with (
            patch(_EXEC, side_effect=fake_exec),
            patch.object(
                provision_mod, "_poll_port_open", AsyncMock(return_value=True)
            ),
        ):
            out = _result(await HandlerLocalProvision(MagicMock()).execute(env))

        assert out.success
        build, up = calls
        base = [
            "docker",
            "compose",
            "--env-file",
            "/p/policy.env",
            "--env-file",
            "/p/local.env",
            "-f",
            env["compose_file_path"],
        ]
        assert build == [*base, "build", "--build-arg", "GIT_SHA=abc"]
        assert up == [*base, "up", "-d"]

    async def test_no_build_by_default(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []

        async def fake_exec(*args: Any, **_kw: Any) -> MagicMock:
            calls.append(list(args))
            return _proc()

        env = _envelope(tmp_path, "provision_local", whole_project=True)
        with (
            patch(_EXEC, side_effect=fake_exec),
            patch.object(
                provision_mod, "_poll_port_open", AsyncMock(return_value=True)
            ),
        ):
            await HandlerLocalProvision(MagicMock()).execute(env)
        assert [c[c.index("-f") + 2] for c in calls] == ["up"]

    async def test_missing_docker_is_a_typed_failure(self, tmp_path: Path) -> None:
        env = _envelope(tmp_path, "provision_local", whole_project=True)
        with patch(_EXEC, side_effect=FileNotFoundError("docker")):
            out = _result(await HandlerLocalProvision(MagicMock()).execute(env))
        assert not out.success
        assert out.error is not None and "Docker is required" in out.error


@pytest.mark.unit
class TestTeardownVolumes:
    async def _down(self, tmp_path: Path, **fields: Any) -> list[str]:
        calls: list[list[str]] = []

        async def fake_exec(*args: Any, **_kw: Any) -> MagicMock:
            calls.append(list(args))
            return _proc()

        env = _envelope(tmp_path, "teardown_local", **fields)
        with patch(_EXEC, side_effect=fake_exec):
            out = _result(await HandlerLocalTeardown(MagicMock()).execute(env))
        assert out.success
        (call,) = calls
        return call

    async def test_down_keeps_volumes_by_default(self, tmp_path: Path) -> None:
        call = await self._down(tmp_path, whole_project=True)
        assert call[-1] == "down"
        assert "--volumes" not in call

    async def test_remove_volumes_adds_the_flag(self, tmp_path: Path) -> None:
        call = await self._down(tmp_path, whole_project=True, remove_volumes=True)
        assert call[-2:] == ["down", "--volumes"]

    async def test_topology_scope_still_names_services(self, tmp_path: Path) -> None:
        call = await self._down(tmp_path)
        assert "omnibase-infra-postgres" in call


@pytest.mark.unit
class TestStatusWholeProject:
    async def test_reports_every_running_service_sorted(self, tmp_path: Path) -> None:
        async def fake_exec(*_args: Any, **_kw: Any) -> MagicMock:
            return _proc(stdout=b"valkey\npostgres\nmigration-gate\n")

        env = _envelope(
            tmp_path, "status_check", whole_project=True, env_files=("/p/a.env",)
        )
        with patch(_EXEC, side_effect=fake_exec):
            out = _result(await HandlerLocalStatus(MagicMock()).execute(env))
        assert out.success
        assert out.services_running == ("migration-gate", "postgres", "valkey")

    async def test_missing_docker_is_a_typed_failure(self, tmp_path: Path) -> None:
        env = _envelope(tmp_path, "status_check", whole_project=True)
        with patch(_EXEC, side_effect=FileNotFoundError("docker")):
            out = _result(await HandlerLocalStatus(MagicMock()).execute(env))
        assert not out.success
        assert out.error is not None and "Docker is required" in out.error
