# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the read-only engine_info operation of the local provision node.

The operation runs ``docker info --format {{json .}}`` and returns the engine's
CPU count and total memory in bytes, so a setup check can compare them with the
selected services' reservations without calling Docker itself.

Coverage: a good read; Docker missing from PATH; the daemon not running (a
non-zero exit); output that is not JSON; NCPU or MemTotal missing; zero,
negative or non-finite values. Every failure comes back as a typed result with
success false, an error and an engine_status naming the kind of failure, never
an exception. The handler runs that one read and never starts, stops or changes
anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.handlers.handler_local_engine_info import (
    HandlerLocalEngineInfo,
)
from omnibase_infra.nodes.node_setup_local_provision_effect.models.model_local_provision_effect_output import (
    ModelLocalProvisionEffectOutput,
)

_EXEC = (
    "omnibase_infra.handlers.handler_local_engine_info.asyncio.create_subprocess_exec"
)
_NODE = (
    Path(__file__).parents[4]
    / "src/omnibase_infra/nodes/node_setup_local_provision_effect"
)
_GOOD = {"NCPU": 8, "MemTotal": 8 * 1024**3, "OSType": "linux"}


def _proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


async def _run(exec_mock: Any) -> ModelLocalProvisionEffectOutput:
    with patch(_EXEC, exec_mock):
        output = await HandlerLocalEngineInfo(MagicMock()).execute(
            {"correlation_id": uuid4()}
        )
    result = output.result
    assert isinstance(result, ModelLocalProvisionEffectOutput)
    return result


@pytest.mark.unit
class TestEngineInfo:
    @pytest.mark.asyncio
    async def test_good_read_returns_cpus_and_memory(self) -> None:
        exec_mock = AsyncMock(return_value=_proc(json.dumps(_GOOD).encode()))
        result = await _run(exec_mock)
        assert result.success is True
        assert result.engine_status == "read"
        assert result.engine_cpus == 8.0
        assert result.engine_memory_bytes == 8 * 1024**3
        assert result.error is None

    @pytest.mark.asyncio
    async def test_runs_only_the_one_read(self) -> None:
        exec_mock = AsyncMock(return_value=_proc(json.dumps(_GOOD).encode()))
        await _run(exec_mock)
        assert exec_mock.await_count == 1
        assert list(exec_mock.await_args.args) == [
            "docker",
            "info",
            "--format",
            "{{json .}}",
        ]

    @pytest.mark.asyncio
    async def test_docker_missing_is_unavailable(self) -> None:
        result = await _run(AsyncMock(side_effect=FileNotFoundError("docker")))
        assert result.success is False
        assert result.engine_status == "unavailable"
        assert result.error and "Docker" in result.error
        assert result.engine_cpus is None and result.engine_memory_bytes is None

    @pytest.mark.asyncio
    async def test_daemon_down_is_unavailable(self) -> None:
        proc = _proc(stderr=b"Cannot connect to the Docker daemon", returncode=1)
        result = await _run(AsyncMock(return_value=proc))
        assert result.success is False
        assert result.engine_status == "unavailable"
        assert result.error == "Cannot connect to the Docker daemon"
        assert result.engine_cpus is None and result.engine_memory_bytes is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stdout",
        [
            b"not json",
            b"[]",
            json.dumps({"MemTotal": 1024}).encode(),
            json.dumps({"NCPU": 2}).encode(),
            json.dumps({**_GOOD, "NCPU": 0}).encode(),
            json.dumps({**_GOOD, "NCPU": -1}).encode(),
            json.dumps({**_GOOD, "MemTotal": 0}).encode(),
            json.dumps({**_GOOD, "MemTotal": -5}).encode(),
            json.dumps({**_GOOD, "NCPU": "lots"}).encode(),
            json.dumps({**_GOOD, "NCPU": True}).encode(),
            json.dumps({**_GOOD, "MemTotal": 1.5}).encode(),
            b'{"NCPU": NaN, "MemTotal": 1024}',
            b'{"NCPU": Infinity, "MemTotal": 1024}',
        ],
        ids=[
            "not-json",
            "not-object",
            "no-ncpu",
            "no-memtotal",
            "zero-cpu",
            "negative-cpu",
            "zero-memory",
            "negative-memory",
            "text-cpu",
            "bool-cpu",
            "fractional-memory",
            "nan-cpu",
            "infinite-cpu",
        ],
    )
    async def test_bad_engine_output_is_malformed(self, stdout: bytes) -> None:
        result = await _run(AsyncMock(return_value=_proc(stdout)))
        assert result.success is False
        assert result.engine_status == "malformed"
        assert result.error
        assert result.engine_cpus is None and result.engine_memory_bytes is None


@pytest.mark.unit
def test_contract_routes_engine_info_to_the_handler() -> None:
    contract = yaml.safe_load((_NODE / "contract.yaml").read_text())
    routes = {
        row["operation"]: row["handler"]
        for row in contract["handler_routing"]["handlers"]
    }
    assert routes["engine_info"] == {
        "name": "HandlerLocalEngineInfo",
        "module": "omnibase_infra.handlers.handler_local_engine_info",
    }
    operations = {row["name"] for row in contract["io_operations"]}
    assert "engine_info" in operations
