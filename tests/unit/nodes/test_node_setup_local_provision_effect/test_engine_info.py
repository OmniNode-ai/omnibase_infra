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

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.handlers.handler_local_engine_info import (
    HandlerLocalEngineInfo,
)
from omnibase_infra.models.model_local_engine_info_result import (
    ModelLocalEngineInfoResult,
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


async def _run(exec_mock: Any) -> ModelLocalEngineInfoResult:
    with patch(_EXEC, exec_mock):
        output = await HandlerLocalEngineInfo(MagicMock()).execute(
            {"correlation_id": uuid4()}
        )
    result = output.result
    assert isinstance(result, ModelLocalEngineInfoResult)
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
    assert contract["config"]["local_engine_info"]["output_model"] == {
        "name": "ModelLocalEngineInfoResult",
        "module": "omnibase_infra.models.model_local_engine_info_result",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_dispatch_entrypoint_returns_the_typed_result() -> None:
    """OMN-14135: a contract-routed handler is executable through handle_async."""
    from omnibase_infra.models.model_local_engine_info_request import (
        ModelLocalEngineInfoRequest,
    )

    correlation_id = uuid4()
    proc = _proc(stdout=json.dumps(_GOOD).encode())
    with patch(_EXEC, AsyncMock(return_value=proc)):
        result = await HandlerLocalEngineInfo(MagicMock()).handle_async(
            ModelLocalEngineInfoRequest(correlation_id=correlation_id)
        )
    assert isinstance(result, ModelLocalEngineInfoResult)
    assert result.correlation_id == correlation_id
    assert result.engine_status == "read"
    assert result.engine_cpus == 8


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [PermissionError("sentinel-secret"), OSError("sentinel-secret")]
)
async def test_launch_os_errors_return_unavailable(error: OSError) -> None:
    result = await _run(AsyncMock(side_effect=error))
    assert not result.success
    assert result.engine_status == "unavailable"
    assert result.error
    assert "sentinel-secret" not in result.model_dump_json()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "cancel", "read-error"])
async def test_interrupted_observation_leaves_no_running_child(outcome: str) -> None:
    from omnibase_infra.models.model_local_engine_info_request import (
        ModelLocalEngineInfoRequest,
    )

    request_data: dict[str, object] = {"correlation_id": uuid4()}
    if outcome == "timeout":
        request_data["timeout_seconds"] = 0.01
    request = ModelLocalEngineInfoRequest.model_validate(request_data)
    # Exercise an actual pending child, without invoking Docker on this host.
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    started = asyncio.Event()

    async def observed_process(
        *args: object, **kwargs: object
    ) -> asyncio.subprocess.Process:
        started.set()
        return proc

    if outcome == "read-error":
        proc.communicate = AsyncMock(side_effect=OSError("sentinel-secret"))
    try:
        with patch(_EXEC, observed_process):
            task = asyncio.create_task(
                HandlerLocalEngineInfo(MagicMock()).handle_async(request)
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            if outcome == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await asyncio.wait_for(task, timeout=5)
                assert not result.success
                assert result.engine_status == "unavailable"
                assert result.error
                assert "sentinel-secret" not in result.model_dump_json()
        assert proc.returncode is not None, (
            "interrupted observation left its child running"
        )
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_requested_deadline_preserves_success() -> None:
    from omnibase_infra.models.model_local_engine_info_request import (
        ModelLocalEngineInfoRequest,
    )

    proc = _proc(stdout=json.dumps(_GOOD).encode())
    with patch(_EXEC, AsyncMock(return_value=proc)):
        result = await HandlerLocalEngineInfo(MagicMock()).handle_async(
            ModelLocalEngineInfoRequest(correlation_id=uuid4(), timeout_seconds=0.1)
        )
    assert result.success
    assert result.engine_cpus == _GOOD["NCPU"]


@pytest.mark.unit
@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_deadline_is_positive_and_finite(timeout: float) -> None:
    from pydantic import ValidationError

    from omnibase_infra.models.model_local_engine_info_request import (
        ModelLocalEngineInfoRequest,
    )

    with pytest.raises(ValidationError):
        ModelLocalEngineInfoRequest(correlation_id=uuid4(), timeout_seconds=timeout)


@pytest.mark.unit
def test_default_deadline_comes_from_operational_contract() -> None:
    from omnibase_infra.models.model_local_engine_info_request import (
        ModelLocalEngineInfoRequest,
    )
    from omnibase_infra.models.operational_timeouts import TIMEOUTS

    request = ModelLocalEngineInfoRequest(correlation_id=uuid4())
    assert request.timeout_seconds == TIMEOUTS.dev_lane_docker_seconds


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("descriptor", [1, 2])
async def test_read_failure_drains_full_pipes_before_reaping(descriptor: int) -> None:
    """A killed child can still block wait() until its output pipes are drained."""
    from omnibase_infra.handlers.handler_local_engine_info import _read_engine

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        f"import os,time; os.write({descriptor}, b'x' * 1048576); time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    original_communicate = proc.communicate
    stream = proc.stdout if descriptor == 1 else proc.stderr
    assert stream is not None

    async def saturated() -> None:
        # Observe actual StreamReader backpressure; a sleep alone can miss the
        # full-pipe condition and falsely pass the old implementation.
        while not getattr(stream, "_paused", False):
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(saturated(), timeout=5)
        proc.communicate = AsyncMock(side_effect=OSError("sentinel-secret"))
        with patch(_EXEC, AsyncMock(return_value=proc)):
            code, error, output = await asyncio.wait_for(_read_engine(0.01), timeout=5)
        assert code != 0
        assert "could not be read" in error
        assert "sentinel-secret" not in error
        assert not output
        assert proc.returncode is not None
    finally:
        if proc.returncode is None:
            proc.kill()
        await original_communicate()
        await proc.wait()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "cancel", "read-error"])
async def test_inherited_pipe_deadline_preserves_the_original_outcome(
    outcome: str,
) -> None:
    from omnibase_infra.handlers.handler_local_engine_info import _read_engine

    release_read, release_write = os.pipe()
    # The descendant keeps both output pipes open until this test releases it.
    # No timing assumption decides whether the inherited-pipe fault is present.
    holder = "import os,sys; os.read(int(sys.argv[1]), 1)"
    parent = (
        "import subprocess,sys,time; fd=int(sys.argv[1]); "
        f"subprocess.Popen([sys.executable, '-c', {holder!r}, str(fd)], pass_fds=(fd,)); "
        "print('ready', file=sys.stderr, flush=True); time.sleep(60)"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        parent,
        str(release_read),
        pass_fds=(release_read,),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    os.close(release_read)
    original_communicate = proc.communicate
    assert proc.stderr is not None
    started = asyncio.Event()

    async def observed_process(
        *args: object, **kwargs: object
    ) -> asyncio.subprocess.Process:
        started.set()
        return proc

    try:
        assert await asyncio.wait_for(proc.stderr.readline(), timeout=5) == b"ready\n"
        if outcome == "read-error":
            proc.communicate = AsyncMock(side_effect=OSError("sentinel-secret"))
        with (
            patch(_EXEC, observed_process),
            patch(
                "omnibase_infra.handlers.handler_local_engine_info.TIMEOUTS",
                SimpleNamespace(kill_wait_seconds=0.05),
            ),
        ):
            task = asyncio.create_task(
                _read_engine(0.01 if outcome == "timeout" else 60)
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            if outcome == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=5)
            else:
                code, error, output = await asyncio.wait_for(task, timeout=5)
                assert code != 0
                assert (
                    "timed out" if outcome == "timeout" else "could not be read"
                ) in error
                assert not output
        assert proc.returncode is not None
        for stream in (proc.stdout, proc.stderr):
            assert stream is not None
            assert stream._transport.is_closing()
    finally:
        # Releasing the descendant prevents a leaked test process even on RED.
        os.write(release_write, b"x")
        os.close(release_write)
        if proc.returncode is None:
            proc.kill()
        await asyncio.wait_for(original_communicate(), timeout=5)
        await asyncio.wait_for(proc.wait(), timeout=5)
