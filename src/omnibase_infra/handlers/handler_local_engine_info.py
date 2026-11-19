# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the Docker engine's CPU count and memory via `docker info`.

Bound from the local provision node's contract as its ``engine_info``
operation. A read only: it runs ``docker info --format {{json .}}`` once and never starts,
stops or changes anything. A setup check compares the result with the selected
services' Compose reservations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID, uuid4

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_local_engine_info_request import (
    ModelLocalEngineInfoRequest,
)
from omnibase_infra.models.model_local_engine_info_result import (
    ModelLocalEngineInfoResult,
)
from omnibase_infra.models.operational_timeouts import TIMEOUTS

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

ENGINE_INFO_COMMAND: tuple[str, ...] = ("docker", "info", "--format", "{{json .}}")
_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_MALFORMED = "Docker returned an invalid or incomplete engine observation"
_DOCKER_MISSING = (
    "Docker is required: the `docker` command was not found on PATH. Install "
    "Docker (https://docs.docker.com/get-docker/) and start it, then retry."
)


class _OutputLimitError(Exception):
    """An engine observation exceeded the setup reader's byte limit."""


async def _read_output(stream: asyncio.StreamReader | None) -> bytes:
    output = bytearray()
    if stream is not None:
        while chunk := await stream.read(
            min(_READ_CHUNK_BYTES, _MAX_OUTPUT_BYTES + 1 - len(output))
        ):
            output.extend(chunk)
            if len(output) > _MAX_OUTPUT_BYTES:
                raise _OutputLimitError
    return bytes(output)


async def _collect_output(proc: asyncio.subprocess.Process) -> tuple[bytes, bytes]:
    stdout = asyncio.create_task(_read_output(proc.stdout))
    stderr = asyncio.create_task(_read_output(proc.stderr))
    waited = asyncio.create_task(proc.wait())
    try:
        out, err, _ = await asyncio.gather(stdout, stderr, waited)
        return out, err
    finally:
        # A failed reader must release its sibling before cleanup drains it.
        for task in (stdout, stderr, waited):
            if not task.done():
                task.cancel()
        await asyncio.gather(stdout, stderr, waited, return_exceptions=True)


async def _read_engine(timeout_seconds: float) -> tuple[int, str, str]:
    """Run the engine read; return its exit code, stderr and stdout, stripped.

    A missing ``docker`` binary is exit 127 with a message naming Docker.
    """
    # Match setup's command boundary: retain Docker host/context/TLS settings,
    # but do not lend the child ONEX, Python import or Git checkout overrides.
    environment = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "OMNI_HOME",
        "OMNIBASE_PATH",
        "ONEX_CONTRACTS_DIR",
        "ONEX_CONFIG_DIR",
        "ONEX_TENANT_ID",
        "ONEX_TENANT_SLUG",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        environment.pop(name, None)
    try:
        proc = await asyncio.create_subprocess_exec(
            *ENGINE_INFO_COMMAND,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
    except FileNotFoundError:
        return 127, _DOCKER_MISSING, ""
    except OSError:
        return 1, "Docker engine observation could not be started", ""
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            _collect_output(proc), timeout=timeout_seconds
        )
    except _OutputLimitError:
        await _kill_and_reap(proc)
        raise
    except TimeoutError:
        await _kill_and_reap(proc)
        return 1, "Docker engine observation timed out", ""
    except OSError:
        await _kill_and_reap(proc)
        return 1, "Docker engine observation could not be read", ""
    except asyncio.CancelledError:
        await _kill_and_reap(proc)
        raise
    returncode = proc.returncode if proc.returncode is not None else 1
    stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
    if returncode != 0 and stderr_text != "Cannot connect to the Docker daemon":
        # Child diagnostics can contain credentials or private Docker context.
        # Only the exact canonical daemon-down message is public; generated
        # missing/start/read/timeout diagnostics returned above stay intact.
        stderr_text = f"docker info exited with code {returncode}"
    return (
        returncode,
        stderr_text,
        stdout_bytes.decode("utf-8", errors="replace").strip(),
    )


async def _drain_output(stream: asyncio.StreamReader) -> None:
    while await stream.read(_READ_CHUNK_BYTES):
        pass


async def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    """End an interrupted engine observation before returning to its caller."""
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    # A paused pipe keeps asyncio's process waiter pending even after SIGKILL.
    # Drain both streams concurrently with the wait, including after a failed
    # read. One failed reader must not cancel the other or the reap.
    try:
        await asyncio.wait_for(
            asyncio.gather(
                *(
                    _drain_output(stream)
                    for stream in (proc.stdout, proc.stderr)
                    if stream
                ),
                proc.wait(),
                return_exceptions=True,
            ),
            timeout=TIMEOUTS.kill_wait_seconds,
        )
    except TimeoutError:
        # A descendant may retain inherited pipes after our child has exited.
        # Process has no public close API; closing its owned transport releases
        # these pipes without replacing the caller's timeout/cancellation.
        transport = cast("asyncio.SubprocessTransport", vars(proc)["_transport"])
        transport.close()


def _positive_int(value: object) -> int | None:
    """Return ``value`` when it is a positive int (not a bool), else None."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


class HandlerLocalEngineInfo:
    """Reads the Docker engine's resources for the ``engine_info`` operation."""

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the handler with a DI container."""
        self._container = container
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the handler type enum value."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the handler category enum value."""
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the handler (no-op for this handler)."""
        self._initialized = True

    async def shutdown(self) -> None:
        """Shut down the handler (no-op for this handler)."""
        self._initialized = False

    async def handle_async(
        self, request: ModelLocalEngineInfoRequest
    ) -> ModelLocalEngineInfoResult:
        """Dispatch entrypoint (OMN-14135): one engine read for one request."""
        output = await self.execute(
            {
                "correlation_id": request.correlation_id,
                "timeout_seconds": request.timeout_seconds,
            }
        )
        result = output.result
        if not isinstance(result, ModelLocalEngineInfoResult):
            raise TypeError("engine_info returned no ModelLocalEngineInfoResult")
        return result

    async def execute(self, envelope: dict[str, object]) -> ModelHandlerOutput:
        """Read the engine's CPU count and total memory in bytes.

        Envelope keys:
            correlation_id: UUID — for tracing.

        Returns:
            ModelHandlerOutput wrapping ModelLocalEngineInfoResult. Every
            failure is a result with success False and an engine_status of
            ``unavailable`` (Docker missing or the engine not answering) or
            ``malformed`` (an answer without a positive whole NCPU and MemTotal).
        """
        correlation_id_raw = envelope.get("correlation_id")
        corr_id: UUID = (
            correlation_id_raw if isinstance(correlation_id_raw, UUID) else uuid4()
        )

        request_data: dict[str, object] = {"correlation_id": corr_id}
        if "timeout_seconds" in envelope:
            request_data["timeout_seconds"] = envelope["timeout_seconds"]
        request = ModelLocalEngineInfoRequest.model_validate(request_data)
        try:
            returncode, stderr_text, stdout_text = await _read_engine(
                request.timeout_seconds
            )
        except _OutputLimitError:
            return _output(
                corr_id,
                "malformed",
                error="Docker engine observation exceeded the output limit",
            )
        if returncode != 0:
            logger.error("docker info failed (rc=%d)", returncode)
            return _output(
                corr_id,
                "unavailable",
                error=stderr_text or f"docker info exited with code {returncode}",
            )

        try:
            info = json.loads(stdout_text)
        except ValueError:
            return _output(corr_id, "malformed", error=_MALFORMED)
        if not isinstance(info, dict):
            return _output(corr_id, "malformed", error=_MALFORMED)
        cpus = _positive_int(info.get("NCPU"))
        memory = _positive_int(info.get("MemTotal"))
        if cpus is None or memory is None:
            return _output(corr_id, "malformed", error=_MALFORMED)
        return _output(corr_id, "read", cpus=cpus, memory=memory)


def _output(
    corr_id: UUID,
    status: Literal["read", "unavailable", "malformed"],
    *,
    error: str | None = None,
    cpus: int | None = None,
    memory: int | None = None,
) -> ModelHandlerOutput:
    """Wrap one engine read in ModelHandlerOutput."""
    return ModelHandlerOutput.for_compute(
        input_envelope_id=uuid4(),
        correlation_id=corr_id,
        handler_id="handler-local-engine-info",
        result=ModelLocalEngineInfoResult(
            success=status == "read",
            correlation_id=corr_id,
            engine_status=status,
            engine_cpus=cpus,
            engine_memory_bytes=memory,
            error=error,
        ),
    )


__all__: list[str] = ["ENGINE_INFO_COMMAND", "HandlerLocalEngineInfo"]
