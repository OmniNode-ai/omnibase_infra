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
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_local_engine_info_request import (
    ModelLocalEngineInfoRequest,
)
from omnibase_infra.models.model_local_engine_info_result import (
    ModelLocalEngineInfoResult,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

ENGINE_INFO_COMMAND: tuple[str, ...] = ("docker", "info", "--format", "{{json .}}")
_MALFORMED = "Docker returned an invalid or incomplete engine observation"
_DOCKER_MISSING = (
    "Docker is required: the `docker` command was not found on PATH. Install "
    "Docker (https://docs.docker.com/get-docker/) and start it, then retry."
)


async def _read_engine() -> tuple[int, str, str]:
    """Run the engine read; return its exit code, stderr and stdout, stripped.

    A missing ``docker`` binary is exit 127 with a message naming Docker.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *ENGINE_INFO_COMMAND,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return 127, _DOCKER_MISSING, ""
    stdout_bytes, stderr_bytes = await proc.communicate()
    returncode = proc.returncode if proc.returncode is not None else 1
    return (
        returncode,
        stderr_bytes.decode("utf-8", errors="replace").strip(),
        stdout_bytes.decode("utf-8", errors="replace").strip(),
    )


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
        output = await self.execute({"correlation_id": request.correlation_id})
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

        returncode, stderr_text, stdout_text = await _read_engine()
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
