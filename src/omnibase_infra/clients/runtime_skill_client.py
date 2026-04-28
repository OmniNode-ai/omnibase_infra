# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Shared host-local runtime skill client implementation."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Literal

from omnibase_core.models.runtime import (
    ModelRuntimeSkillError,
    ModelRuntimeSkillRequest,
    ModelRuntimeSkillResponse,
)
from omnibase_core.protocols.runtime import ProtocolRuntimeSkillClient

_DEFAULT_RUNTIME_SOCKET_PATH = "/tmp/onex-runtime.sock"  # noqa: S108


def default_runtime_socket_path() -> str:
    """Resolve the local runtime ingress socket path."""

    return os.environ.get(
        "ONEX_LOCAL_RUNTIME_SOCKET_PATH", _DEFAULT_RUNTIME_SOCKET_PATH
    )


class LocalRuntimeSkillClient(ProtocolRuntimeSkillClient):
    """Unix-socket client for the canonical host-local runtime skill path."""

    def __init__(
        self,
        *,
        socket_path: str | None = None,
        connect_timeout_seconds: float = 5.0,
    ) -> None:
        self._socket_path = socket_path or default_runtime_socket_path()
        self._connect_timeout_seconds = connect_timeout_seconds

    async def dispatch_async(
        self,
        request: ModelRuntimeSkillRequest,
    ) -> ModelRuntimeSkillResponse:
        writer: asyncio.StreamWriter | None = None
        try:
            reader, connected_writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self._socket_path),
                timeout=self._connect_timeout_seconds,
            )
            writer = connected_writer
            writer.write(
                request.model_dump_json(exclude_none=True).encode("utf-8") + b"\n"
            )
            await writer.drain()
            raw_response = await asyncio.wait_for(
                reader.readline(),
                timeout=request.timeout_ms / 1000.0,
            )
            if not raw_response:
                return _transport_error_response(
                    request=request,
                    code="dispatch_error",
                    message="Runtime local ingress closed the socket without a response.",
                )
            payload = json.loads(raw_response.decode("utf-8"))
            return ModelRuntimeSkillResponse.model_validate(payload)
        except TimeoutError:
            return _transport_error_response(
                request=request,
                code="dispatch_timeout",
                message=f"Local runtime transport timed out after {request.timeout_ms} ms.",
                retryable=True,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return _transport_error_response(
                request=request,
                code="runtime_unavailable"
                if isinstance(exc, OSError)
                else "dispatch_error",
                message=str(exc),
                retryable=isinstance(exc, OSError),
            )
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()

    def dispatch_sync(
        self,
        request: ModelRuntimeSkillRequest,
    ) -> ModelRuntimeSkillResponse:
        return asyncio.run(self.dispatch_async(request))


def _transport_error_response(
    *,
    request: ModelRuntimeSkillRequest,
    code: Literal[
        "validation_error",
        "unknown_command",
        "runtime_unavailable",
        "dispatch_timeout",
        "dispatch_error",
    ],
    message: str,
    retryable: bool = False,
) -> ModelRuntimeSkillResponse:
    return ModelRuntimeSkillResponse(
        ok=False,
        command_name=request.command_name,
        correlation_id=request.correlation_id,
        error=ModelRuntimeSkillError(
            code=code,
            message=message,
            retryable=retryable,
        ),
    )


__all__ = ["LocalRuntimeSkillClient", "default_runtime_socket_path"]
