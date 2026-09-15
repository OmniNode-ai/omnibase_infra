# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stable local client for the overlay-selected action-authorization socket."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import ValidationError

from omnibase_infra.runtime.action_authorization_claim.enum_action_authorization_claim_outcome import (
    EnumActionAuthorizationClaimOutcome,
)
from omnibase_infra.runtime.action_authorization_claim.model_action_authorization_claim_request import (
    ModelActionAuthorizationClaimRequest,
)
from omnibase_infra.runtime.action_authorization_claim.model_action_authorization_claim_result import (
    ModelActionAuthorizationClaimResult,
)

_MAX_FRAME_BYTES = 16_384
_CLAIM_TIMEOUT_SECONDS = 10.0


async def claim_action_authorization_via_unix_socket(
    *, socket_path: Path, request: ModelActionAuthorizationClaimRequest
) -> ModelActionAuthorizationClaimResult:
    """Claim one request through a local endpoint resolved by an approved overlay.

    The caller supplies a concrete path only after its composition root resolves
    the overlay's local-tool reference. This client does not read environment or
    configuration files, and it sends no authority flags beyond the canonical
    registry request.
    """
    wire = json.dumps(
        {"operation": "claim", "request": request.model_dump(mode="json")},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(wire) + 1 > _MAX_FRAME_BYTES:
        return ModelActionAuthorizationClaimResult(
            outcome=EnumActionAuthorizationClaimOutcome.ERROR
        )
    try:
        async with asyncio.timeout(_CLAIM_TIMEOUT_SECONDS):
            reader, writer = await asyncio.open_unix_connection(
                str(socket_path), limit=_MAX_FRAME_BYTES
            )
            try:
                writer.write(wire + b"\n")
                await writer.drain()
                response = await reader.readuntil(b"\n")
            finally:
                writer.close()
                await writer.wait_closed()
    except (
        TimeoutError,
        asyncio.IncompleteReadError,
        asyncio.LimitOverrunError,
        OSError,
    ):
        return ModelActionAuthorizationClaimResult(
            outcome=EnumActionAuthorizationClaimOutcome.ERROR
        )
    if not response or len(response) > _MAX_FRAME_BYTES:
        return ModelActionAuthorizationClaimResult(
            outcome=EnumActionAuthorizationClaimOutcome.ERROR
        )
    try:
        response_data = json.loads(response)
        return ModelActionAuthorizationClaimResult.model_validate(response_data)
    except (json.JSONDecodeError, ValidationError):
        return ModelActionAuthorizationClaimResult(
            outcome=EnumActionAuthorizationClaimOutcome.ERROR
        )


__all__ = ["claim_action_authorization_via_unix_socket"]
