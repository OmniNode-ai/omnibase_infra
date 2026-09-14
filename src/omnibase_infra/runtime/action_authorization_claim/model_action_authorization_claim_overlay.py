# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed contract-overlay references for the local claim bootstrap seam."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

OverlayReference = Annotated[
    str,
    Field(
        min_length=11,
        max_length=256,
        pattern=r"^overlay://[A-Za-z0-9][A-Za-z0-9._/-]{0,244}$",
    ),
]


class ModelActionAuthorizationClaimOverlay(BaseModel):
    """The sole source of connection and local-tool references for bootstrap.

    References are opaque overlay identifiers. Resolution is intentionally
    injected by the composition root; this package does not read process
    environment or a fallback configuration file.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    postgres_connection_ref: OverlayReference
    local_tool_ref: OverlayReference
    unix_socket_path: str = Field(min_length=2, max_length=104)
    restricted_principal: Literal["rsd_action_authorization_claim"]
    socket_owner_uid: int = Field(ge=0)
    authorized_unix_uid: int = Field(ge=0)

    @field_validator("unix_socket_path")
    @classmethod
    def _validate_unix_socket_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or "\x00" in value:
            msg = "unix_socket_path must be an absolute local socket path"
            raise ValueError(msg)
        return value


__all__ = ["ModelActionAuthorizationClaimOverlay"]
