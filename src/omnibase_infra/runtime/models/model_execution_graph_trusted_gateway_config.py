# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed runtime composition input for the trusted graph-read gateway."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelExecutionGraphTrustedGatewayConfig(BaseModel):
    """Contract-derived command and the one signer/key file admitted for it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command_topic: str = Field(min_length=1)
    runtime_id: str = Field(min_length=1)  # string-id-ok: named gateway signer
    realm: str = Field(min_length=1)
    bus_id: str = Field(min_length=1)  # string-id-ok: named message bus
    public_key_path: Path

    @field_validator("public_key_path", mode="before")  # type: ignore[untyped-decorator]
    @classmethod
    def _path(cls, value: object) -> Path:
        if not isinstance(value, (str, Path)):
            raise ValueError("graph gateway public_key_path must be a path")
        path = Path(value)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                "graph gateway public_key_path must be an existing absolute file"
            )
        return path


__all__ = ["ModelExecutionGraphTrustedGatewayConfig"]
