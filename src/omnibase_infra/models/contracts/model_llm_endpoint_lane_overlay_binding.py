# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane overlay binding for an LLM endpoint slot (OMN-24025).

A slot flagged ``endpoint_from_overlay`` may carry a binding that names the
lane bifrost overlay backend whose address is rewritten to the overlay's
base URL, and the env var that carries that backend's complete endpoint URL.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class ModelLlmEndpointLaneOverlayBinding(BaseModel):
    """The lane bifrost overlay backend a slot's address is bound to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    env_var: str

    @field_validator("backend", "env_var", mode="after")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value


__all__ = ["ModelLlmEndpointLaneOverlayBinding"]
