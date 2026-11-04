# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live resources acquired during registration bootstrap, before dispatch starts."""

from __future__ import annotations

from asyncpg import Pool
from pydantic import BaseModel, ConfigDict


class ModelRegistrationBootstrapResult(BaseModel):
    """Resources whose lifecycle remains owned by the registration plugin."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True, from_attributes=True
    )

    pool: Pool | None = None


__all__ = ["ModelRegistrationBootstrapResult", "Pool"]
