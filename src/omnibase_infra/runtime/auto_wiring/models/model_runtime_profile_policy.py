# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-level narrowing hints for runtime profile eligibility.

Runtime profile policy is authoritative in the kernel/runtime layer. Contracts
may narrow that policy for a specific profile, but they must never broaden it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeProfilePolicy(BaseModel):
    """Optional per-profile contract hints consumed by auto-wiring."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    eligible: bool | None = Field(
        default=None,
        description=(
            "Contract-level narrowing hint. False excludes the contract from the "
            "named runtime profile. True does not broaden profile eligibility."
        ),
    )
    optional: bool = Field(
        default=False,
        description=(
            "Whether this contract is optional for the named runtime profile. "
            "Optional unresolved contracts degrade startup instead of hard-failing."
        ),
    )
