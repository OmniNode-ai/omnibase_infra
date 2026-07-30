# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One explicit comparison in a family transformation receipt."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.migration.cutover.enums import EnumReceiptDimension

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ModelReceiptCheck(BaseModel):
    """Immutable outcome for one required receipt dimension."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    dimension: EnumReceiptDimension
    passed: bool
    source_digest: str = Field(..., pattern=_SHA256_PATTERN)
    target_digest: str = Field(..., pattern=_SHA256_PATTERN)
    detail: str = Field(..., min_length=1, max_length=500)


__all__ = ["ModelReceiptCheck"]
