# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-owned bounds for announcement settling."""

from pydantic import BaseModel, ConfigDict, Field


class ModelBootRuntimeConfig(BaseModel):
    """Validated operational bounds; values are supplied by runtime composition."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    catalog_settle_timeout_seconds: float = Field(gt=0)
