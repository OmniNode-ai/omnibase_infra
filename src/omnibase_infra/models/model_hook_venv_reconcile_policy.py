# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Process budgets supplied from the hook reconciliation node contract."""

from pydantic import BaseModel, ConfigDict, Field


class ModelHookVenvReconcilePolicy(BaseModel):
    """Bound local ancestry reads and package operations."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    git_timeout_seconds: float = Field(gt=0)
    sync_timeout_seconds: float = Field(gt=0)
