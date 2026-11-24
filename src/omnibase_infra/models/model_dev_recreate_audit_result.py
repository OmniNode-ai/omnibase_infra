# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the dev recreate reader found in one read window (OMN-25912)."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_dev_recreate_violation import ModelDevRecreateViolation


class ModelDevRecreateAuditResult(BaseModel):
    """The violations, oldest first, and the number of recreates read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recreates_read: int
    violations: tuple[ModelDevRecreateViolation, ...] = ()
