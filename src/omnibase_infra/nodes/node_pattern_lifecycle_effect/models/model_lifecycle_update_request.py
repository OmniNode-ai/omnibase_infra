# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Request to apply a verdict to pattern lifecycle state."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums import EnumValidationVerdict
from omnibase_infra.nodes.node_pattern_lifecycle_effect.models.model_lifecycle_state import (
    ModelLifecycleState,
)


class ModelLifecycleUpdateRequest(BaseModel):
    """Pattern, verdict, and optional state and tracing context."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    pattern_id: UUID
    verdict: EnumValidationVerdict
    correlation_id: UUID | None = None
    current_state: ModelLifecycleState | None = None
