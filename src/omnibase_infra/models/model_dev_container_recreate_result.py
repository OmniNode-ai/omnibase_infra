# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the container recreate handler decided at one tick (OMN-25912)."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_dev_container_recreated import (
    ModelDevContainerRecreated,
)


class ModelDevContainerRecreateResult(BaseModel):
    """The event for a new start, or ``None`` when the container has not started again."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event: ModelDevContainerRecreated | None = None
