# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the container recreate handler reads at one tick (OMN-25912)."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_dev_container_reading import ModelDevContainerReading
from omnibase_infra.models.model_dev_roll_manifest import ModelDevRollManifest
from omnibase_infra.models.model_dev_runtime_started import ModelDevRuntimeStarted


class ModelDevContainerRecreateRequest(BaseModel):
    """The reading now, the reading at the previous tick and the deploy agent's own events.

    ``start_match_tolerance_seconds`` is the node contract's
    ``audit_start_match_tolerance_seconds``, never a default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reading: ModelDevContainerReading
    previous: ModelDevContainerReading | None = None
    starts: tuple[ModelDevRuntimeStarted, ...] = ()
    manifests: tuple[ModelDevRollManifest, ...] = ()
    start_match_tolerance_seconds: int = Field(ge=0)
