# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The rows the dev recreate reader reads (OMN-25912)."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_dev_container_observation import (
    ModelDevContainerObservation,
)
from omnibase_infra.models.model_dev_recreate_announcement import (
    ModelDevRecreateAnnouncement,
)
from omnibase_infra.models.model_dev_recreate_policy import ModelDevRecreatePolicy
from omnibase_infra.models.model_dev_roll_manifest import ModelDevRollManifest
from omnibase_infra.models.model_dev_runtime_started import ModelDevRuntimeStarted


class ModelDevRecreateAuditRequest(BaseModel):
    """The declared window policy and the events of one read window.

    ``announcement_lookback_seconds`` and ``start_match_tolerance_seconds`` are node
    contract settings, never defaults.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: ModelDevRecreatePolicy
    announcement_lookback_seconds: int = Field(ge=1)
    start_match_tolerance_seconds: int = Field(ge=0)
    starts: tuple[ModelDevRuntimeStarted, ...] = ()
    manifests: tuple[ModelDevRollManifest, ...] = ()
    announcements: tuple[ModelDevRecreateAnnouncement, ...] = ()
    container_observations: tuple[ModelDevContainerObservation, ...] = ()
