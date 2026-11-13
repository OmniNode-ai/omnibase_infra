# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed dev roll as the deploy agent publishes it."""

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from omnibase_infra.models.model_dev_log_capture import ModelDevLogCapture

JobId = Annotated[str, Field(min_length=1)]


class ModelDevRollIncident(BaseModel):
    """Published when the dev boot guard recovers a failed roll (five whys R2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    job_id: JobId
    issuer: str = Field(min_length=1)
    commit: str
    error: str
    project: str
    at: datetime
    restored_revision: str
    recovery_verified: bool
    recovery_skipped_healthy: bool | None = None
    recovery_error: str | None = None
    runtime_kept_revision: str | None = None
    health_probes: list[dict[str, JsonValue]]
    log_captures: list[ModelDevLogCapture] = Field(default_factory=list)

    @classmethod
    def from_guard_incident(
        cls,
        incident: Mapping[str, object],
        *,
        event_id: UUID,
        issuer: str,
    ) -> "ModelDevRollIncident":
        """Map the boot guard's incident dict; the guard's pending keys are dropped."""
        known = {k: v for k, v in incident.items() if k in cls.model_fields}
        return cls.model_validate(
            {**known, "event_id": event_id, "job_id": incident["job"], "issuer": issuer}
        )
