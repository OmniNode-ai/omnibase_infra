# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed dev roll as the deploy agent publishes it."""

from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_dev_log_capture import ModelDevLogCapture


class ModelDevRollIncident(BaseModel):
    """Published when the dev boot guard recovers a failed roll (five whys R2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    job_id: str = Field(min_length=1)
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
    health_probes: list[dict[str, Any]]
    log_captures: list[ModelDevLogCapture] = Field(default_factory=list)

    @classmethod
    def from_guard_incident(
        cls, incident: Mapping[str, Any], *, event_id: UUID, issuer: str
    ) -> "ModelDevRollIncident":
        """Map the boot guard's incident dict; the guard's pending keys are dropped."""
        return cls(
            event_id=event_id,
            job_id=incident["job"],
            issuer=issuer,
            commit=incident["commit"],
            error=incident["error"],
            project=incident["project"],
            at=datetime.fromisoformat(incident["at"].replace("Z", "+00:00")),
            restored_revision=incident["restored_revision"],
            recovery_verified=incident["recovery_verified"],
            recovery_skipped_healthy=incident.get("recovery_skipped_healthy"),
            recovery_error=incident.get("recovery_error"),
            runtime_kept_revision=incident.get("runtime_kept_revision"),
            health_probes=incident.get("health_probes", []),
            log_captures=incident.get("log_captures", []),
        )
