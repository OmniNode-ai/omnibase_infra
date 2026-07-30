# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed full-day application-database activity-result evidence."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.validation.models.model_application_database_activity_principal_observation import (
    ModelApplicationDatabaseActivityPrincipalObservation,
)


class ModelApplicationDatabaseActivityResultEvidence(BaseModel):
    """Canonical full-day activity-result content for one database."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    database_ref: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    physical_database: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    window_started_at: AwareDatetime
    window_ended_at: AwareDatetime
    activity_query_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    observation_count: int = Field(..., ge=0)
    active_principals: tuple[
        ModelApplicationDatabaseActivityPrincipalObservation,
        ...,
    ]

    @model_validator(mode="after")
    def validate_full_day_window(
        self,
    ) -> ModelApplicationDatabaseActivityResultEvidence:
        """Reject unordered or partial-day result windows."""
        started_at = datetime.fromisoformat(self.window_started_at.isoformat())
        ended_at = datetime.fromisoformat(self.window_ended_at.isoformat())
        if ended_at - started_at < timedelta(hours=24):
            raise ValueError("activity result window must span at least 24 hours")
        principals = [row.principal for row in self.active_principals]
        if len(set(principals)) != len(principals):
            raise ValueError("activity result principal pairs must be unique")
        observed_count = sum(row.observation_count for row in self.active_principals)
        if observed_count != self.observation_count:
            raise ValueError(
                "activity result observation_count must equal exact pair-row counts"
            )
        return self


__all__ = ["ModelApplicationDatabaseActivityResultEvidence"]
