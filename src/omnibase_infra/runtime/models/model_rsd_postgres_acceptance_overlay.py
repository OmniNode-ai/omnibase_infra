# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed, topology-free operator contract for RSD PostgreSQL acceptance."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelRsdPostgresAcceptanceOverlay(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    schema_version: Literal["rsd_postgres_acceptance_overlay.v1"]
    lane: str = Field(pattern=r"^[a-z][a-z0-9-]{1,63}$")
    locale: Literal["lab"]
    rsd_distribution_ref: str = Field(pattern=r"^[a-z0-9._/-]{3,255}$")
    postgres_capability_ref: str = Field(
        pattern=r"^capability://rsd/postgres/(?:acceptance|[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$"
    )
