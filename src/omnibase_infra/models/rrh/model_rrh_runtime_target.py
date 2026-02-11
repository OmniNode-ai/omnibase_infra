# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Deployment target context for RRH validation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRRHRuntimeTarget(BaseModel):
    """Deployment target context.

    Attributes:
        environment: Target environment label (e.g. ``"dev"``, ``"staging"``).
        kafka_broker: Kafka bootstrap server address (empty if N/A).
        kubernetes_context: Active kubectl context (empty if N/A).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    environment: str = Field(default="dev", description="Target environment label.")
    kafka_broker: str = Field(default="", description="Kafka bootstrap server.")
    kubernetes_context: str = Field(default="", description="Active kubectl context.")


__all__: list[str] = ["ModelRRHRuntimeTarget"]
