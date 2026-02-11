# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Input model for the RRH emit effect node."""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelRRHEmitRequest(BaseModel):
    """Request to collect environment data for RRH validation.

    Attributes:
        repo_path: Absolute path to the repository root to inspect.
        environment: Target environment label override (defaults to ``"dev"``).
        kafka_broker: Kafka bootstrap server override (empty = auto-detect from env).
        kubernetes_context: kubectl context override (empty = auto-detect).
        correlation_id: Distributed tracing correlation ID.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    repo_path: str = Field(..., description="Absolute path to repository root.")
    environment: str = Field(default="dev", description="Target environment label.")
    kafka_broker: str = Field(
        default="", description="Kafka bootstrap server override."
    )
    kubernetes_context: str = Field(default="", description="kubectl context override.")
    correlation_id: UUID = Field(
        default_factory=uuid4, description="Correlation ID for tracing."
    )


__all__: list[str] = ["ModelRRHEmitRequest"]
