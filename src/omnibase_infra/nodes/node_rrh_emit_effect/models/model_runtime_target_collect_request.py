# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request model for ``HandlerRuntimeTargetCollect``."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeTargetCollectRequest(BaseModel):
    """Overrides for runtime deployment target collection.

    Every field is optional — an empty value tells the handler to fall
    back to its corresponding environment variable.

    Attributes:
        environment: Target environment override (falls back to
            ``ONEX_ENVIRONMENT``).
        kafka_broker: Kafka bootstrap server override (falls back to
            ``KAFKA_BOOTSTRAP_SERVERS``).
        kubernetes_context: kubectl context override (falls back to
            ``KUBECONFIG_CONTEXT``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    environment: str = Field(default="", description="Target environment override.")
    kafka_broker: str = Field(
        default="", description="Kafka bootstrap server override."
    )
    kubernetes_context: str = Field(default="", description="kubectl context override.")


__all__: list[str] = ["ModelRuntimeTargetCollectRequest"]
