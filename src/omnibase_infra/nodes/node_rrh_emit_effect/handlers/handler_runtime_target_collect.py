# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler that collects runtime target context for RRH validation.

Captures deployment environment, Kafka broker address, and Kubernetes
context.  Values come from request overrides or environment variables.
"""

from __future__ import annotations

import os

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.rrh.model_rrh_environment_data import ModelRRHRuntimeTarget


class HandlerRuntimeTargetCollect:
    """Collect runtime deployment target context.

    Gathers: environment, kafka_broker, kubernetes_context.
    Uses request overrides when provided; falls back to env vars.

    Attributes:
        handler_type: ``INFRA_HANDLER``
        handler_category: ``EFFECT``
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        *,
        environment: str = "",
        kafka_broker: str = "",
        kubernetes_context: str = "",
    ) -> ModelRRHRuntimeTarget:
        """Collect runtime target from overrides or environment.

        Args:
            environment: Target environment override.
            kafka_broker: Kafka bootstrap server override.
            kubernetes_context: kubectl context override.

        Returns:
            Populated ``ModelRRHRuntimeTarget``.
        """
        return ModelRRHRuntimeTarget(
            environment=environment or os.environ.get("ENVIRONMENT", "dev"),
            kafka_broker=kafka_broker or os.environ.get("KAFKA_BOOTSTRAP_SERVERS", ""),
            kubernetes_context=kubernetes_context
            or os.environ.get("KUBECONFIG_CONTEXT", ""),
        )


__all__: list[str] = ["HandlerRuntimeTargetCollect"]
