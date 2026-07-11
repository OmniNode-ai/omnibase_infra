# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that collects runtime target context for RRH validation.

Captures deployment environment, Kafka broker address, and Kubernetes
context.  Values come from request overrides or environment variables.
"""

from __future__ import annotations

import os

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.rrh.model_rrh_runtime_target import ModelRRHRuntimeTarget
from omnibase_infra.nodes.node_rrh_emit_effect.models.model_runtime_target_collect_request import (
    ModelRuntimeTargetCollectRequest,
)


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
        self, payload: ModelRuntimeTargetCollectRequest
    ) -> ModelRRHRuntimeTarget:
        """Collect runtime target from overrides or environment.

        Args:
            payload: Optional environment/Kafka/Kubernetes overrides. Any
                empty field falls back to its corresponding env var.

        Returns:
            Populated ``ModelRRHRuntimeTarget``.
        """
        environment = payload.environment
        kafka_broker = payload.kafka_broker
        kubernetes_context = payload.kubernetes_context
        return ModelRRHRuntimeTarget(
            environment=environment or os.environ.get("ONEX_ENVIRONMENT", "local"),
            kafka_broker=kafka_broker or os.environ.get("KAFKA_BOOTSTRAP_SERVERS", ""),
            kubernetes_context=kubernetes_context
            or os.environ.get("KUBECONFIG_CONTEXT", ""),
        )


__all__: list[str] = ["HandlerRuntimeTargetCollect"]
