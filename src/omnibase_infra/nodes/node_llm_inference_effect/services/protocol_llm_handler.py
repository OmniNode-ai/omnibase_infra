# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Protocol defining the minimal LLM handler interface used by ServiceLlmMetricsPublisher.

Decouples the service from concrete handler types via a structural Protocol.

Related:
    - OMN-2443: Wire NodeLlmInferenceEffect to emit llm-call-completed events
    - ServiceLlmMetricsPublisher: Consumer of this Protocol
    - HandlerLlmOpenaiCompatible: Satisfies this Protocol
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.models.llm.model_llm_inference_response import (
        ModelLlmInferenceResponse,
    )
    from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
        ModelLlmInferenceRequest,
    )


class ProtocolLlmHandler(Protocol):
    """Structural protocol for LLM inference handlers.

    Any object that provides a ``handle`` coroutine accepting an LLM
    inference request and returning a ``ModelLlmInferenceResponse``
    satisfies this protocol.  Implementations may optionally expose a
    ``last_call_metrics`` attribute; ``ServiceLlmMetricsPublisher`` reads it
    via ``getattr(handler, "last_call_metrics", None)`` so it is not required
    by the protocol itself.

    Implementors:
        - ``HandlerLlmOpenaiCompatible`` -- satisfies nominally
    """

    async def handle(
        self,
        request: ModelLlmInferenceRequest,
        correlation_id: UUID | None = None,
    ) -> ModelLlmInferenceResponse:
        """Execute an LLM inference call and return the response.

        Args:
            request: LLM inference request parameters.
            correlation_id: Optional correlation ID for distributed tracing.
                If ``None``, implementations may generate their own UUID.

        Returns:
            ``ModelLlmInferenceResponse`` from the underlying provider.
        """
        ...


__all__: list[str] = ["ProtocolLlmHandler"]
