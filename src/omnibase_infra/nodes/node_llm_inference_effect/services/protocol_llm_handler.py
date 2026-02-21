# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Protocol defining the minimal LLM handler interface used by ServiceLlmMetricsPublisher.

Replacing the ``HandlerLlmOpenaiCompatible | HandlerLlmOllama`` union type with
a structural Protocol keeps the union count within the pre-commit limit and
decouples the service from concrete handler types.

Note on request type:
    ``HandlerLlmOpenaiCompatible`` uses
    ``node_llm_inference_effect.models.ModelLlmInferenceRequest`` while
    ``HandlerLlmOllama`` uses ``effects.models.ModelLlmInferenceRequest``.
    These are distinct classes that share the same fields at runtime but are
    not related by inheritance.  The Protocol uses
    ``node_llm_inference_effect.models.ModelLlmInferenceRequest`` as the
    declared parameter type; ``HandlerLlmOllama`` satisfies the protocol
    structurally because its request class has identical fields.  See ADR
    docs/decisions/adr-any-type-pydantic-workaround.md.

Related:
    - OMN-2443: Wire NodeLlmInferenceEffect to emit llm-call-completed events
    - ServiceLlmMetricsPublisher: Consumer of this Protocol
    - HandlerLlmOpenaiCompatible: Satisfies this Protocol (structurally)
    - HandlerLlmOllama: Satisfies this Protocol (structurally)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.nodes.effects.models.model_llm_inference_response import (
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

    Note on request type annotation:
        The ``handle`` method is typed with
        ``node_llm_inference_effect.models.ModelLlmInferenceRequest``.
        ``HandlerLlmOpenaiCompatible`` uses the same type and satisfies this
        protocol nominally.  ``HandlerLlmOllama`` uses the structurally
        identical ``effects.models.ModelLlmInferenceRequest`` (different class,
        same fields, no inheritance relationship); callers that pass a
        ``HandlerLlmOllama`` instance must cast it to ``ProtocolLlmHandler``
        at the call site — see ``register_ollama_with_metrics`` in
        ``registry_infra_llm_inference_effect.py``.  This is the
        ADR-approved approach: ``cast`` instead of ``Any``; see
        ``docs/decisions/adr-any-type-pydantic-workaround.md``.

    Implementors:
        - ``HandlerLlmOpenaiCompatible`` -- satisfies nominally
        - ``HandlerLlmOllama`` -- satisfies structurally; use ``cast`` at
          call site due to dual-request-type naming conflict
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
