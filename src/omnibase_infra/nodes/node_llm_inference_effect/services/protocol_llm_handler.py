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
    not related by inheritance.  The Protocol therefore uses ``Any`` for the
    request parameter so that both handlers satisfy it structurally without a
    ``# type: ignore``.  See ADR docs/decisions/adr-any-type-pydantic-workaround.md.

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
        Implementors that accept a structurally compatible type (i.e. a class
        with the same fields, such as ``effects.models.ModelLlmInferenceRequest``
        used by ``HandlerLlmOllama``) will satisfy this protocol at runtime
        because Python's structural subtyping checks field/method presence, not
        class identity.  The two request classes are not related by inheritance
        but share identical fields; see the module docstring and
        ``docs/decisions/adr-any-type-pydantic-workaround.md`` for details.

    Implementors:
        - ``HandlerLlmOpenaiCompatible`` -- satisfies structurally
        - ``HandlerLlmOllama`` -- satisfies structurally (accepts structurally
          compatible request type from ``effects.models``)
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
