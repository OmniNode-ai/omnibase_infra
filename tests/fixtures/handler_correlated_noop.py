# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test-only fixture handler that round-trips ``correlation_id`` (OMN-17295).

This module is NOT production code. It is the delegate-shaped sibling of
``handler_proof_noop``: its request model declares ``correlation_id``, so
``RuntimeLocal`` propagates the caller's id onto the terminal event exactly as
the real ``ModelDelegateSkillRequest`` does (that model is frozen, so
``RuntimeLocal``'s event-driven correlation overwrite is refused and the CLI's
minted id survives onto the wire — verified live).

``handler_proof_noop``'s request model declares no ``correlation_id`` at all,
so a run against it produces a terminal stamped with a runtime-minted id that
no caller can attribute. That is a legitimate refusal under the OMN-17295
correlation join, which makes it the wrong stand-in for tests whose subject is
the *shape* of the delegate receipt. Production handlers MUST NOT import from
or depend on this module.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelCorrelatedNoopRequest(BaseModel):
    """Test-only input model carrying the caller's correlation id."""

    correlation_id: str = ""
    prompt: str = ""
    task_type: str = ""


class ModelCorrelatedNoopResult(BaseModel):
    """Test-only output model paired with :class:`ModelCorrelatedNoopRequest`."""

    status: str = "success"
    correlation_id: str
    echoed_prompt: str


class HandlerCorrelatedNoop:
    """Echo the request back, preserving the correlation id it arrived with."""

    def handle(self, request: ModelCorrelatedNoopRequest) -> ModelCorrelatedNoopResult:
        return ModelCorrelatedNoopResult(
            status="success",
            correlation_id=request.correlation_id,
            echoed_prompt=request.prompt,
        )
