# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for HandlerStampTenantId."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_tenant_ingress_effect.handlers.handler_stamp_tenant_id import (
    HandlerStampTenantId,
)


def test_stamp_overwrites_forged_payload_tenant_id() -> None:
    handler = HandlerStampTenantId()
    envelope = ModelEventEnvelope[dict](payload={"tenant_id": "evil", "prompt": "hi"})

    stamped = handler.stamp(envelope, "acme")

    assert stamped.payload["tenant_id"] == "acme"
    assert stamped.payload["prompt"] == "hi"


def test_stamp_adds_tenant_id_when_missing() -> None:
    handler = HandlerStampTenantId()
    envelope = ModelEventEnvelope[dict](payload={"prompt": "hi"})

    stamped = handler.stamp(envelope, "beta")

    assert stamped.payload["tenant_id"] == "beta"


def test_stamp_preserves_correlation_id() -> None:
    handler = HandlerStampTenantId()
    cid = uuid4()
    envelope = ModelEventEnvelope[dict](payload={}, correlation_id=cid)

    stamped = handler.stamp(envelope, "acme")

    assert stamped.correlation_id == cid


@pytest.mark.asyncio
async def test_handle_requires_tenant_slug_metadata() -> None:
    """The dispatch entrypoint has no topic context -- it must fail closed,
    never silently stamp an unverified/absent slug."""
    handler = HandlerStampTenantId()
    envelope = ModelEventEnvelope[dict](payload={"prompt": "hi"})

    with pytest.raises(ValueError, match="tenant_slug"):
        await handler.handle(envelope)


@pytest.mark.asyncio
async def test_handle_dispatch_entrypoint_returns_model_handler_output() -> None:
    from omnibase_core.models.core.model_envelope_metadata import (
        ModelEnvelopeMetadata,
    )

    handler = HandlerStampTenantId()
    envelope = ModelEventEnvelope[dict](
        payload={"prompt": "hi"},
        metadata=ModelEnvelopeMetadata(tags={"tenant_slug": "acme"}),
    )

    result = await handler.handle(envelope)

    assert isinstance(result, ModelHandlerOutput)
    assert result.result is not None
    assert result.result.payload["tenant_id"] == "acme"
