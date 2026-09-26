# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Only a trusted gateway's signed, matching read request mints authority."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.crypto.crypto_ed25519_signer import generate_keypair
from omnibase_core.models.contracts.subcontracts.model_event_bus_subcontract import (
    ModelEventBusSubcontract,
)
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_core.protocols.event_bus.protocol_event_message import (
    ProtocolEventMessage,
)
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.runtime.dispatch_envelope_context import (
    current_execution_graph_read_authority,
)
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    ExecutionGraphReadAuthorityError,
    TrustedExecutionGraphGatewayPolicy,
    TrustedGatewaySignerScope,
    VerifiedExecutionGraphReadAuthority,
    verify_signed_execution_graph_read_authority,
)
from omnibase_infra.runtime.models.model_execution_graph_read_ingress_config import (
    ModelExecutionGraphReadIngressConfig,
)
from tests.helpers.projection_tenant_authority import InMemoryKeyProvider

_SCOPE = TrustedGatewaySignerScope(
    runtime_id="trusted-api-gateway", realm="test", bus_id="graph-read"
)
_POLICY = TrustedExecutionGraphGatewayPolicy(scopes=frozenset({_SCOPE}))


def _signed_request(
    *,
    tenant_id: UUID | None = None,
    correlation_id: UUID | None = None,
    inner_tenant_id: str | None = None,
    inner_correlation_id: str | None = None,
    runtime_id: str = _SCOPE.runtime_id,
    realm: str = _SCOPE.realm,
    bus_id: str = _SCOPE.bus_id,
    event_type: str | None = None,
) -> tuple[ModelMessageEnvelope[dict[str, object]], InMemoryKeyProvider]:
    tenant = tenant_id or uuid4()
    correlation = correlation_id or uuid4()
    inner = ModelEventEnvelope[dict[str, object]](
        tenant_id=inner_tenant_id or str(tenant),
        correlation_id=UUID(inner_correlation_id)
        if inner_correlation_id
        else correlation,
        event_type=event_type,
        payload={
            "correlation_id": str(correlation),
            "cursor_mode": "latest",
            "source_cursors": None,
        },
    ).model_dump(mode="json")
    keys = generate_keypair()
    envelope = ModelMessageEnvelope[dict[str, object]].create_signed(
        realm=realm,
        runtime_id=runtime_id,
        bus_id=bus_id,
        trace_id=correlation,
        tenant_id=str(tenant),
        payload=inner,
        private_key=keys.private_key_bytes,
    )
    return envelope, InMemoryKeyProvider({runtime_id: keys.public_key_bytes})


@pytest.mark.unit
def test_trusted_signed_request_mints_sealed_authority() -> None:
    envelope, provider = _signed_request()

    authority = verify_signed_execution_graph_read_authority(
        envelope, provider, _POLICY
    )

    assert type(authority) is VerifiedExecutionGraphReadAuthority
    assert authority.tenant_id == UUID(envelope.tenant_id or "")
    assert authority.correlation_id == envelope.trace_id
    with pytest.raises(TypeError):
        VerifiedExecutionGraphReadAuthority(  # type: ignore[call-arg]
            tenant_id=authority.tenant_id,
            correlation_id=authority.correlation_id,
        )


@pytest.mark.unit
def test_payload_or_signed_tenant_tampering_is_refused() -> None:
    envelope, provider = _signed_request()
    tampered_payload = envelope.model_copy(
        update={"payload": {**envelope.payload, "tenant_id": str(uuid4())}}
    )
    tampered_tenant = envelope.model_copy(update={"tenant_id": str(uuid4())})

    with pytest.raises(ExecutionGraphReadAuthorityError, match="signature"):
        verify_signed_execution_graph_read_authority(
            tampered_payload, provider, _POLICY
        )
    with pytest.raises(ExecutionGraphReadAuthorityError, match="signature"):
        verify_signed_execution_graph_read_authority(tampered_tenant, provider, _POLICY)


@pytest.mark.unit
def test_unknown_or_wrong_scope_signer_is_refused() -> None:
    envelope, provider = _signed_request(runtime_id="other-gateway")
    with pytest.raises(ExecutionGraphReadAuthorityError, match="trusted gateway"):
        verify_signed_execution_graph_read_authority(envelope, provider, _POLICY)

    trusted_envelope, trusted_provider = _signed_request()
    empty_provider = InMemoryKeyProvider()
    with pytest.raises(ExecutionGraphReadAuthorityError, match="signature"):
        verify_signed_execution_graph_read_authority(
            trusted_envelope, empty_provider, _POLICY
        )
    assert trusted_provider.has_key(_SCOPE.runtime_id)


@pytest.mark.unit
def test_signed_conflicting_inner_tenant_or_correlation_is_refused() -> None:
    tenant_conflict, provider = _signed_request(inner_tenant_id=str(uuid4()))
    with pytest.raises(ExecutionGraphReadAuthorityError, match="tenant"):
        verify_signed_execution_graph_read_authority(tenant_conflict, provider, _POLICY)

    correlation_conflict, provider = _signed_request(inner_correlation_id=str(uuid4()))
    with pytest.raises(ExecutionGraphReadAuthorityError, match="correlation"):
        verify_signed_execution_graph_read_authority(
            correlation_conflict, provider, _POLICY
        )


@pytest.mark.unit
def test_unsigned_or_noncanonical_tenant_is_refused() -> None:
    envelope, provider = _signed_request()
    with pytest.raises(ExecutionGraphReadAuthorityError, match="signed"):
        verify_signed_execution_graph_read_authority(
            envelope.payload, provider, _POLICY
        )
    noncanonical, provider = _signed_request(inner_tenant_id="not-a-uuid")
    with pytest.raises(ExecutionGraphReadAuthorityError, match="tenant"):
        verify_signed_execution_graph_read_authority(noncanonical, provider, _POLICY)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_ingress_refuses_unsigned_and_binds_only_verified_authority() -> (
    None
):
    signed, provider = _signed_request(
        event_type="omnibase-infra.delegation-execution-graph-requested"
    )
    topic = "onex.cmd.omnibase-infra.delegation-execution-graph-requested.v1"
    bus = AsyncMock(spec=ProtocolEventBusSubscriber)
    bus._publish_raw_to_dlq = AsyncMock()
    engine = AsyncMock()
    observed: list[VerifiedExecutionGraphReadAuthority | None] = []

    async def capture(*_args: object, **_kwargs: object) -> None:
        observed.append(current_execution_graph_read_authority())

    engine.dispatch.side_effect = capture
    wiring = EventBusSubcontractWiring(
        event_bus=bus,
        dispatch_engine=engine,
        environment="test",
        node_name="graph-read",
        service="omnibase-infra",
        version="v1",
        execution_graph_read_ingress=ModelExecutionGraphReadIngressConfig(
            command_topic=topic, gateway_policy=_POLICY
        ),
        execution_graph_read_key_provider=provider,
    )
    callback = wiring._create_dispatch_callback(topic, "test.graph-read")
    unsigned = ModelEventMessage(
        topic=topic,
        key=b"k",
        value=json.dumps(signed.payload).encode(),
        headers=ModelEventHeaders(
            source="test", event_type="test", timestamp=datetime.now(UTC)
        ),
    )
    await callback(cast("ProtocolEventMessage", unsigned))
    engine.dispatch.assert_not_called()

    valid = unsigned.model_copy(
        update={"value": json.dumps(signed.model_dump(mode="json")).encode()}
    )
    await callback(cast("ProtocolEventMessage", valid))
    assert len(observed) == 1
    assert observed[0] is not None
    assert observed[0].tenant_id == UUID(signed.tenant_id or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_declared_signed_ingress_cannot_wire_without_verifier() -> None:
    topic = "onex.cmd.omnibase-infra.delegation-execution-graph-requested.v1"
    bus = AsyncMock(spec=ProtocolEventBusSubscriber)
    wiring = EventBusSubcontractWiring(
        event_bus=bus,
        dispatch_engine=AsyncMock(),
        environment="test",
        node_name="graph-read",
        service="omnibase-infra",
        version="v1",
    )
    subcontract = ModelEventBusSubcontract(
        version=ModelSemVer(major=1, minor=0, patch=0),
        subscribe_topics=[topic],
        signed_ingress_topics=[topic],
    )

    with pytest.raises(ProtocolConfigurationError, match="signed ingress"):
        await wiring.wire_subscriptions(subcontract, "graph-read")
    bus.subscribe.assert_not_called()
