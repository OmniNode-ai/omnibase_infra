# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Exercise fault-route budgets across the serialized consumer boundary (AC3)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import yaml

from omnibase_core.models.delegation.wire import ModelDelegationRequest
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef
from omnibase_infra.runtime.dogfood_delegation_fault_routes import (
    validate_dogfood_delegation_fault_request,
)


class _RecordingHandler:
    def __init__(self) -> None:
        self.requests: list[ModelDelegationRequest] = []

    async def handle(self, envelope: ModelEventEnvelope[object]) -> None:
        assert isinstance(envelope.payload, ModelDelegationRequest)
        self.requests.append(envelope.payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize(
    ("declared_timeout", "requested_timeout", "environment", "refusal"),
    [
        (300, 300, "dogfood", ""),
        (240, 240, "dogfood", ""),
        (300, 240, "dogfood", "exact timeout policy"),
        (300, 301, "dogfood", "exact timeout policy"),
        (300, 300, "dev", "dogfood lane"),
    ],
)
async def test_serialized_fault_request_obeys_declared_budget_before_dispatch(
    tmp_path: Path,
    status: int,
    declared_timeout: int,
    requested_timeout: int,
    environment: str,
    refusal: str,
) -> None:
    """Real wire coercion and route loading must precede handler invocation."""
    broker = "dogfood-broker:9092"
    backend_id = f"dogfood-fault-{status}"
    declaration = tmp_path / "ci_bus_lanes.yaml"
    declaration.write_text(
        yaml.safe_dump(
            {
                "lanes": {
                    "dogfood": {
                        "broker": broker,
                        "security_protocol": "PLAINTEXT",
                        "delegation_fault_routes": [
                            {
                                "backend_id": backend_id,
                                "endpoint_url": (
                                    f"http://dogfood-delegation-fault-{status}:8080"
                                    "/v1/chat/completions"
                                ),
                                "expected_http_status": status,
                                "requested_timeout_seconds": declared_timeout,
                                "max_attempts": 1,
                                "no_escalation": True,
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    trusted_bus_identity = SimpleNamespace(
        environment=environment, bootstrap_servers=broker
    )

    def guard(request: object) -> None:
        validate_dogfood_delegation_fault_request(
            request=request,
            event_bus=trusted_bus_identity,
            path_for_test=declaration,
        )

    handler = _RecordingHandler()
    callback = _make_dispatch_callback(
        handler,
        event_model=ModelHandlerRef(
            name="ModelDelegationRequest",
            module="omnibase_core.models.delegation.wire",
        ),
        pre_dispatch_guard=guard,
    )
    request = ModelDelegationRequest(
        prompt="exercise the declared fault-route budget",
        task_type="document",
        correlation_id=uuid4(),
        emitted_at=datetime.now(UTC),
        backend_id=backend_id,
        no_escalation=True,
        requested_timeout_seconds=requested_timeout,
    )
    envelope = ModelEventEnvelope[object](
        payload=request.model_dump(mode="json"),
        correlation_id=request.correlation_id,
        envelope_timestamp=request.emitted_at,
        event_type="omnibase-infra.delegation-request",
        payload_type="ModelDelegationRequest",
        source_tool="omn21604-integration",
    )
    wire_envelope = ModelEventEnvelope[object].model_validate_json(
        envelope.model_dump_json()
    )

    if refusal:
        with pytest.raises(InfraUnavailableError, match=refusal):
            await callback(wire_envelope)
        assert handler.requests == []
    else:
        await callback(wire_envelope)
        assert len(handler.requests) == 1
        delivered = handler.requests[0]
        assert isinstance(delivered, ModelDelegationRequest)
        assert delivered == request
        assert delivered.requested_timeout_seconds == declared_timeout
        assert delivered.backend_id == backend_id
        assert delivered.no_escalation is True
        assert delivered.correlation_id == wire_envelope.correlation_id
