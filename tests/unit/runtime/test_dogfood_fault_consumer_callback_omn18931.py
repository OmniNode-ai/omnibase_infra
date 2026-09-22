# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Raw delegation-request consumer guard regression for OMN-18931."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from omnibase_core.models.delegation.wire import ModelDelegationRequest
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef
from omnibase_infra.runtime.dogfood_delegation_fault_routes import (
    validate_dogfood_delegation_fault_request,
)

pytestmark = pytest.mark.unit


class _MustNotReachWorkflow:
    def __init__(self) -> None:
        self.called = False

    def handle(self, request: ModelDelegationRequest) -> list[object]:
        self.called = True
        return []


def _write_route(path: Path) -> None:
    path.write_text(
        """lanes:
  dogfood:
    broker: dogfood-broker:9092
    security_protocol: PLAINTEXT
    delegation_fault_routes:
      - backend_id: dogfood-fault-429
        endpoint_url: http://dogfood-delegation-fault-429:8080/v1/chat/completions
        expected_http_status: 429
        requested_timeout_seconds: 240
        max_attempts: 1
        no_escalation: true
""",
        encoding="utf-8",
    )


def _request() -> ModelDelegationRequest:
    return ModelDelegationRequest(
        prompt="produce a bounded fault control",
        task_type="document",
        correlation_id=uuid4(),
        emitted_at=datetime.now(UTC),
        backend_id="dogfood-fault-429",
        no_escalation=True,
        requested_timeout_seconds=240,
    )


@pytest.mark.asyncio
async def test_raw_consumer_request_is_rejected_before_workflow_on_bad_trusted_route(
    tmp_path: Path,
) -> None:
    route_path = tmp_path / "ci_bus_lanes.yaml"
    _write_route(route_path)
    handler = _MustNotReachWorkflow()
    bus = SimpleNamespace(environment="dev", bootstrap_servers="dogfood-broker:9092")

    def guard(request: object) -> None:
        validate_dogfood_delegation_fault_request(
            request=request,
            event_bus=bus,
            path_for_test=route_path,
        )

    callback = _make_dispatch_callback(
        handler,  # type: ignore[arg-type]
        event_model=ModelHandlerRef(
            name="ModelDelegationRequest",
            module="omnibase_core.models.delegation.wire",
        ),
        pre_dispatch_guard=guard,
    )
    request = _request()
    envelope = ModelEventEnvelope[object](
        payload=request.model_dump(mode="json"),
        correlation_id=request.correlation_id,
        envelope_timestamp=datetime.now(UTC),
        event_type="omnibase-infra.delegation-request",
        payload_type="ModelDelegationRequest",
        source_tool="omn18931-test",
    )

    with pytest.raises(InfraUnavailableError, match="dogfood lane"):
        await callback(envelope)
    assert handler.called is False


def test_auto_wiring_selects_guard_only_for_delegation_request_entry() -> None:
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _delegation_fault_pre_dispatch_guard,
    )

    event_model = ModelHandlerRef(
        name="ModelDelegationRequest",
        module="omnibase_core.models.delegation.wire",
    )
    selected = _delegation_fault_pre_dispatch_guard(
        contract=SimpleNamespace(name="node_delegation_orchestrator"),
        entry=SimpleNamespace(event_model=event_model),
        event_bus=SimpleNamespace(environment="dogfood", bootstrap_servers="x:1"),
    )
    non_delegation = _delegation_fault_pre_dispatch_guard(
        contract=SimpleNamespace(name="node_other"),
        entry=SimpleNamespace(event_model=event_model),
        event_bus=None,
    )

    assert selected is not None
    assert non_delegation is None
