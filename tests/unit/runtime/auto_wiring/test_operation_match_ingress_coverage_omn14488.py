# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14488 (operation_match extension) — declared input_model must accept the
producer's real payload for operation_match contracts.

The sibling gate (`test_dispatcher_route_coverage_omn14488.py`) covers
`payload_type_match` contracts, where the dispatch engine type-scopes on the
declared `event_model`. `operation_match` contracts route purely on the envelope
operation and DO NOT type-scope the payload at dispatch — so that gate is blind to
them. Their load-bearing validation is instead `validate_runtime_local_ingress_payload`,
which resolves `contract.input_model` and `model_validate`s the payload against it.

A contract whose declared `input_model` cannot validate the producer's real payload
(the OMN-14489 stub-vs-canonical defect: local stub required `agent_id` + `extra="forbid"`
while the producer publishes the rich `omnibase_core` `ModelInvocationCommand`) does NOT
DLQ — `operation_match` skips dispatch validation — but the ingress path silently rejects
every payload, starving the node (`remote-agent-invoke.v1` HW=19 IN / `agent-task-lifecycle.v1`
HW=0 OUT). This gate makes that class fail loudly: the declared input_model MUST accept a
valid producer payload, and a mismatched/stub input_model MUST reject it (non-vacuity).
"""

from __future__ import annotations

import enum
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from omnibase_core.models.delegation.model_invocation_command import (
    ModelInvocationCommand,
)
from omnibase_infra.runtime.runtime_local_ingress import (
    ModelRuntimeLocalIngressRoute,
    validate_runtime_local_ingress_payload,
)

_THIS_MODULE = (
    "tests.unit.runtime.auto_wiring.test_operation_match_ingress_coverage_omn14488"
)
_CANONICAL_MODULE = "omnibase_core.models.delegation.model_invocation_command"


class _StubInvocationCommand(BaseModel):
    """A stub/mismatched input_model of the kind OMN-14489 removed: requires a field
    the producer never sends (`agent_id`) and forbids the rich fields it does send."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: object = Field(...)
    agent_id: object = Field(...)
    payload: dict[str, object] = Field(default_factory=dict)


def _first_enum_value(annotation: object) -> object:
    candidates = list(getattr(annotation, "__args__", None) or [annotation])
    for cand in candidates:
        if isinstance(cand, type) and issubclass(cand, enum.Enum):
            return next(iter(cand))
    raise AssertionError(f"no Enum member in {annotation!r}")


def _producer_invocation_payload() -> dict[str, object]:
    """The RICH ModelInvocationCommand the producer actually publishes."""
    kind = _first_enum_value(
        ModelInvocationCommand.model_fields["invocation_kind"].annotation
    )
    agent_protocol = _first_enum_value(
        ModelInvocationCommand.model_fields["agent_protocol"].annotation
    )
    return ModelInvocationCommand(
        task_id=uuid4(),
        correlation_id=uuid4(),
        invocation_kind=kind,
        agent_protocol=agent_protocol,
        target_ref="agent://test-target",
    ).model_dump(mode="json")


def _operation_match_route(
    *, input_model_module: str, input_model_name: str
) -> ModelRuntimeLocalIngressRoute:
    """An operation_match ingress route pointed at the given input_model."""
    return ModelRuntimeLocalIngressRoute(
        node_name="node_under_test",
        contract_name="node_under_test",
        command_topic="onex.cmd.omnibase-infra.remote-agent-invoke.v1",
        event_type=None,
        terminal_event=None,
        contract_path="/fake/contract.yaml",
        package_name="omnibase_infra",
        input_model_module=input_model_module,
        input_model_name=input_model_name,
    )


@pytest.mark.unit
def test_canonical_input_model_accepts_producer_payload() -> None:
    """LOAD-BEARING: when the operation_match contract's input_model is the canonical
    class the producer uses, the runtime ingress accepts the real payload with no field
    drop. This is the GREEN target."""
    route = _operation_match_route(
        input_model_module=_CANONICAL_MODULE, input_model_name="ModelInvocationCommand"
    )
    payload = _producer_invocation_payload()
    # Non-vacuity guard: the producer payload is dumped WITHOUT exclude_none, so it
    # carries an explicit ``model_backend: None`` that the canonical model drops on
    # re-dump. A passthrough (input_model unresolved -> payload returned unchanged)
    # would leave it in; its absence below proves the ingress seam actually resolved
    # and model_validate'd against the declared input_model rather than short-circuiting.
    assert "model_backend" in payload and payload["model_backend"] is None
    normalized = validate_runtime_local_ingress_payload(route, payload)
    for field in ("task_id", "invocation_kind", "target_ref"):
        assert field in normalized, f"{field} dropped by input_model validation"
    assert "model_backend" not in normalized, (
        "ingress seam must have re-normalized through the declared input_model "
        "(exclude_none drops the null model_backend) — not passed the payload through."
    )


@pytest.mark.unit
def test_stub_input_model_rejects_producer_payload() -> None:
    """NON-VACUITY (RED against a stub/mismatched contract): an operation_match contract
    whose declared input_model is a stub that does not match the producer's payload MUST
    reject it at ingress (the silent-starvation defect OMN-14489 fixed) — proving the gate
    discriminates, not merely that the canonical case passes."""
    route = _operation_match_route(
        input_model_module=_THIS_MODULE, input_model_name="_StubInvocationCommand"
    )
    with pytest.raises(ValidationError):
        validate_runtime_local_ingress_payload(route, _producer_invocation_payload())
