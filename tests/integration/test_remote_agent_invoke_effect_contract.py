# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary contract test for node_remote_agent_invoke_effect (OMN-14489).

The contract's input_model/output_model MUST be the CANONICAL omnibase_core
delegation classes that the producer (the delegation orchestrator, via its
`InvocationCommand` publish) and the consumer (`HandlerA2ATask`) actually use —
not local stub classes.

WHY (live signal): the contract previously declared local stub models
(`ModelInvocationCommand` required `agent_id` and `extra="forbid"`;
`ModelAgentTaskLifecycleEvent` used a different `status` enum). The producer
publishes the RICH omnibase_core `ModelInvocationCommand` (task_id /
invocation_kind / target_ref / agent_protocol / model_backend, no agent_id).
Because `routing_strategy=operation_match` skips payload-type validation at
dispatch it did not DLQ, BUT `validate_runtime_local_ingress_payload` resolves
`contract.input_model` and `model_validate`s the payload against it — the stub
rejected the real payload (missing `agent_id`; forbidden extras), so the
remote-agent invoke starved: `remote-agent-invoke.v1` HW=19 IN /
`agent-task-lifecycle.v1` HW=0 OUT on the stability lane.

These tests drive the PRODUCER's real payload through the REAL runtime ingress
validation path against the CONTRACT-declared model (not a hardcoded class), so
they go RED against the stub contract (exists-but-wrong) and GREEN once the
contract points at the canonical classes. Single cross-boundary suite — not two
independent unit suites (OMN-14208 shape).
"""

from __future__ import annotations

import enum
import importlib
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
import yaml
from pydantic import BaseModel

from omnibase_core.models.delegation.model_agent_task_lifecycle_event import (
    ModelAgentTaskLifecycleEvent,
)
from omnibase_core.models.delegation.model_invocation_command import (
    ModelInvocationCommand,
)
from omnibase_infra.runtime.runtime_local_ingress import (
    ModelRuntimeLocalIngressRoute,
    validate_runtime_local_ingress_payload,
)

CONTRACT_PATH = (
    Path("src")
    / "omnibase_infra"
    / "nodes"
    / "node_remote_agent_invoke_effect"
    / "contract.yaml"
)

_CANONICAL_INPUT_MODULE = "omnibase_core.models.delegation.model_invocation_command"
_CANONICAL_OUTPUT_MODULE = (
    "omnibase_core.models.delegation.model_agent_task_lifecycle_event"
)


def _load_contract() -> dict[str, object]:
    with CONTRACT_PATH.open() as f:
        return cast("dict[str, object]", yaml.safe_load(f))


def _contract_model_cls(field: str) -> type[BaseModel]:
    """Import the model class the CONTRACT declares for ``field`` (input/output)."""
    data = _load_contract()
    ref = data[field]
    module = importlib.import_module(ref["module"])
    cls = getattr(module, ref["name"])
    assert isinstance(cls, type) and issubclass(cls, BaseModel)
    return cls


def _first_enum_value(annotation: object) -> object:
    """Return the first member of an Enum annotation (handles `Enum | None`)."""
    candidates = [annotation]
    args = getattr(annotation, "__args__", None)
    if args:
        candidates = list(args)
    for cand in candidates:
        if isinstance(cand, type) and issubclass(cand, enum.Enum):
            return next(iter(cand))
    raise AssertionError(f"no Enum member found in annotation {annotation!r}")


def _producer_invocation_payload() -> dict[str, object]:
    """The RICH ModelInvocationCommand the producer publishes on remote-agent-invoke.v1."""
    kind = _first_enum_value(
        ModelInvocationCommand.model_fields["invocation_kind"].annotation
    )
    # invocation_kind=AGENT requires agent_protocol (model validator); set it so the
    # payload is a genuinely valid producer command regardless of enum ordering.
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


def _handler_lifecycle_event_payload() -> dict[str, object]:
    """The RICH ModelAgentTaskLifecycleEvent HandlerA2ATask emits on agent-task-lifecycle.v1."""
    lifecycle = _first_enum_value(
        ModelAgentTaskLifecycleEvent.model_fields["lifecycle_type"].annotation
    )
    return ModelAgentTaskLifecycleEvent(
        task_id=uuid4(),
        correlation_id=uuid4(),
        lifecycle_type=lifecycle,
        occurred_at=datetime.now(UTC),
    ).model_dump(mode="json")


def _ingress_route_from_contract() -> ModelRuntimeLocalIngressRoute:
    data = _load_contract()
    return ModelRuntimeLocalIngressRoute(
        node_name="node_remote_agent_invoke_effect",
        contract_name="node_remote_agent_invoke_effect",
        command_topic=data["event_bus"]["subscribe_topics"][0],
        event_type=None,
        terminal_event=None,
        contract_path=str(CONTRACT_PATH),
        package_name="omnibase_infra",
        input_model_module=data["input_model"]["module"],
        input_model_name=data["input_model"]["name"],
    )


@pytest.mark.integration
class TestRemoteAgentInvokeEffectContract:
    def test_contract_declares_canonical_core_models(self) -> None:
        data = _load_contract()
        assert data["input_model"]["module"] == _CANONICAL_INPUT_MODULE
        assert data["output_model"]["module"] == _CANONICAL_OUTPUT_MODULE

    def test_contract_input_model_is_the_class_the_handler_consumes(self) -> None:
        """Seam match: the contract's declared input_model class is IDENTICAL to the
        ModelInvocationCommand HandlerA2ATask imports and consumes."""
        handler_mod = importlib.import_module(
            "omnibase_infra.nodes.node_remote_agent_invoke_effect.handlers.handler_a2a_task"
        )
        assert _contract_model_cls("input_model") is handler_mod.ModelInvocationCommand
        assert (
            _contract_model_cls("output_model")
            is handler_mod.ModelAgentTaskLifecycleEvent
        )

    def test_producer_payload_validates_through_runtime_ingress(self) -> None:
        """LOAD-BEARING cross-boundary (prove-RED-against-exists-but-wrong): the
        producer's real rich payload must pass `validate_runtime_local_ingress_payload`
        against the contract-declared input_model — the exact runtime path that
        starved the node. RED against the stub contract (stub rejects: missing
        agent_id / forbidden extras); GREEN against the canonical contract."""
        route = _ingress_route_from_contract()
        payload = _producer_invocation_payload()
        normalized = validate_runtime_local_ingress_payload(route, payload)
        # No field drop: the rich invocation fields survive validation.
        for field in ("task_id", "invocation_kind", "target_ref"):
            assert field in normalized, f"{field} dropped by input_model validation"

    def test_contract_output_model_validates_handler_event(self) -> None:
        """The contract's declared output_model must validate the rich lifecycle event
        HandlerA2ATask actually emits (RED against the stub output_model / wrong enum)."""
        output_cls = _contract_model_cls("output_model")
        output_cls.model_validate(_handler_lifecycle_event_payload())
