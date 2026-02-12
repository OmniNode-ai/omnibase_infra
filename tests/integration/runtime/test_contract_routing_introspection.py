# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for contract routing introspection (OMN-2081).

Tests that node introspection events dispatch through contract-loaded
handler routing. Verifies:

1. Contract YAML declares the correct handler routing for introspection events
2. Dispatch engine routes introspection events to the correct dispatcher
3. Introspection handler receives correct dispatch context (time, correlation_id)
4. Contract handler routing matches runtime-importable module paths

Related:
    - OMN-2081: Investor demo - runtime contract routing verification
    - src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml
    - src/omnibase_infra/runtime/service_message_dispatch_engine.py
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_context import ModelDispatchContext
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.dispatch_context_enforcer import DispatchContextEnforcer


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml).

    Note: Canonical implementation lives in ``tests.helpers.path_utils.find_project_root``.
    This local wrapper passes the correct start directory.
    """
    from tests.helpers.path_utils import find_project_root

    return find_project_root(start=Path(__file__).resolve().parent)


# Path to the registration orchestrator contract
try:
    PROJECT_ROOT = _find_project_root()
    CONTRACT_PATH = (
        PROJECT_ROOT
        / "src"
        / "omnibase_infra"
        / "nodes"
        / "node_registration_orchestrator"
        / "contract.yaml"
    )
except RuntimeError:
    PROJECT_ROOT = None
    CONTRACT_PATH = None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(CONTRACT_PATH is None, reason="Project root not found"),
]


# =============================================================================
# Context-Capturing Dispatcher (local helper)
# =============================================================================


class ContextCapturingDispatcher:
    """Test dispatcher that captures context and envelope for assertions."""

    def __init__(
        self,
        dispatcher_id: str,
        node_kind: EnumNodeKind,
        category: EnumMessageCategory = EnumMessageCategory.EVENT,
        message_types: set[str] | None = None,
    ) -> None:
        self._dispatcher_id = dispatcher_id
        self._node_kind = node_kind
        self._category = category
        self._message_types = message_types or set()

        self.captured_context: ModelDispatchContext | None = None
        self.captured_envelope: object | None = None
        self.invocation_count: int = 0

    @property
    def dispatcher_id(self) -> str:
        return self._dispatcher_id

    @property
    def category(self) -> EnumMessageCategory:
        return self._category

    @property
    def message_types(self) -> set[str]:
        return self._message_types

    @property
    def node_kind(self) -> EnumNodeKind:
        return self._node_kind

    async def handle(
        self,
        envelope: object,
        context: ModelDispatchContext | None = None,
    ) -> ModelDispatchResult:
        """Handle the message and capture context for assertions."""
        self.captured_envelope = envelope
        self.captured_context = context
        self.invocation_count += 1

        return ModelDispatchResult(
            dispatch_id=uuid4(),
            status=EnumDispatchStatus.SUCCESS,
            topic="test.events.v1",
            dispatcher_id=self._dispatcher_id,
            message_type=type(envelope).__name__ if envelope else None,
            started_at=datetime.now(UTC),
        )


# =============================================================================
# Tests
# =============================================================================


class TestContractDeclaresIntrospectionRouting:
    """Tests that the contract YAML declares the correct handler routing."""

    def test_contract_declares_introspection_handler_routing(self) -> None:
        """Load contract.yaml and verify handler_routing has an entry for
        ModelNodeIntrospectionEvent mapped to HandlerNodeIntrospected.
        """
        assert CONTRACT_PATH.exists(), f"Contract not found at {CONTRACT_PATH}"

        with open(CONTRACT_PATH, encoding="utf-8") as f:
            contract = yaml.safe_load(f)

        # Verify handler_routing section exists
        assert "handler_routing" in contract, "Missing handler_routing in contract"
        handler_routing = contract["handler_routing"]

        assert handler_routing["routing_strategy"] == "payload_type_match"
        assert "handlers" in handler_routing

        handlers = handler_routing["handlers"]
        assert isinstance(handlers, list)
        assert len(handlers) > 0

        # Find the introspection handler entry
        introspection_entry = None
        for entry in handlers:
            event_model = entry.get("event_model", {})
            if event_model.get("name") == "ModelNodeIntrospectionEvent":
                introspection_entry = entry
                break

        assert introspection_entry is not None, (
            "No handler routing entry found for ModelNodeIntrospectionEvent. "
            f"Available entries: {[e.get('event_model', {}).get('name') for e in handlers]}"
        )

        # Verify it maps to HandlerNodeIntrospected
        handler_def = introspection_entry["handler"]
        assert handler_def["name"] == "HandlerNodeIntrospected"
        assert "module" in handler_def
        assert "handler_node_introspected" in handler_def["module"]


class TestDispatchEngineRoutesIntrospection:
    """Tests that the dispatch engine routes introspection events correctly."""

    @pytest.mark.asyncio
    async def test_dispatch_engine_routes_introspection_to_correct_dispatcher(
        self,
    ) -> None:
        """Register a ContextCapturingDispatcher for ORCHESTRATOR node kind
        matching ModelNodeIntrospectionEvent on EVENT category, dispatch a
        mock envelope, and verify the dispatcher was invoked.
        """
        from omnibase_infra.runtime.service_message_dispatch_engine import (
            MessageDispatchEngine,
        )

        engine = MessageDispatchEngine()

        # Create a capturing dispatcher for introspection events
        dispatcher = ContextCapturingDispatcher(
            dispatcher_id="test-introspection-orchestrator",
            node_kind=EnumNodeKind.ORCHESTRATOR,
            category=EnumMessageCategory.EVENT,
            message_types={"ModelNodeIntrospectionEvent"},
        )

        # Register the dispatcher with the engine
        engine.register_dispatcher(
            dispatcher_id=dispatcher.dispatcher_id,
            dispatcher=dispatcher.handle,
            category=EnumMessageCategory.EVENT,
            message_types={"ModelNodeIntrospectionEvent"},
            node_kind=EnumNodeKind.ORCHESTRATOR,
        )

        # Register a route for the topic
        from omnibase_infra.models.dispatch.model_dispatch_route import (
            ModelDispatchRoute,
        )

        route = ModelDispatchRoute(
            route_id="introspection-route",
            topic_pattern="onex.evt.platform.node-introspection.v1",
            message_category=EnumMessageCategory.EVENT,
            dispatcher_id=dispatcher.dispatcher_id,
        )
        engine.register_route(route)

        engine.freeze()

        # Create a mock envelope with the expected event_type
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            correlation_id=uuid4(),
            event_type="ModelNodeIntrospectionEvent",
            payload={"node_id": str(uuid4()), "node_type": "EFFECT"},
        )

        # Dispatch
        result = await engine.dispatch(
            topic="onex.evt.platform.node-introspection.v1",
            envelope=envelope,
        )

        # Verify dispatch succeeded
        assert result.status == EnumDispatchStatus.SUCCESS
        assert dispatcher.invocation_count == 1

    @pytest.mark.asyncio
    async def test_introspection_handler_receives_correct_context(self) -> None:
        """Verify dispatching an introspection event to an orchestrator
        dispatcher provides a context with ``now`` set (time injection)
        and correct correlation_id propagation.
        """
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )

        enforcer = DispatchContextEnforcer()
        correlation_id = uuid4()

        dispatcher = ContextCapturingDispatcher(
            dispatcher_id="test-introspection-context",
            node_kind=EnumNodeKind.ORCHESTRATOR,
            category=EnumMessageCategory.EVENT,
            message_types={"ModelNodeIntrospectionEvent"},
        )

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            correlation_id=correlation_id,
            event_type="ModelNodeIntrospectionEvent",
            payload={"node_id": str(uuid4()), "node_type": "EFFECT"},
        )

        before_time = datetime.now(UTC)
        ctx = enforcer.create_context_for_dispatcher(
            dispatcher=dispatcher,
            envelope=envelope,
        )
        after_time = datetime.now(UTC)

        # Simulate dispatch with context
        await dispatcher.handle(envelope, context=ctx)

        # Verify context has time injection (orchestrator should receive now)
        assert dispatcher.captured_context is not None
        assert dispatcher.captured_context.now is not None
        assert before_time <= dispatcher.captured_context.now <= after_time
        assert dispatcher.captured_context.has_time_injection
        assert dispatcher.captured_context.correlation_id == correlation_id
        assert dispatcher.captured_context.node_kind == EnumNodeKind.ORCHESTRATOR


class TestContractHandlerRoutingMatchesRuntime:
    """Tests that contract handler modules are importable at runtime."""

    def test_contract_handler_routing_matches_runtime_wiring(self) -> None:
        """Load handler_routing from contract.yaml, verify each handler's
        module path is importable and the class exists.

        This does NOT instantiate handlers -- only verifies that the
        declared module and class are importable, ensuring no drift
        between contract declarations and actual code.
        """
        assert CONTRACT_PATH.exists(), f"Contract not found at {CONTRACT_PATH}"

        with open(CONTRACT_PATH, encoding="utf-8") as f:
            contract = yaml.safe_load(f)

        handler_routing = contract["handler_routing"]
        handlers = handler_routing["handlers"]

        for entry in handlers:
            event_model = entry.get("event_model", {})
            handler_def = entry.get("handler", {})

            event_model_name = event_model.get("name")
            event_model_module = event_model.get("module")
            handler_name = handler_def.get("name")
            handler_module = handler_def.get("module")

            # Verify event model is importable
            if event_model_module:
                mod = importlib.import_module(event_model_module)
                assert hasattr(mod, event_model_name), (
                    f"Event model class '{event_model_name}' not found "
                    f"in module '{event_model_module}'"
                )

            # Verify handler is importable
            if handler_module:
                mod = importlib.import_module(handler_module)
                assert hasattr(mod, handler_name), (
                    f"Handler class '{handler_name}' not found "
                    f"in module '{handler_module}'"
                )


__all__: list[str] = [
    "TestContractDeclaresIntrospectionRouting",
    "TestDispatchEngineRoutesIntrospection",
    "TestContractHandlerRoutingMatchesRuntime",
]
