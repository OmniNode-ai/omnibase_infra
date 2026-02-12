# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for contract routing introspection (OMN-2081).

Tests that node introspection events dispatch through contract-loaded
handler routing. Verifies:

1. Contract YAML declares the correct handler routing for introspection events
2. Dispatch engine routes introspection events to the correct dispatcher
3. DispatchContextEnforcer creates correct context for orchestrator nodes (time, correlation_id)
4. Contract handler routing matches runtime-importable module paths

Related:
    - OMN-2081: Investor demo - runtime contract routing verification
    - src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml
    - src/omnibase_infra/runtime/service_message_dispatch_engine.py
"""

from __future__ import annotations

import importlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.dispatch_context_enforcer import DispatchContextEnforcer
from tests.helpers.dispatchers import ContextCapturingDispatcher
from tests.helpers.path_utils import find_project_root

logger = logging.getLogger(__name__)

# Path to the registration orchestrator contract
try:
    PROJECT_ROOT = find_project_root(start=Path(__file__).resolve().parent)
    CONTRACT_PATH = (
        PROJECT_ROOT
        / "src"
        / "omnibase_infra"
        / "nodes"
        / "node_registration_orchestrator"
        / "contract.yaml"
    )
except RuntimeError:
    logger.warning(
        "Could not find project root from %s; contract routing tests will be skipped",
        Path(__file__).resolve().parent,
        exc_info=True,
    )
    PROJECT_ROOT = None  # type: ignore[assignment]
    CONTRACT_PATH = None  # type: ignore[assignment]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        CONTRACT_PATH is None,
        reason=(
            "Project root not found: no pyproject.toml found walking up from "
            f"{Path(__file__).resolve().parent}. Check test working directory."
        ),
    ),
]


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

    def test_dispatch_context_enforcer_creates_correct_context(self) -> None:
        """Verify DispatchContextEnforcer creates a properly-shaped context
        for ORCHESTRATOR nodes processing introspection events.

        This is a unit-level test of ``DispatchContextEnforcer.create_context_for_node_kind()``.
        It does NOT test end-to-end dispatch engine context injection; it verifies
        that the enforcer produces a context with time injection enabled,
        correct correlation_id propagation, and correct node_kind for an
        ORCHESTRATOR handling an introspection event.
        """
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope,
        )

        enforcer = DispatchContextEnforcer()
        correlation_id = uuid4()

        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            correlation_id=correlation_id,
            event_type="ModelNodeIntrospectionEvent",
            payload={"node_id": str(uuid4()), "node_type": "EFFECT"},
        )

        before_time = datetime.now(UTC)
        ctx = enforcer.create_context_for_node_kind(
            node_kind=EnumNodeKind.ORCHESTRATOR,
            envelope=envelope,
            dispatcher_id="test-introspection-context",
        )
        after_time = datetime.now(UTC)

        # Verify context has time injection (orchestrator should receive now)
        assert ctx is not None
        assert ctx.now is not None
        assert before_time <= ctx.now <= after_time
        assert ctx.has_time_injection
        assert ctx.correlation_id == correlation_id
        assert ctx.node_kind == EnumNodeKind.ORCHESTRATOR


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
