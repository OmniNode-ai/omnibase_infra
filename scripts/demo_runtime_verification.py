#!/usr/bin/env -S poetry run python
"""Demo: ONEX Runtime Contract Routing Verification (OMN-2081).

Demonstrates the full runtime lifecycle for investor-facing verification:
1. Runtime starts to ready state with timing measurement
2. Introspection event dispatched through contract routing
3. Handler routing resolved from contract YAML
4. Contract handler routing entries verified for structural correctness

Usage:
    ./scripts/demo_runtime_verification.py

Exit codes:
    0 - All verifications passed
    1 - One or more verifications failed
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import yaml

# =============================================================================
# Constants
# =============================================================================


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml).

    Note: The canonical shared implementation lives in
    ``tests.helpers.path_utils.find_project_root``.  This script is a
    standalone entry point that cannot import from the tests package, so the
    logic is duplicated here.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    msg = "Could not find project root (no pyproject.toml found)"
    raise RuntimeError(msg)


PROJECT_ROOT = _find_project_root()
CONTRACT_PATH = (
    PROJECT_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_registration_orchestrator"
    / "contract.yaml"
)

READY_STATE_SLA_SECONDS = 10.0


# =============================================================================
# Handler seeding helper (mirrors tests.helpers.runtime_helpers.seed_mock_handlers)
# =============================================================================


def _seed_mock_handlers(process: object) -> None:
    """Seed mock handlers on a RuntimeHostProcess to bypass fail-fast validation.

    This is the demo-script equivalent of ``seed_mock_handlers`` from
    ``tests.helpers.runtime_helpers``.  This script is a standalone entry
    point that cannot import from the tests package, so the logic is
    intentionally duplicated here.

    Differences from the canonical ``tests.helpers.runtime_helpers.seed_mock_handlers``:

    * Handler name is ``"demo-handler"`` (vs ``"mock"`` in tests) to make
      demo output clearly distinguishable from test output.
    * ``execute`` returns ``{"success": True}`` (vs ``{"success": True,
      "result": "mock"}`` in tests) because the demo never inspects the
      result payload.
    * Does not accept ``handlers`` or ``initialized`` keyword arguments
      because the demo always uses a single default handler.

    The RuntimeHostProcess.start() method validates that handlers are
    registered.  This helper sets up a minimal mock handler to satisfy
    that check, allowing the demo to focus on other runtime functionality.

    Args:
        process: The RuntimeHostProcess instance to seed handlers on.
            Typed as ``object`` to avoid import-order issues; must have
            a ``_handlers`` attribute.
    """
    from unittest.mock import AsyncMock, MagicMock

    from omnibase_infra.protocols.protocol_container_aware import ProtocolContainerAware

    # spec=ProtocolContainerAware constrains the mock to the handler protocol,
    # preventing accidental reliance on auto-created attributes.  Async methods
    # are explicitly overridden because spec alone produces synchronous stubs.
    mock_handler = MagicMock(spec=ProtocolContainerAware)
    mock_handler.execute = AsyncMock(return_value={"success": True})
    mock_handler.initialize = AsyncMock()
    mock_handler.shutdown = AsyncMock()
    mock_handler.health_check = AsyncMock(return_value={"healthy": True})
    mock_handler.initialized = True
    # Private attribute access: RuntimeHostProcess does not expose a public API
    # for handler seeding.  Setting _handlers directly is the only way to inject
    # handlers without the full registry wiring, which is not available in this
    # standalone demo script.
    process._handlers = {"demo-handler": mock_handler}  # type: ignore[attr-defined]


# =============================================================================
# Result tracking
# =============================================================================


class VerificationResult:
    """Track pass/fail status and timing for verification steps.

    Accumulates test results with optional timing data for summary reporting.
    """

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def record(
        self, name: str, passed: bool, detail: str = "", elapsed_ms: float = 0.0
    ) -> None:
        """Record verification step result with optional timing and detail.

        Args:
            name: Human-readable step name for reporting.
            passed: Whether the step passed (True) or failed (False).
            detail: Optional context or diagnostic information.
            elapsed_ms: Optional execution time in milliseconds.
        """
        self.steps.append(
            {
                "name": name,
                "passed": passed,
                "detail": detail,
                "elapsed_ms": elapsed_ms,
            }
        )

    @property
    def all_passed(self) -> bool:
        """Return True if all recorded steps passed, False otherwise."""
        return bool(self.steps) and all(s["passed"] for s in self.steps)


# =============================================================================
# Step 1: Runtime startup timing
# =============================================================================


async def verify_runtime_startup(results: VerificationResult) -> None:
    """Verify runtime reaches ready state within SLA timing.

    Starts a runtime with in-memory event bus, measures startup time, and
    verifies health status. Records pass/fail based on SLA compliance.

    Args:
        results: VerificationResult to record step outcome and timing.
    """
    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
    from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess

    print("\n--- Step 1: Runtime Startup Timing ---")

    event_bus = EventBusInmemory()
    config: dict[str, object] = {
        "service_name": "demo-runtime-verification",
        "node_name": "demo-node",
        "env": "demo",
        "version": "v1",
    }
    runtime = RuntimeHostProcess(event_bus=event_bus, config=config)

    async def noop_populate() -> None:
        pass

    with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
        _seed_mock_handlers(runtime)

        t_start = time.monotonic()
        await runtime.start()
        try:
            health = await runtime.health_check()
            t_elapsed = time.monotonic() - t_start
            elapsed_ms = t_elapsed * 1000

            healthy = health.get("healthy", False)
            is_running = health.get("is_running", False)

            print(f"  Startup time:  {elapsed_ms:.1f} ms")
            print(f"  Healthy:       {healthy}")
            print(f"  Running:       {is_running}")
            print(
                f"  SLA (<{READY_STATE_SLA_SECONDS}s): {'PASS' if t_elapsed < READY_STATE_SLA_SECONDS else 'FAIL'}"
            )

            results.record(
                name="Runtime reaches ready state",
                passed=bool(
                    healthy and is_running and t_elapsed < READY_STATE_SLA_SECONDS
                ),
                detail=f"{elapsed_ms:.1f} ms startup, healthy={healthy}",
                elapsed_ms=elapsed_ms,
            )
        finally:
            await runtime.stop()


# =============================================================================
# Step 2: Contract routing verification
# =============================================================================


def verify_contract_routing(results: VerificationResult) -> None:
    """Verify contract.yaml handler routing declarations are importable.

    Loads the ONEX contract YAML, parses handler routing entries, and verifies
    each handler module/class can be imported successfully.

    Args:
        results: VerificationResult to record importability check outcome.
    """
    print("\n--- Step 2: Contract Handler Routing ---")

    if not CONTRACT_PATH.exists():
        print(f"  ERROR: Contract not found at {CONTRACT_PATH}")
        results.record(
            name="Contract file exists",
            passed=False,
            detail=f"Not found: {CONTRACT_PATH}",
        )
        return

    with open(CONTRACT_PATH, encoding="utf-8") as f:
        contract = yaml.safe_load(f)

    handler_routing = contract.get("handler_routing", {})
    handlers = handler_routing.get("handlers", [])
    routing_strategy = handler_routing.get("routing_strategy", "unknown")

    print(f"  Contract:         {CONTRACT_PATH.name}")
    print(f"  Routing strategy: {routing_strategy}")
    print(f"  Handler count:    {len(handlers)}")
    print()

    # Verify each handler entry
    all_importable = True
    for entry in handlers:
        event_model = entry.get("event_model", {})
        handler_def = entry.get("handler", {})
        event_name = event_model.get("name", "?")
        handler_name = handler_def.get("name", "?")
        handler_module = handler_def.get("module", "?")

        importable = False
        try:
            mod = importlib.import_module(handler_module)
            importable = hasattr(mod, handler_name)
        except Exception as e:
            print(
                f"    WARNING: Could not import {handler_module}: {type(e).__name__}: {e}"
            )
            importable = False

        status = "OK" if importable else "FAIL"
        print(f"  [{status}] {event_name} -> {handler_name}")
        print(f"        module: {handler_module}")

        if not importable:
            all_importable = False

    results.record(
        name="Contract handler routing importable",
        passed=all_importable,
        detail=f"{len(handlers)} handlers, strategy={routing_strategy}",
    )


# =============================================================================
# Step 3: Dispatch engine routing
# =============================================================================


async def verify_dispatch_routing(results: VerificationResult) -> None:
    """Verify dispatch engine routes introspection events to registered dispatchers.

    Registers a capturing dispatcher with the dispatch engine, sends a test
    introspection event, and verifies the dispatcher is invoked with correct
    context injection (correlation ID, time for orchestrators).

    Args:
        results: VerificationResult to record dispatch routing outcome.
    """
    from omnibase_core.enums.enum_node_kind import EnumNodeKind
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
    from omnibase_infra.enums.enum_message_category import EnumMessageCategory
    from omnibase_infra.models.dispatch.model_dispatch_context import (
        ModelDispatchContext,
    )
    from omnibase_infra.models.dispatch.model_dispatch_result import (
        ModelDispatchResult,
    )
    from omnibase_infra.runtime.service_message_dispatch_engine import (
        MessageDispatchEngine,
    )

    print("\n--- Step 3: Dispatch Engine Routing ---")

    engine = MessageDispatchEngine()

    # Track whether the dispatcher was invoked
    invoked = False
    received_context: ModelDispatchContext | None = None

    async def capturing_dispatcher(
        envelope: object,
        context: ModelDispatchContext,
    ) -> ModelDispatchResult:
        nonlocal invoked, received_context
        invoked = True
        received_context = context
        return ModelDispatchResult(
            dispatch_id=uuid4(),
            correlation_id=uuid4(),
            status=EnumDispatchStatus.SUCCESS,
            topic="onex.evt.platform.node-introspection.v1",
            dispatcher_id="demo-introspection-dispatcher",
            started_at=datetime.now(UTC),
        )

    engine.register_dispatcher(
        dispatcher_id="demo-introspection-dispatcher",
        dispatcher=capturing_dispatcher,
        category=EnumMessageCategory.EVENT,
        message_types={"ModelNodeIntrospectionEvent"},
        node_kind=EnumNodeKind.ORCHESTRATOR,
    )

    from omnibase_infra.models.dispatch.model_dispatch_route import (
        ModelDispatchRoute,
    )

    route = ModelDispatchRoute(
        route_id="demo-introspection-route",
        topic_pattern="onex.evt.platform.node-introspection.v1",
        message_category=EnumMessageCategory.EVENT,
        dispatcher_id="demo-introspection-dispatcher",
    )
    engine.register_route(route)

    engine.freeze()

    correlation_id = uuid4()
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        correlation_id=correlation_id,
        event_type="ModelNodeIntrospectionEvent",
        payload={"node_id": str(uuid4()), "node_type": "EFFECT"},
    )

    result = await engine.dispatch(
        topic="onex.evt.platform.node-introspection.v1",
        envelope=envelope,
    )

    dispatched_ok = result.status == EnumDispatchStatus.SUCCESS and invoked
    has_context = received_context is not None
    has_time = (
        has_context
        and received_context is not None
        and received_context.now is not None
    )

    print(f"  Dispatched:       {dispatched_ok}")
    print(f"  Context injected: {has_context}")
    print(f"  Time injected:    {has_time} (orchestrator should receive now)")
    if has_context and received_context is not None:
        print(f"  Correlation ID:   {received_context.correlation_id}")

    results.record(
        name="Dispatch engine routes introspection event",
        passed=dispatched_ok and has_time,
        detail=f"invoked={invoked}, context={has_context}, time={has_time}",
    )


# =============================================================================
# Step 4: Contract handler routing structure verification
# =============================================================================


def verify_contract_handler_structure(results: VerificationResult) -> None:
    """Verify contract handler routing entries have valid structure and async methods.

    Validates routing strategy declaration, checks each handler entry has
    required fields (event model, handler class), and verifies handlers have
    async handle() methods.

    Args:
        results: VerificationResult to record structure validation outcome.
    """
    print("\n--- Step 4: Contract Handler Routing Structure ---")

    if not CONTRACT_PATH.exists():
        print(f"  ERROR: Contract not found at {CONTRACT_PATH}")
        results.record(
            name="Contract handler routing structure valid",
            passed=False,
            detail=f"Not found: {CONTRACT_PATH}",
        )
        return

    with open(CONTRACT_PATH, encoding="utf-8") as f:
        contract = yaml.safe_load(f)

    handler_routing = contract.get("handler_routing", {})
    handlers = handler_routing.get("handlers", [])
    routing_strategy = handler_routing.get("routing_strategy", "")

    # Verify routing strategy is declared
    has_strategy = routing_strategy == "payload_type_match"

    # Verify each handler entry has required structure and importable modules
    all_valid = True
    handler_count = 0
    for entry in handlers:
        event_model = entry.get("event_model", {})
        handler_def = entry.get("handler", {})

        event_name = event_model.get("name", "")
        event_module = event_model.get("module", "")
        handler_name = handler_def.get("name", "")
        handler_module = handler_def.get("module", "")

        # Verify required fields exist
        has_fields = bool(
            event_name and event_module and handler_name and handler_module
        )

        # Verify handler class is importable and has a handle method
        has_handle_method = False
        if has_fields:
            try:
                mod = importlib.import_module(handler_module)
                handler_cls = getattr(mod, handler_name, None)
                if handler_cls is not None:
                    # Check the class has a handle method
                    has_handle_method = hasattr(handler_cls, "handle") and callable(
                        handler_cls.handle
                    )
                    # Verify handle is async
                    if has_handle_method:
                        handle_method = handler_cls.handle
                        has_handle_method = inspect.iscoroutinefunction(handle_method)
            except (ImportError, ModuleNotFoundError):
                has_handle_method = False

        entry_valid = has_fields and has_handle_method
        status = "OK" if entry_valid else "FAIL"
        print(
            f"  [{status}] {event_name} -> {handler_name} (async handle: {has_handle_method})"
        )

        if not entry_valid:
            all_valid = False
        handler_count += 1

    overall_passed = has_strategy and all_valid and handler_count > 0

    print()
    print(
        f"  Routing strategy:   {routing_strategy} ({'OK' if has_strategy else 'FAIL'})"
    )
    print(f"  Handlers verified:  {handler_count}")
    print(f"  All handlers valid: {all_valid}")

    results.record(
        name="Contract handler routing structure valid",
        passed=overall_passed,
        detail=f"{handler_count} handlers, strategy={routing_strategy}, all_valid={all_valid}",
    )


# =============================================================================
# Summary display
# =============================================================================


def display_summary(results: VerificationResult) -> None:
    """Display formatted summary table of all verification step results.

    Args:
        results: VerificationResult containing all recorded steps.
    """
    print("\n" + "=" * 70)
    print(" ONEX Runtime Contract Routing Verification Summary")
    print("=" * 70)
    print()
    print(f"  {'Step':<50} {'Status':>8}")
    print("  " + "-" * 60)

    for step in results.steps:
        status = "PASS" if step["passed"] else "FAIL"
        name = step["name"]
        detail = step["detail"]
        elapsed = step.get("elapsed_ms", 0)

        timing = f" ({elapsed:.0f}ms)" if elapsed > 0 else ""
        print(f"  {name:<50} [{status}]{timing}")
        if detail:
            print(f"    {detail}")

    print()
    total = len(results.steps)
    passed = sum(1 for s in results.steps if s["passed"])
    overall = "ALL PASSED" if results.all_passed else "SOME FAILED"
    print(f"  Result: {passed}/{total} checks passed - {overall}")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


async def run_verification() -> bool:
    """Execute all verification steps and return overall pass/fail status.

    Returns:
        True if all verification steps passed, False if any failed.
    """
    results = VerificationResult()

    print("=" * 70)
    print(" OMN-2081: ONEX Runtime Contract Routing Verification")
    print(f" Time: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    # Step 1: Runtime startup timing
    await verify_runtime_startup(results)

    # Step 2: Contract routing verification (sync)
    verify_contract_routing(results)

    # Step 3: Dispatch engine routing
    await verify_dispatch_routing(results)

    # Step 4: Contract handler routing structure
    verify_contract_handler_structure(results)

    # Summary
    display_summary(results)

    return results.all_passed


def main() -> None:
    """Entry point for runtime verification demo script."""
    all_passed = asyncio.run(run_verification())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
