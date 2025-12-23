# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Reducer Purity Enforcement Gates.

These tests make reducer purity violations IMPOSSIBLE, not just discouraged.
If the same introspection event is replayed tomorrow, next week, or after a crash:
- Reducer emits the same intents
- Effects converge to the same external state
- Observed outcome is identical

If this is not true, the system is broken.

Ticket: OMN-914
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# =============================================================================
# STRUCTURAL GATES: Dependency Graph
# =============================================================================

REDUCER_FILE = Path("src/omnibase_infra/nodes/reducers/registration_reducer.py")

# Forbidden I/O libraries that must NEVER appear in reducer imports
FORBIDDEN_IO_MODULES: set[str] = {
    # Database
    "psycopg",
    "psycopg2",
    "sqlalchemy",
    "asyncpg",
    # HTTP clients
    "requests",
    "httpx",
    "aiohttp",
    "urllib3",
    # Message brokers
    "aiokafka",
    "confluent_kafka",
    "kafka",
    # Service discovery
    "consul",
    "python_consul",
    # Other I/O
    "redis",
    "valkey",
    "socket",
}


class TestStructuralPurityGates:
    """Structural gates that enforce reducer purity via static analysis.

    These tests use AST parsing to verify that reducer modules do not import
    I/O libraries. This is a compile-time guarantee that prevents accidental
    introduction of side effects into pure reducer code.
    """

    def test_reducer_has_no_io_imports(self) -> None:
        """Reducer module must not import I/O libraries.

        Structural gate: If reducer imports any I/O library, this test fails.
        This prevents accidental introduction of I/O dependencies.
        """
        assert REDUCER_FILE.exists(), f"Reducer file not found: {REDUCER_FILE}"

        tree = ast.parse(REDUCER_FILE.read_text())

        # Collect all import statements
        imported_modules: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Extract root module (e.g., "consul" from "consul.client")
                    root_module = alias.name.split(".")[0]
                    imported_modules.add(root_module)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    imported_modules.add(root_module)

        violations = imported_modules & FORBIDDEN_IO_MODULES
        assert not violations, (
            f"Reducer imports forbidden I/O modules: {sorted(violations)}. "
            f"Reducers must be pure - move I/O to Effect layer."
        )

    def test_reducer_state_model_has_no_io_imports(self) -> None:
        """Reducer state model must not import I/O libraries.

        The state model is part of the reducer's pure function boundary.
        """
        state_model_file = Path(
            "src/omnibase_infra/nodes/reducers/models/model_registration_state.py"
        )

        if not state_model_file.exists():
            pytest.skip("State model file not found")

        tree = ast.parse(state_model_file.read_text())

        imported_modules: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    imported_modules.add(root_module)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    imported_modules.add(root_module)

        violations = imported_modules & FORBIDDEN_IO_MODULES
        assert not violations, (
            f"State model imports forbidden I/O modules: {sorted(violations)}. "
            f"State models must be pure data classes."
        )


__all__ = [
    "TestStructuralPurityGates",
    "TestDeterminismGates",
    "TestBehavioralPurityGates",
    "TestAdditionalBehavioralGates",
]


# =============================================================================
# DETERMINISM GATES: Same Input -> Same Output
# =============================================================================


class TestDeterminismGates:
    """Determinism gates that verify reducer produces identical output for same input.

    These tests validate the core pure function property: given identical state
    and event, the reducer MUST produce identical output (new state and intents).

    Determinism is essential for:
    - Event replay after crashes (same events replay to same state)
    - Testing reproducibility (tests are not flaky)
    - Debugging (same input always produces same output)
    - System convergence (replayed events converge to same external state)
    """

    def test_reducer_determinism_same_input_same_output(self) -> None:
        """Same input must produce identical output.

        This is the core guarantee of pure functions.
        Given identical state and event, the reducer MUST produce
        identical new state and intents.
        """
        from datetime import UTC, datetime
        from uuid import UUID

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        # Use fixed UUIDs and timestamp for determinism
        fixed_node_id = UUID("12345678-1234-1234-1234-123456789abc")
        fixed_correlation_id = UUID("abcdef12-abcd-abcd-abcd-abcdefabcdef")
        fixed_timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="effect",
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
        )

        # Run reducer multiple times with same input
        result1 = reducer.reduce(state, event)

        # Reset state for second run (since first run marks event as processed)
        state2 = ModelRegistrationState()
        result2 = reducer.reduce(state2, event)

        # Compare outputs (excluding non-deterministic fields like operation_id)
        assert result1.result.status == result2.result.status, (
            "Reducer produced different status for same input"
        )
        assert result1.result.node_id == result2.result.node_id, (
            "Reducer produced different node_id for same input"
        )
        assert len(result1.intents) == len(result2.intents), (
            "Reducer produced different number of intents for same input"
        )

        # Compare intent types and targets
        for intent1, intent2 in zip(result1.intents, result2.intents, strict=True):
            assert intent1.intent_type == intent2.intent_type, (
                f"Intent type mismatch: {intent1.intent_type} != {intent2.intent_type}"
            )
            assert intent1.target == intent2.target, (
                f"Intent target mismatch: {intent1.target} != {intent2.target}"
            )

    def test_reducer_idempotency(self) -> None:
        """Re-processing same event must not change state.

        Idempotency guarantee: If an event is replayed (same event_id),
        the reducer returns the current state unchanged with no new intents.
        """
        from datetime import UTC, datetime
        from uuid import UUID

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        fixed_node_id = UUID("12345678-1234-1234-1234-123456789abc")
        fixed_correlation_id = UUID("abcdef12-abcd-abcd-abcd-abcdefabcdef")
        fixed_timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        reducer = RegistrationReducer()
        initial_state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="effect",
            node_version="1.0.0",
            endpoints={},
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
        )

        # First processing - should transition state
        result1 = reducer.reduce(initial_state, event)
        assert result1.result.status == "pending", (
            "First reduce should transition to pending"
        )
        assert len(result1.intents) == 2, "First reduce should emit 2 intents"

        # Second processing with SAME event on the NEW state
        # This simulates replay after the first processing
        result2 = reducer.reduce(result1.result, event)

        # Idempotency: second run should return same state with no intents
        assert result2.result.status == result1.result.status, (
            "Idempotent replay changed state"
        )
        assert len(result2.intents) == 0, "Idempotent replay should emit no intents"

    def test_reducer_deterministic_event_id_derivation(self) -> None:
        """Event ID derivation must be deterministic.

        When an event uses a specific correlation_id, the reducer uses that ID.
        This test verifies the internal _derive_deterministic_event_id method
        produces consistent results for content-based ID derivation.
        """
        from datetime import UTC, datetime
        from uuid import UUID

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers.registration_reducer import (
            RegistrationReducer,
        )

        fixed_node_id = UUID("12345678-1234-1234-1234-123456789abc")
        fixed_correlation_id = UUID("abcdef12-abcd-abcd-abcd-abcdefabcdef")
        fixed_timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        reducer = RegistrationReducer()

        # Create identical events with same correlation_id
        event1 = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="effect",
            node_version="1.0.0",
            endpoints={},
            timestamp=fixed_timestamp,
            correlation_id=fixed_correlation_id,
        )
        event2 = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="effect",
            node_version="1.0.0",
            endpoints={},
            timestamp=fixed_timestamp,
            correlation_id=fixed_correlation_id,
        )

        # Derive IDs - should be identical for identical events
        # Since correlation_id is now required, we test the private method directly
        id1 = reducer._derive_deterministic_event_id(event1)
        id2 = reducer._derive_deterministic_event_id(event2)

        assert id1 == id2, (
            f"Deterministic ID derivation produced different IDs: {id1} != {id2}"
        )

        # Different event should produce different ID
        event3 = ModelNodeIntrospectionEvent(
            node_id=UUID("99999999-9999-9999-9999-999999999999"),  # Different node_id
            node_type="effect",
            node_version="1.0.0",
            endpoints={},
            timestamp=fixed_timestamp,
            correlation_id=fixed_correlation_id,  # Same correlation_id
        )
        id3 = reducer._derive_deterministic_event_id(event3)

        assert id1 != id3, "Different events should produce different derived IDs"

    def test_reducer_output_consistency_across_runs(self) -> None:
        """Multiple reduce calls with same inputs produce consistent results.

        This test validates that the reducer's output is not affected by:
        - Internal caching
        - Timing variations
        - System state

        The reducer must be a pure function with no hidden state.
        """
        from datetime import UTC, datetime
        from uuid import UUID

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        fixed_node_id = UUID("12345678-1234-1234-1234-123456789abc")
        fixed_correlation_id = UUID("abcdef12-abcd-abcd-abcd-abcdefabcdef")
        fixed_timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Create SEPARATE reducer instances to prove no instance-level caching
        reducer1 = RegistrationReducer()
        reducer2 = RegistrationReducer()

        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type="effect",
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
        )

        # Run on different reducer instances
        result1 = reducer1.reduce(state, event)
        result2 = reducer2.reduce(state, event)

        # Results must be identical
        assert result1.result.status == result2.result.status
        assert result1.result.node_id == result2.result.node_id
        assert result1.result.consul_confirmed == result2.result.consul_confirmed
        assert result1.result.postgres_confirmed == result2.result.postgres_confirmed
        assert len(result1.intents) == len(result2.intents)

        # Intent payloads must be identical (excluding timestamps if any)
        for i1, i2 in zip(result1.intents, result2.intents, strict=True):
            assert i1.intent_type == i2.intent_type
            assert i1.target == i2.target
            # Compare payload structure (excluding runtime-generated fields)
            assert i1.payload.get("correlation_id") == i2.payload.get("correlation_id")
            assert i1.payload.get("service_id") == i2.payload.get("service_id")


# =============================================================================
# BEHAVIORAL GATES: Runtime Constraints
# =============================================================================


class TestBehavioralPurityGates:
    """Behavioral purity gates for RegistrationReducer.

    These tests validate runtime constraints that ensure the reducer
    remains pure and does not perform I/O operations.
    """

    def test_reducer_has_no_handler_dependencies(self) -> None:
        """Reducer constructor must not accept handlers or I/O clients.

        Behavioral gate: If reducer __init__ accepts any I/O-related parameters,
        this test fails. Reducers are pure - they don't need handlers.
        """
        import inspect

        from omnibase_infra.nodes.reducers import RegistrationReducer

        sig = inspect.signature(RegistrationReducer.__init__)
        param_names = [p.lower() for p in sig.parameters if p != "self"]

        # Forbidden parameter name patterns
        forbidden_patterns = [
            "consul",
            "handler",
            "adapter",
            "db",
            "client",
            "producer",
            "consumer",
            "kafka",
            "redis",
            "postgres",
            "connection",
            "session",
        ]

        for param in param_names:
            for forbidden in forbidden_patterns:
                assert forbidden not in param, (
                    f"Reducer has I/O dependency parameter: '{param}' contains '{forbidden}'. "
                    f"Reducers must be pure - no I/O dependencies allowed."
                )

    def test_reducer_reduce_is_synchronous(self) -> None:
        """Reducer reduce() method must be synchronous (not async).

        Pure functions are synchronous. Async implies I/O waiting.
        """
        import inspect

        from omnibase_infra.nodes.reducers import RegistrationReducer

        assert not inspect.iscoroutinefunction(RegistrationReducer.reduce), (
            "Reducer.reduce() must be synchronous (not async). "
            "Pure reducers don't perform I/O, so they don't need async."
        )

    def test_reducer_reduce_reset_is_synchronous(self) -> None:
        """Reducer reduce_reset() method must be synchronous (not async).

        All reducer methods must be pure (synchronous).
        """
        import inspect

        from omnibase_infra.nodes.reducers import RegistrationReducer

        if hasattr(RegistrationReducer, "reduce_reset"):
            assert not inspect.iscoroutinefunction(RegistrationReducer.reduce_reset), (
                "Reducer.reduce_reset() must be synchronous (not async). "
                "Pure reducers don't perform I/O, so they don't need async."
            )

    def test_reducer_no_network_access(self) -> None:
        """Reducer must not make network calls during reduce().

        Runtime behavioral gate: Mock socket to detect any network access.
        """
        import socket
        from datetime import UTC, datetime
        from unittest.mock import patch
        from uuid import uuid4

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        # Create test fixtures
        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            timestamp=datetime.now(UTC),
            correlation_id=uuid4(),
        )

        # Patch socket.socket to detect any network access
        with patch.object(socket, "socket") as mock_socket:
            # Run the reducer
            _result = reducer.reduce(state, event)

            # Assert no network calls were made
            mock_socket.assert_not_called()

    def test_reducer_no_file_access(self) -> None:
        """Reducer must not access filesystem during reduce().

        Runtime behavioral gate: Mock open() to detect file access.
        """
        from datetime import UTC, datetime
        from unittest.mock import patch
        from uuid import uuid4

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            endpoints={},
            timestamp=datetime.now(UTC),
            correlation_id=uuid4(),
        )

        # Track if builtin open was called
        open_called = False
        original_open = open

        def tracking_open(*args, **kwargs):
            nonlocal open_called
            open_called = True
            return original_open(*args, **kwargs)

        with patch("builtins.open", tracking_open):
            _result = reducer.reduce(state, event)

        assert not open_called, (
            "Reducer accessed filesystem during reduce(). "
            "Reducers must be pure - no I/O allowed."
        )


# =============================================================================
# ADDITIONAL BEHAVIORAL GATES: Comprehensive Purity Validation
# =============================================================================


class TestAdditionalBehavioralGates:
    """Additional behavioral gates for comprehensive purity validation."""

    def test_reducer_no_subprocess_calls(self) -> None:
        """Reducer must not spawn subprocesses during reduce().

        Runtime behavioral gate: Mock subprocess module to detect any spawning.
        """
        import subprocess
        from datetime import UTC, datetime
        from unittest.mock import patch
        from uuid import uuid4

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="compute",
            node_version="1.0.0",
            endpoints={},
            timestamp=datetime.now(UTC),
            correlation_id=uuid4(),
        )

        with (
            patch.object(subprocess, "run") as mock_run,
            patch.object(subprocess, "Popen") as mock_popen,
            patch.object(subprocess, "call") as mock_call,
        ):
            _result = reducer.reduce(state, event)

            mock_run.assert_not_called()
            mock_popen.assert_not_called()
            mock_call.assert_not_called()

    def test_reducer_no_http_requests(self) -> None:
        """Reducer must not make HTTP requests during reduce().

        Runtime behavioral gate: Mock urllib and requests to detect HTTP calls.
        """
        import urllib.request
        from datetime import UTC, datetime
        from unittest.mock import patch
        from uuid import uuid4

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="reducer",
            node_version="1.0.0",
            endpoints={},
            timestamp=datetime.now(UTC),
            correlation_id=uuid4(),
        )

        with patch.object(urllib.request, "urlopen") as mock_urlopen:
            _result = reducer.reduce(state, event)
            mock_urlopen.assert_not_called()

    def test_reducer_init_has_no_required_params(self) -> None:
        """Reducer __init__ should have no required parameters (aside from self).

        This ensures reducers can be instantiated without any external dependencies.
        Variadic parameters (*args, **kwargs) are allowed as they are not required.
        """
        import inspect

        from omnibase_infra.nodes.reducers import RegistrationReducer

        sig = inspect.signature(RegistrationReducer.__init__)
        # Find required positional/keyword-only parameters (not variadic)
        required_params = [
            name
            for name, param in sig.parameters.items()
            if name != "self"
            and param.default is inspect.Parameter.empty
            and param.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,  # *args
                inspect.Parameter.VAR_KEYWORD,  # **kwargs
            )
        ]

        assert len(required_params) == 0, (
            f"Reducer __init__ has required parameters: {required_params}. "
            f"Pure reducers should be instantiable without dependencies."
        )

    def test_reducer_all_public_methods_are_synchronous(self) -> None:
        """All public methods on the reducer must be synchronous.

        Comprehensive check that no public method is async.
        """
        import inspect

        from omnibase_infra.nodes.reducers import RegistrationReducer

        reducer = RegistrationReducer()

        public_methods = [
            name
            for name in dir(reducer)
            if not name.startswith("_") and callable(getattr(reducer, name))
        ]

        async_methods = [
            name
            for name in public_methods
            if inspect.iscoroutinefunction(getattr(reducer, name))
        ]

        assert len(async_methods) == 0, (
            f"Reducer has async public methods: {async_methods}. "
            f"All reducer methods must be synchronous (pure, no I/O)."
        )

    def test_reducer_class_has_no_class_variables_storing_state(self) -> None:
        """Reducer class should not have class-level mutable state.

        Class variables that store mutable state would violate purity.
        """
        from omnibase_infra.nodes.reducers import RegistrationReducer

        # Get class variables (not instance variables, not methods)
        class_vars = {
            name: value
            for name, value in vars(RegistrationReducer).items()
            if not name.startswith("_")
            and not callable(value)
            and not isinstance(value, (property, classmethod, staticmethod))
        }

        # Check for mutable types
        mutable_types = (list, dict, set)
        mutable_class_vars = [
            name
            for name, value in class_vars.items()
            if isinstance(value, mutable_types)
        ]

        assert len(mutable_class_vars) == 0, (
            f"Reducer has mutable class variables: {mutable_class_vars}. "
            f"This violates the pure function contract."
        )
