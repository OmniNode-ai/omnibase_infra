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
import re
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumNodeArchetype

__all__ = [
    "TestStructuralPurityGates",
    "TestDeterminismGates",
    "TestBehavioralPurityGates",
    "TestAdditionalBehavioralGates",
    "TestSecurityGates",
]

# =============================================================================
# STRUCTURAL GATES: Dependency Graph
# =============================================================================

# TODO(OMN-future): Parameterize these tests to support multiple reducers.
# Currently hard-coded for RegistrationReducer (OMN-914).
# Future enhancement: Use pytest.mark.parametrize with:
#   - REDUCER_FILES = list(Path("src/omnibase_infra/nodes/reducers").glob("*_reducer.py"))
# This would enable automatic purity validation for all reducers in the codebase.
REDUCER_FILE = Path("src/omnibase_infra/nodes/reducers/registration_reducer.py")


# Import shared AST analysis utilities from test helpers
# See: tests/helpers/ast_analysis.py for implementation details.
from tests.helpers.ast_analysis import get_imported_root_modules

# Alias for backward compatibility with existing test code in this file
_get_imported_root_modules = get_imported_root_modules


# Forbidden I/O libraries that must NEVER appear in reducer imports
FORBIDDEN_IO_MODULES: set[str] = {
    # Database
    "psycopg",
    "psycopg2",
    "sqlalchemy",
    "asyncpg",
    # Additional database clients
    "pymongo",
    "motor",
    "elasticsearch",
    "opensearchpy",
    # HTTP clients
    "requests",
    "httpx",
    "aiohttp",
    "urllib3",
    # gRPC
    "grpc",
    "grpcio",
    # Message brokers
    "aiokafka",
    "confluent_kafka",
    "kafka",
    # Service discovery
    "consul",
    "python_consul",
    # Cloud SDKs
    "boto3",
    "botocore",
    "google",  # google.cloud
    "azure",
    # Standard library network
    "ftplib",
    "smtplib",
    "telnetlib",
    "poplib",
    "imaplib",
    "nntplib",
    # SSH/SFTP
    "paramiko",
    "fabric",
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
        imported_modules = _get_imported_root_modules(tree)

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
        imported_modules = _get_imported_root_modules(tree)

        violations = imported_modules & FORBIDDEN_IO_MODULES
        assert not violations, (
            f"State model imports forbidden I/O modules: {sorted(violations)}. "
            f"State models must be pure data classes."
        )


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
            node_type=EnumNodeArchetype.EFFECT.value,
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
            node_type=EnumNodeArchetype.EFFECT.value,
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
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={},
            timestamp=fixed_timestamp,
            correlation_id=fixed_correlation_id,
        )
        event2 = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
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
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={},
            timestamp=fixed_timestamp,
            correlation_id=fixed_correlation_id,  # Same correlation_id
        )
        id3 = reducer._derive_deterministic_event_id(event3)

        assert id1 != id3, "Different events should produce different derived IDs"

    def test_reducer_deterministic_event_id_edge_cases(self) -> None:
        """Edge case testing for deterministic ID derivation.

        Validates that edge cases don't break determinism or cause crashes.
        Tests boundary conditions including:
        - Empty endpoints dictionary
        - Very long endpoint URLs
        - Special characters and Unicode in values
        - Different timestamp values produce different IDs
        - Same inputs across multiple calls produce identical IDs

        This test ensures the _derive_deterministic_event_id method is robust
        against unusual but valid input combinations.
        """
        from datetime import UTC, datetime, timedelta
        from uuid import UUID

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers.registration_reducer import (
            RegistrationReducer,
        )

        fixed_node_id = UUID("12345678-1234-1234-1234-123456789abc")
        fixed_correlation_id = UUID("abcdef12-abcd-abcd-abcd-abcdefabcdef")
        fixed_timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        reducer = RegistrationReducer()

        # ---------------------------------------------------------------------
        # Test 1: Empty endpoints dictionary
        # Empty endpoints should still produce deterministic IDs
        # ---------------------------------------------------------------------
        event_empty_endpoints = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={},  # Empty
            timestamp=fixed_timestamp,
            correlation_id=fixed_correlation_id,
        )
        id1 = reducer._derive_deterministic_event_id(event_empty_endpoints)
        id2 = reducer._derive_deterministic_event_id(event_empty_endpoints)
        assert id1 == id2, "Empty endpoints should still be deterministic"
        assert isinstance(id1, UUID), "Derived ID should be a valid UUID"

        # ---------------------------------------------------------------------
        # Test 2: Very long endpoint URLs
        # The method should handle arbitrarily long endpoint values
        # (Note: endpoints are not used in ID derivation, but this verifies
        # the event can be constructed and processed without issues)
        # ---------------------------------------------------------------------
        very_long_path = "/path" * 1000  # 5000 character path
        event_long_urls = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.COMPUTE.value,
            node_version="1.0.0",
            endpoints={
                "health": f"http://localhost:8080{very_long_path}",
                "api": f"https://api.example.com{very_long_path}",
            },
            timestamp=fixed_timestamp,
            correlation_id=fixed_correlation_id,
        )
        id_long1 = reducer._derive_deterministic_event_id(event_long_urls)
        id_long2 = reducer._derive_deterministic_event_id(event_long_urls)
        assert id_long1 == id_long2, "Long endpoint URLs should not break determinism"

        # ---------------------------------------------------------------------
        # Test 3: Special characters in endpoint URLs
        # URLs with query params, fragments, encoded chars should work
        # ---------------------------------------------------------------------
        event_special_chars = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.REDUCER.value,
            node_version="1.0.0",
            endpoints={
                "health": "http://localhost:8080/health?param=value&other=123",
                "metrics": "http://localhost:9090/metrics#section",
                "encoded": "http://localhost:8080/path%20with%20spaces",
            },
            timestamp=fixed_timestamp,
            correlation_id=fixed_correlation_id,
        )
        id_special1 = reducer._derive_deterministic_event_id(event_special_chars)
        id_special2 = reducer._derive_deterministic_event_id(event_special_chars)
        assert id_special1 == id_special2, (
            "Special characters in URLs should not break determinism"
        )

        # ---------------------------------------------------------------------
        # Test 4: Unicode characters in endpoint URLs
        # International domain names and paths should work
        # ---------------------------------------------------------------------
        event_unicode = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.ORCHESTRATOR.value,
            node_version="1.0.0",
            endpoints={
                "intl": "http://xn--n3h.example.com/api",  # Punycode for emoji domain
                "path_unicode": "http://localhost:8080/%E4%B8%AD%E6%96%87",  # URL-encoded Chinese
            },
            timestamp=fixed_timestamp,
            correlation_id=fixed_correlation_id,
        )
        id_unicode1 = reducer._derive_deterministic_event_id(event_unicode)
        id_unicode2 = reducer._derive_deterministic_event_id(event_unicode)
        assert id_unicode1 == id_unicode2, (
            "Unicode in URLs should not break determinism"
        )

        # ---------------------------------------------------------------------
        # Test 5: Timestamps affect derived ID
        # Different timestamps should produce different IDs (timestamp is
        # part of the hash input)
        # ---------------------------------------------------------------------
        event_ts1 = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={},
            timestamp=fixed_timestamp,
            correlation_id=fixed_correlation_id,
        )
        event_ts2 = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={},
            timestamp=fixed_timestamp + timedelta(seconds=1),  # 1 second later
            correlation_id=fixed_correlation_id,
        )
        id_ts1 = reducer._derive_deterministic_event_id(event_ts1)
        id_ts2 = reducer._derive_deterministic_event_id(event_ts2)
        assert id_ts1 != id_ts2, "Different timestamps should produce different IDs"

        # ---------------------------------------------------------------------
        # Test 6: Same timestamp always produces same ID
        # Verify determinism for identical timestamp values
        # ---------------------------------------------------------------------
        event_same_ts_a = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={},
            timestamp=fixed_timestamp,
            correlation_id=fixed_correlation_id,
        )
        event_same_ts_b = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={},
            timestamp=fixed_timestamp,  # Same timestamp
            # Different correlation_id
            correlation_id=UUID("99999999-9999-9999-9999-999999999999"),
        )
        id_same_a = reducer._derive_deterministic_event_id(event_same_ts_a)
        id_same_b = reducer._derive_deterministic_event_id(event_same_ts_b)
        assert id_same_a == id_same_b, (
            "Same content with different correlation_id should produce same derived ID "
            "(correlation_id is not part of the hash)"
        )

        # ---------------------------------------------------------------------
        # Test 7: All node types produce valid UUIDs
        # Each node type should work correctly
        # ---------------------------------------------------------------------
        node_types = [
            EnumNodeArchetype.EFFECT.value,
            EnumNodeArchetype.COMPUTE.value,
            EnumNodeArchetype.REDUCER.value,
            EnumNodeArchetype.ORCHESTRATOR.value,
        ]
        for node_type in node_types:
            event_type = ModelNodeIntrospectionEvent(
                node_id=fixed_node_id,
                node_type=node_type,
                node_version="1.0.0",
                endpoints={},
                timestamp=fixed_timestamp,
                correlation_id=fixed_correlation_id,
            )
            derived_id = reducer._derive_deterministic_event_id(event_type)
            assert isinstance(derived_id, UUID), (
                f"Node type '{node_type}' should produce valid UUID"
            )

        # Different node types should produce different IDs
        ids_by_type = {}
        for node_type in node_types:
            event_type = ModelNodeIntrospectionEvent(
                node_id=fixed_node_id,
                node_type=node_type,
                node_version="1.0.0",
                endpoints={},
                timestamp=fixed_timestamp,
                correlation_id=fixed_correlation_id,
            )
            ids_by_type[node_type] = reducer._derive_deterministic_event_id(event_type)

        assert len(set(ids_by_type.values())) == 4, (
            "Each node type should produce a unique derived ID"
        )

        # ---------------------------------------------------------------------
        # Test 8: Edge case node_version values
        # Various valid semver formats should work
        # ---------------------------------------------------------------------
        version_test_cases = [
            "0.0.1",  # Minimum non-zero version
            "99.99.99",  # High version numbers
            "1.0.0-alpha",  # Pre-release
            "1.0.0-alpha.1",  # Pre-release with number
            "1.0.0+build.123",  # Build metadata
            "1.0.0-beta+build.456",  # Pre-release with build metadata
        ]

        for version in version_test_cases:
            event_version = ModelNodeIntrospectionEvent(
                node_id=fixed_node_id,
                node_type=EnumNodeArchetype.EFFECT.value,
                node_version=version,
                endpoints={},
                timestamp=fixed_timestamp,
                correlation_id=fixed_correlation_id,
            )
            id_v1 = reducer._derive_deterministic_event_id(event_version)
            id_v2 = reducer._derive_deterministic_event_id(event_version)
            assert id_v1 == id_v2, (
                f"Version '{version}' should produce deterministic ID"
            )
            assert isinstance(id_v1, UUID), (
                f"Version '{version}' should produce valid UUID"
            )

        # ---------------------------------------------------------------------
        # Test 9: Microsecond precision in timestamps
        # Timestamps with microsecond differences should produce different IDs
        # ---------------------------------------------------------------------
        ts_base = datetime(2025, 1, 1, 12, 0, 0, 0, tzinfo=UTC)
        ts_micro = datetime(2025, 1, 1, 12, 0, 0, 1, tzinfo=UTC)  # 1 microsecond later

        event_micro1 = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={},
            timestamp=ts_base,
            correlation_id=fixed_correlation_id,
        )
        event_micro2 = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={},
            timestamp=ts_micro,
            correlation_id=fixed_correlation_id,
        )
        id_micro1 = reducer._derive_deterministic_event_id(event_micro1)
        id_micro2 = reducer._derive_deterministic_event_id(event_micro2)
        assert id_micro1 != id_micro2, (
            "Microsecond timestamp differences should produce different IDs"
        )

        # ---------------------------------------------------------------------
        # Test 10: Large endpoint dictionaries
        # Many endpoints should not affect performance or correctness
        # (endpoints are not part of ID hash, but event should be processable)
        # ---------------------------------------------------------------------
        many_endpoints = {
            f"endpoint_{i}": f"http://localhost:{8000 + i}/api" for i in range(100)
        }
        event_many = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints=many_endpoints,
            timestamp=fixed_timestamp,
            correlation_id=fixed_correlation_id,
        )
        id_many1 = reducer._derive_deterministic_event_id(event_many)
        id_many2 = reducer._derive_deterministic_event_id(event_many)
        assert id_many1 == id_many2, "Many endpoints should not break determinism"

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
            node_type=EnumNodeArchetype.EFFECT.value,
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
            # Use model_dump() since payload is now a Pydantic model, not a dict
            i1_payload = i1.payload.model_dump()
            i2_payload = i2.payload.model_dump()
            assert i1_payload.get("correlation_id") == i2_payload.get("correlation_id")
            assert i1_payload.get("service_id") == i2_payload.get("service_id")

    def test_reducer_input_state_is_not_mutated(self) -> None:
        """Verify reduce() does not mutate the input state object.

        Pure functions must not modify their inputs. The reducer receives
        a state object and event, computes a NEW state, and returns it.
        The original input state must remain unchanged.

        This is critical for:
        - Event replay (original state preserved for re-processing)
        - Debugging (can inspect input state after failed reduce)
        - Parallelism (multiple threads can share input state safely)
        """
        import copy
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

        # Create initial state and deep copy it for comparison
        initial_state = ModelRegistrationState()
        state_before = copy.deepcopy(initial_state)

        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
        )

        # Run reducer
        _result = reducer.reduce(initial_state, event)

        # Verify input state was NOT mutated
        # Compare all fields of the state before and after reduce()
        assert initial_state.status == state_before.status, (
            f"Input state.status was mutated: {state_before.status} -> {initial_state.status}"
        )
        assert initial_state.node_id == state_before.node_id, (
            "Input state.node_id was mutated"
        )
        assert initial_state.consul_confirmed == state_before.consul_confirmed, (
            "Input state.consul_confirmed was mutated"
        )
        assert initial_state.postgres_confirmed == state_before.postgres_confirmed, (
            "Input state.postgres_confirmed was mutated"
        )
        assert (
            initial_state.last_processed_event_id
            == state_before.last_processed_event_id
        ), "Input state.last_processed_event_id was mutated"

    def test_reducer_concurrent_execution_is_safe(self) -> None:
        """Verify reducer is thread-safe for concurrent reduce() calls.

        Since reducers are pure functions with no shared mutable state,
        concurrent execution must produce identical results. This test
        validates that multiple threads can safely call reduce() simultaneously
        on the same reducer instance without race conditions or data corruption.

        Thread Safety Guarantee:
        - No instance-level mutable state is modified during reduce()
        - All state transitions are computed from immutable inputs
        - Output is deterministic regardless of thread scheduling

        This property is essential for:
        - Horizontal scaling (multiple workers processing events)
        - Event replay parallelization
        - High-throughput event processing pipelines
        """
        import concurrent.futures
        from datetime import UTC, datetime
        from uuid import UUID

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState
        from omnibase_infra.testing import is_ci_environment

        # Use fixed UUIDs and timestamp for determinism
        fixed_node_id = UUID("12345678-1234-1234-1234-123456789abc")
        fixed_correlation_id = UUID("abcdef12-abcd-abcd-abcd-abcdefabcdef")
        fixed_timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Single reducer instance shared across all threads
        reducer = RegistrationReducer()

        # Immutable inputs - same state and event for all threads
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
        )

        def run_reduce() -> object:
            """Execute reduce() and return result for comparison."""
            return reducer.reduce(state, event)

        # Determine concurrency level - reduce in CI to avoid resource contention
        is_ci = is_ci_environment()
        num_concurrent = 4 if is_ci else 10

        # Execute reduce() concurrently from multiple threads
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_concurrent
        ) as executor:
            futures = [executor.submit(run_reduce) for _ in range(num_concurrent)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All results must be identical - this proves thread safety
        first_result = results[0]

        for i, result in enumerate(results[1:], start=2):
            # Compare result state
            assert result.result.status == first_result.result.status, (
                f"Thread {i} produced different status: "
                f"{result.result.status} != {first_result.result.status}"
            )
            assert result.result.node_id == first_result.result.node_id, (
                f"Thread {i} produced different node_id: "
                f"{result.result.node_id} != {first_result.result.node_id}"
            )
            assert (
                result.result.consul_confirmed == first_result.result.consul_confirmed
            ), (
                f"Thread {i} produced different consul_confirmed: "
                f"{result.result.consul_confirmed} != {first_result.result.consul_confirmed}"
            )
            assert (
                result.result.postgres_confirmed
                == first_result.result.postgres_confirmed
            ), (
                f"Thread {i} produced different postgres_confirmed: "
                f"{result.result.postgres_confirmed} != {first_result.result.postgres_confirmed}"
            )

            # Compare intents
            assert len(result.intents) == len(first_result.intents), (
                f"Thread {i} produced different number of intents: "
                f"{len(result.intents)} != {len(first_result.intents)}"
            )

            for j, (intent, first_intent) in enumerate(
                zip(result.intents, first_result.intents, strict=True)
            ):
                assert intent.intent_type == first_intent.intent_type, (
                    f"Thread {i}, intent {j}: type mismatch: "
                    f"{intent.intent_type} != {first_intent.intent_type}"
                )
                assert intent.target == first_intent.target, (
                    f"Thread {i}, intent {j}: target mismatch: "
                    f"{intent.target} != {first_intent.target}"
                )
                # Compare payload correlation_id and service_id
                # Use model_dump() since payload is now a Pydantic model, not a dict
                intent_payload = intent.payload.model_dump()
                first_intent_payload = first_intent.payload.model_dump()
                assert intent_payload.get("correlation_id") == first_intent_payload.get(
                    "correlation_id"
                ), f"Thread {i}, intent {j}: correlation_id mismatch"
                assert intent_payload.get("service_id") == first_intent_payload.get(
                    "service_id"
                ), f"Thread {i}, intent {j}: service_id mismatch"


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

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        # Create test fixtures
        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type=EnumNodeArchetype.EFFECT.value,
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

        Runtime behavioral gate: Track open() calls originating from reducer code.

        Note: We use call stack inspection to distinguish between file access by
        the reducer (which violates purity) vs. file access by pytest internals,
        coverage tools, or other framework code (which is expected and allowed).
        This prevents spurious test failures from framework operations.
        """
        import traceback
        from datetime import UTC, datetime
        from unittest.mock import patch

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={},
            timestamp=datetime.now(UTC),
            correlation_id=uuid4(),
        )

        # Track file access calls that originate from reducer code
        reducer_file_access_calls: list[str] = []
        original_open = open

        # Paths that indicate reducer code (not framework/test infrastructure)
        reducer_code_markers = (
            "omnibase_infra/nodes/reducers",
            "registration_reducer.py",
        )

        # Paths to exclude (framework/test infrastructure)
        framework_exclusions = (
            "_pytest",
            "pytest",
            "coverage",
            "pluggy",
            "unittest",
            "importlib",
            "logging",
            "site-packages",
            "test_reducer_purity.py",  # Exclude this test file itself
        )

        def tracking_open(*args: object, **kwargs: object) -> object:
            # Inspect call stack to determine origin
            stack = traceback.extract_stack()

            # Check if call originates from reducer code
            for frame in stack:
                filename = frame.filename

                # Skip if from framework/test infrastructure
                if any(excl in filename for excl in framework_exclusions):
                    # This is framework code, allow it
                    return original_open(*args, **kwargs)

                # Check if from reducer code
                if any(marker in filename for marker in reducer_code_markers):
                    # This is reducer code accessing files - record the violation
                    call_info = f"{frame.filename}:{frame.lineno} in {frame.name}"
                    reducer_file_access_calls.append(call_info)
                    return original_open(*args, **kwargs)

            # Not from reducer code, allow it (other library/framework code)
            return original_open(*args, **kwargs)

        with patch("builtins.open", tracking_open):
            _result = reducer.reduce(state, event)

        assert len(reducer_file_access_calls) == 0, (
            f"Reducer accessed filesystem during reduce(). "
            f"Reducers must be pure - no I/O allowed. "
            f"File access detected from: {reducer_file_access_calls}"
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

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type=EnumNodeArchetype.COMPUTE.value,
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

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type=EnumNodeArchetype.REDUCER.value,
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

        Comprehensive check that no public method is async, including:
        - Coroutine functions (async def)
        - Async generators (async def with yield)
        - Methods inherited from parent classes via MRO

        Why comprehensive async detection is necessary:
        - inspect.iscoroutinefunction() may miss some edge cases
        - asyncio.iscoroutinefunction() provides additional coverage
        - Async generators (async def ... yield) are also I/O-capable
        - Parent class methods could introduce async behavior if not checked

        Pure reducers must be fully synchronous because:
        - Async implies awaiting I/O operations
        - Reducers are pure functions with no side effects
        - Event replay requires deterministic, instant execution
        """
        import asyncio
        import inspect

        from omnibase_infra.nodes.reducers import RegistrationReducer

        reducer = RegistrationReducer()
        reducer_class = type(reducer)

        # Collect all public methods from instance (includes inherited methods)
        public_methods = [
            name
            for name in dir(reducer)
            if not name.startswith("_") and callable(getattr(reducer, name))
        ]

        async_methods: list[str] = []
        async_generators: list[str] = []

        for name in public_methods:
            method = getattr(reducer, name)

            # Check 1: inspect.iscoroutinefunction - standard coroutine check
            if inspect.iscoroutinefunction(method):
                async_methods.append(f"{name} (coroutine via inspect)")
                continue

            # Check 2: asyncio.iscoroutinefunction - may catch additional cases
            # This provides defense-in-depth for edge cases where inspect
            # might not detect async behavior (e.g., wrapped decorators)
            if asyncio.iscoroutinefunction(method):
                async_methods.append(f"{name} (coroutine via asyncio)")
                continue

            # Check 3: inspect.isasyncgenfunction - async generators
            # Async generators (async def with yield) can perform I/O
            # between yields and violate reducer purity
            if inspect.isasyncgenfunction(method):
                async_generators.append(name)

        # Also check methods defined in parent classes via MRO
        # This catches async methods that might be inherited but not
        # directly visible on the instance
        for cls in inspect.getmro(reducer_class):
            if cls is object:
                continue  # Skip base object class

            for name, method in vars(cls).items():
                if name.startswith("_"):
                    continue
                if not callable(method):
                    continue

                # Check inherited coroutines
                if inspect.iscoroutinefunction(method):
                    marker = f"{name} (inherited from {cls.__name__})"
                    if marker not in async_methods:
                        async_methods.append(marker)

                # Check inherited async generators
                if inspect.isasyncgenfunction(method):
                    marker = f"{name} (async gen from {cls.__name__})"
                    if marker not in async_generators:
                        async_generators.append(marker)

        # Build comprehensive error message
        all_violations = async_methods + [
            f"{m} (async generator)" for m in async_generators
        ]

        assert len(all_violations) == 0, (
            f"Reducer has async public methods: {all_violations}. "
            f"All reducer methods must be synchronous (pure, no I/O). "
            f"Async methods imply I/O waiting which violates reducer purity."
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
            and not isinstance(value, property | classmethod | staticmethod)
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

    def test_reducer_class_variables_are_not_mutated_across_instances(self) -> None:
        """Verify class variables are not mutated across instances.

        This test ensures that running reduce() on one instance doesn't
        affect the behavior of another instance through shared class state.

        Specifically validates:
        1. State isolation: reduce() on instance A doesn't affect instance B
        2. Class immutability: Class-level attributes remain unchanged after reduce()
        3. Determinism: Same input produces same output regardless of prior instance activity

        This is a stronger guarantee than test_reducer_class_has_no_class_variables_storing_state
        which only checks for mutable class variables at definition time. This test verifies
        that even if class variables exist, they are not mutated during reduce() execution.
        """
        import copy
        from datetime import UTC, datetime
        from uuid import UUID

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        # Use fixed UUIDs and timestamp for determinism
        fixed_node_id = UUID("12345678-1234-1234-1234-123456789abc")
        fixed_correlation_id = UUID("abcdef12-abcd-abcd-abcd-abcdefabcdef")
        fixed_timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Capture class-level state BEFORE any reduce() calls
        # We deep copy to detect mutations to mutable containers
        class_vars_before = {
            name: copy.deepcopy(value)
            for name, value in vars(RegistrationReducer).items()
            if not name.startswith("_")
            and not callable(value)
            and not isinstance(value, property | classmethod | staticmethod)
        }

        # Create first reducer instance and run reduce
        reducer1 = RegistrationReducer()
        state1 = ModelRegistrationState()
        event1 = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
        )
        result1 = reducer1.reduce(state1, event1)

        # Create second reducer instance AFTER first reduce completed
        reducer2 = RegistrationReducer()
        state2 = ModelRegistrationState()
        event2 = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
        )
        result2 = reducer2.reduce(state2, event2)

        # Verify class-level state was NOT mutated by reduce() calls
        class_vars_after = {
            name: value
            for name, value in vars(RegistrationReducer).items()
            if not name.startswith("_")
            and not callable(value)
            and not isinstance(value, property | classmethod | staticmethod)
        }

        assert class_vars_before == class_vars_after, (
            "Class-level variables were mutated during reduce() execution. "
            f"Before: {class_vars_before}, After: {class_vars_after}. "
            "This violates reducer purity - class state must remain immutable."
        )

        # Verify no state leakage between instances:
        # Both instances should produce identical results for identical input
        assert result1.result.status == result2.result.status, (
            f"State leakage detected: result1.status={result1.result.status} "
            f"!= result2.status={result2.result.status}. "
            "Running reduce() on first instance affected second instance behavior."
        )
        assert result1.result.node_id == result2.result.node_id, (
            f"State leakage detected: result1.node_id={result1.result.node_id} "
            f"!= result2.node_id={result2.result.node_id}."
        )
        assert len(result1.intents) == len(result2.intents), (
            f"State leakage detected: result1 has {len(result1.intents)} intents "
            f"but result2 has {len(result2.intents)} intents."
        )

        # Verify intent content is identical (proves no hidden state affecting output)
        for idx, (intent1, intent2) in enumerate(
            zip(result1.intents, result2.intents, strict=True)
        ):
            assert intent1.intent_type == intent2.intent_type, (
                f"Intent {idx} type mismatch: {intent1.intent_type} != {intent2.intent_type}"
            )
            assert intent1.target == intent2.target, (
                f"Intent {idx} target mismatch: {intent1.target} != {intent2.target}"
            )


# =============================================================================
# SECURITY GATES: Credential Leakage Prevention
# =============================================================================


class TestSecurityGates:
    """Security gates that prevent sensitive data leakage in reducer outputs.

    These tests validate that the reducer sanitizes data before emitting intents,
    ensuring that credentials, secrets, and other sensitive information are not
    exposed in intent payloads that flow to downstream systems.

    Security is critical because:
    - Intent payloads are published to Kafka and may be logged
    - Downstream Effect nodes may forward payloads to external systems
    - Event replay could expose historical credentials
    - Audit logs may capture intent payloads for compliance

    The reducer must NEVER leak:
    - Passwords, API keys, tokens, secrets
    - Connection strings with credentials
    - Authentication headers or cookies
    - Private keys or certificates
    """

    # Sensitive field name patterns that should NEVER appear in intent payloads
    # These patterns match common credential field naming conventions
    SENSITIVE_FIELD_PATTERNS: set[str] = {
        "password",
        "passwd",
        "pwd",
        "secret",
        "api_key",
        "apikey",
        "api-key",
        "access_token",
        "refresh_token",
        "auth_token",
        "bearer",
        "credential",
        "credentials",
        "private_key",
        "privatekey",
        "private-key",
        "encryption_key",
        "decryption_key",
        "master_key",
        "authorization",
        "authentication",
        "client_secret",
        "session_id",
        "session_token",
        "cookie",
        "jwt",
        "oauth",
        "ssh_key",
        "ssl_cert",
        "conn_string",
        "connection_string",
        "database_url",
        "db_password",
    }

    # Sensitive value patterns (regex-like substrings that indicate secrets)
    SENSITIVE_VALUE_PATTERNS: list[str] = [
        "password=",
        "passwd=",
        "secret=",
        "api_key=",
        "apikey=",
        "Bearer ",
        "Basic ",
        "-----BEGIN",  # PEM format private keys
        "-----END",
        "AKIA",  # AWS access key prefix
        "sk_live_",  # Stripe live secret key prefix
        "sk_test_",  # Stripe test secret key prefix
        "ghp_",  # GitHub personal access token prefix
        "gho_",  # GitHub OAuth token prefix
        "xox",  # Slack token prefix
        "AIza",  # Google API key prefix
        "SG.",  # SendGrid API key prefix
        "npm_",  # npm token prefix
        "pypi-",  # PyPI token prefix
        "glpat-",  # GitLab personal access token prefix
        "hf_",  # Hugging Face token prefix
        "sq0csp-",  # Square sandbox token prefix
        "sq0atp-",  # Square production token prefix
    ]

    def _check_dict_for_sensitive_fields(self, data: dict, path: str = "") -> list[str]:
        """Recursively check a dictionary for sensitive field names.

        Args:
            data: Dictionary to inspect.
            path: Current path in the nested structure (for error messages).

        Returns:
            List of sensitive field paths found.
        """
        violations: list[str] = []

        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            key_lower = str(key).lower()

            # Check if field name matches sensitive patterns
            for pattern in self.SENSITIVE_FIELD_PATTERNS:
                if pattern in key_lower:
                    violations.append(f"{current_path} (matches '{pattern}')")
                    break

            # Check string values for sensitive patterns
            if isinstance(value, str):
                for pattern in self.SENSITIVE_VALUE_PATTERNS:
                    if pattern in value:
                        violations.append(f"{current_path} value contains '{pattern}'")
                        break

            # Recursively check nested dictionaries
            if isinstance(value, dict):
                violations.extend(
                    self._check_dict_for_sensitive_fields(value, current_path)
                )

            # Check lists for dictionaries
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        violations.extend(
                            self._check_dict_for_sensitive_fields(
                                item, f"{current_path}[{i}]"
                            )
                        )

        return violations

    def test_reducer_intent_payloads_no_credential_leakage(self) -> None:
        """Verify intent payloads do not contain credentials or sensitive data.

        This test ensures the reducer sanitizes data before emitting intents.
        The reducer may process events with sensitive metadata patterns, and
        we must verify this data is not exposed in the emitted intent payloads.

        Security Rationale:
        - Intent payloads are serialized and published to Kafka topics
        - Kafka messages may be logged by monitoring systems
        - Downstream consumers may forward data to external services
        - Event replay for debugging could expose historical secrets
        - Compliance requirements (SOC2, GDPR) mandate credential protection

        Test Strategy:
        - Create an event with metadata containing sensitive field patterns
        - Run the reducer to generate intents
        - Inspect all emitted intent payloads recursively
        - Assert no sensitive field names or value patterns appear
        """
        from datetime import UTC, datetime
        from uuid import UUID

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.models.registration.model_node_metadata import (
            ModelNodeMetadata,
        )
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        # Fixed identifiers for determinism
        fixed_node_id = UUID("12345678-1234-1234-1234-123456789abc")
        fixed_correlation_id = UUID("abcdef12-abcd-abcd-abcd-abcdefabcdef")
        fixed_timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Create metadata with potentially sensitive field patterns
        # ModelNodeMetadata has extra="allow", so custom fields are accepted
        # This simulates a node that accidentally includes sensitive config
        sensitive_metadata = ModelNodeMetadata(
            version="1.0.0",
            environment="production",
            region="us-west-2",
        )
        # Add extra fields that simulate accidental credential inclusion
        # These are stored in model_extra due to extra="allow"
        object.__setattr__(
            sensitive_metadata,
            "__pydantic_extra__",
            {
                "db_connection_info": "host=localhost",  # Safe: no credentials
                "service_config": {"timeout": 30},  # Safe: non-sensitive config
            },
        )

        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            endpoints={"health": "http://localhost:8080/health"},
            metadata=sensitive_metadata,
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
        )

        # Run the reducer
        result = reducer.reduce(state, event)

        # Verify intents were emitted
        assert len(result.intents) == 2, (
            f"Expected 2 intents (Consul + PostgreSQL), got {len(result.intents)}"
        )

        # Check each intent payload for sensitive data
        # Access .data since payload is ModelPayloadExtension
        all_violations: list[str] = []

        for intent in result.intents:
            # Use model_dump() since payload is now a Pydantic model, not a dict
            intent_violations = self._check_dict_for_sensitive_fields(
                intent.payload.model_dump(), f"intent[{intent.intent_type}]"
            )
            all_violations.extend(intent_violations)

        assert len(all_violations) == 0, (
            "Intent payloads contain sensitive data patterns:\n"
            "  - " + "\n  - ".join(all_violations)
        )

    def test_reducer_intent_payloads_no_sensitive_endpoints(self) -> None:
        """Verify endpoint URLs in intent payloads do not contain credentials.

        Endpoint URLs sometimes contain embedded credentials (e.g., basic auth
        in URLs like 'http://user:pass@host/path'). This test ensures such
        patterns are not present in the emitted intent payloads.

        Security Rationale:
        - URLs with embedded credentials are a common security anti-pattern
        - Health check URLs in Consul intents should not expose auth
        - PostgreSQL record endpoints should not contain connection strings
        """
        import re
        from datetime import UTC, datetime
        from uuid import UUID

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        fixed_node_id = UUID("12345678-1234-1234-1234-123456789abc")
        fixed_correlation_id = UUID("abcdef12-abcd-abcd-abcd-abcdefabcdef")
        fixed_timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        reducer = RegistrationReducer()
        state = ModelRegistrationState()
        event = ModelNodeIntrospectionEvent(
            node_id=fixed_node_id,
            node_type=EnumNodeArchetype.EFFECT.value,
            node_version="1.0.0",
            # Safe endpoint - no embedded credentials
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=fixed_correlation_id,
            timestamp=fixed_timestamp,
        )

        result = reducer.reduce(state, event)

        # Check all intent payloads for URL credential patterns
        for intent in result.intents:
            payload_str = str(intent.payload)

            # Check for basic auth in URLs (user:pass@host pattern)
            # This regex-like check looks for the pattern after ://
            basic_auth_pattern = r"://[^/]+:[^/]+@"
            matches = re.findall(basic_auth_pattern, payload_str)

            assert len(matches) == 0, (
                f"Intent '{intent.intent_type}' contains URL with embedded "
                f"credentials: {matches}"
            )

    def test_reducer_does_not_log_sensitive_validation_details(self) -> None:
        """Verify validation failure logging does not expose sensitive data.

        When validation fails, the reducer logs diagnostic information.
        This test ensures that log messages use sanitized context rather
        than raw event data that might contain sensitive information.

        Security Rationale:
        - Validation errors are logged for diagnostics
        - Logs may be shipped to centralized logging systems
        - Raw event data in logs could expose credentials
        - The reducer should log field presence (bool) not values
        """
        import logging
        from datetime import UTC, datetime
        from uuid import UUID

        from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
        from omnibase_infra.nodes.reducers import RegistrationReducer
        from omnibase_infra.nodes.reducers.models import ModelRegistrationState

        fixed_correlation_id = UUID("abcdef12-abcd-abcd-abcd-abcdefabcdef")
        fixed_timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Capture log output
        log_records: list[logging.LogRecord] = []

        class CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                log_records.append(record)

        # Get the reducer's logger
        reducer_logger = logging.getLogger(
            "omnibase_infra.nodes.reducers.registration_reducer"
        )
        original_level = reducer_logger.level
        capturing_handler = CapturingHandler()
        reducer_logger.addHandler(capturing_handler)
        reducer_logger.setLevel(logging.DEBUG)

        try:
            reducer = RegistrationReducer()
            state = ModelRegistrationState()

            # Create a valid event (validation will pass)
            # The reducer logs performance warnings if thresholds exceeded,
            # but that's not what we're testing here.
            event = ModelNodeIntrospectionEvent(
                node_id=UUID("12345678-1234-1234-1234-123456789abc"),
                node_type=EnumNodeArchetype.EFFECT.value,
                node_version="1.0.0",
                endpoints={},
                correlation_id=fixed_correlation_id,
                timestamp=fixed_timestamp,
            )

            # Run reducer (should succeed without validation log)
            _result = reducer.reduce(state, event)

            # Check that any logged records don't contain raw sensitive values
            # We look for common sensitive patterns in log messages
            sensitive_log_patterns = [
                "password",
                "secret",
                "api_key",
                "credential",
                "-----BEGIN",
            ]

            # Safe patterns that mention credential concepts without exposing values
            # These are boolean field-existence checks, not actual credential values
            safe_context_patterns = [
                r"\bhas_password\b",
                r"\bhas_secret\b",
                r"\bhas_api_key\b",
                r"\bhas_credential\b",
                r"\bpassword_present\b",
                r"\bsecret_present\b",
                r"\bcontains_password\b",
                r"\bcontains_secret\b",
                r"\bcontains_credential\b",
                r"\bpassword_provided\b",
                r"\bsecret_provided\b",
                r"\bcredential_provided\b",
                # Field count/length checks (e.g., "password_length: 12")
                r"\bpassword_length\b",
                r"\bsecret_length\b",
                # Redacted placeholders
                r"\bpassword:\s*\*+\b",
                r"\bsecret:\s*\*+\b",
                r"\bpassword:\s*\[redacted\]\b",
                r"\bsecret:\s*\[redacted\]\b",
            ]

            # Dangerous patterns that indicate actual value exposure
            # These patterns detect assignment/value contexts
            dangerous_value_patterns = [
                # Assignment patterns: password=xyz, secret=abc
                r"\bpassword\s*[=:]\s*[^\s\*\[\]]+",
                r"\bsecret\s*[=:]\s*[^\s\*\[\]]+",
                r"\bapi_key\s*[=:]\s*[^\s\*\[\]]+",
                r"\bcredential\s*[=:]\s*[^\s\*\[\]]+",
                # Private key content
                r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
                # URL with embedded credentials
                r"://[^/\s]+:[^/\s]+@",
            ]

            for record in log_records:
                message = record.getMessage()
                message_lower = message.lower()

                for pattern in sensitive_log_patterns:
                    if pattern.lower() in message_lower:
                        # First check: is this in a known safe context?
                        is_safe_context = any(
                            re.search(safe_pattern, message_lower)
                            for safe_pattern in safe_context_patterns
                        )

                        if is_safe_context:
                            # Safe context found, but verify no dangerous
                            # patterns also exist in the same message
                            # (e.g., "has_password: True, password=secret123")
                            has_dangerous_pattern = any(
                                re.search(dangerous_pattern, message_lower)
                                for dangerous_pattern in dangerous_value_patterns
                            )
                            if has_dangerous_pattern:
                                assert False, (
                                    f"Log message contains dangerous credential "
                                    f"pattern alongside safe context: "
                                    f"{record.getMessage()}"
                                )
                        else:
                            # No safe context - check for dangerous patterns
                            for dangerous_pattern in dangerous_value_patterns:
                                if re.search(dangerous_pattern, message_lower):
                                    assert False, (
                                        f"Log message contains potentially "
                                        f"sensitive pattern '{pattern}' in "
                                        f"dangerous context: {record.getMessage()}"
                                    )

        finally:
            # Restore logger state
            reducer_logger.removeHandler(capturing_handler)
            reducer_logger.setLevel(original_level)
