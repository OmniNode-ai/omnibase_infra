# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Workflow integration tests for node registration E2E scenarios.

This package provides mocked E2E tests for the registration workflow,
proving the architecture with ZERO real infrastructure dependencies.

Test Files:
    - conftest.py: Shared fixtures for workflow tests
    - test_workflow_*.py: Individual test modules for scenarios A0-A6

Design Principles:
    - No real infrastructure: All external dependencies are mocked
    - Call count tracking: Mocks track invocations for purity verification
    - Deterministic time: DeterministicClock enables time control
    - Snapshot normalization: Helpers for reproducible test output
    - Correlation ID tracing: All fixtures support correlation tracking

Available Fixtures (from conftest.py):
    Clock & UUID:
        - deterministic_clock: DeterministicClock for time control
        - uuid_generator: DeterministicUUIDGenerator for reproducible UUIDs
        - correlation_id: Pre-generated deterministic correlation ID

    Mock Effects (with call tracking):
        - mock_consul_effect: MockConsulEffect with failure injection
        - mock_postgres_effect: MockPostgresEffect with failure injection
        - registry_effect_with_mocks: NodeRegistryEffect wired with mocks

    Component Fixtures:
        - registration_reducer: RegistrationReducer instance
        - initial_state: ModelRegistrationState in idle status
        - tracked_reducer: TrackedRegistrationReducer for call tracking
        - tracked_effect: TrackedNodeRegistryEffect for call tracking
        - call_tracker: CallOrderTracker for verifying execution order

    Event Factories:
        - introspection_event_factory: Creates random introspection events
        - deterministic_introspection_event_factory: Deterministic events
        - registry_request_factory: Creates registry requests

    Observability:
        - caplog: Use pytest's built-in caplog fixture for log capture

    Test Helpers:
        - snapshot_normalizer: SnapshotNormalizer for stable assertions
        - workflow_context: WorkflowScenarioContext with all components
        - failure_injector: FailureInjector for setting up failure scenarios

Related:
    - OMN-915: Mocked E2E tests proving the registration architecture
"""

__all__: list[str] = []
