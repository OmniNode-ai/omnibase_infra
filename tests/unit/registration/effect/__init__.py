# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Effect node tests for the registration system.

This package contains unit tests for the Registry Effect Node (OMN-890)
and related effect-layer registration functionality.

Test Modules:
    - test_idempotency: Idempotency guard tests for registration operations
    - test_circuit_breaker: Circuit breaker resilience tests
    - test_retry: Retry logic and exponential backoff tests

Fixtures (conftest.py):
    - inmemory_idempotency_store: StoreIdempotencyInmemory for testing
    - mock_consul_client: Mock Consul client for service registration
    - mock_postgres_handler: Mock PostgreSQL handler
    - sample_registry_request: Sample registration request
    - sample_introspection_event: Sample node introspection event
    - correlation_id: UUID fixture for request tracing

Related Tickets:
    - OMN-954: Registry Effect Node tests
    - OMN-890: Registry Effect Node implementation
    - OMN-945: Idempotency system
"""
