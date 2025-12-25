# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Chaos testing module for OMN-955.

This module contains chaos engineering tests that validate system resilience
under various failure conditions. Tests in this module simulate real-world
failure scenarios to ensure the infrastructure can handle:

- Handler failures (random exceptions during processing)
- Network partitions (event bus disconnections)
- Timeouts (handlers exceeding time limits)
- Partial failures (some effects succeed, others fail)

Test Organization:
    Chaos Scenarios:
        - test_chaos_handler_failures.py: Tests for handler failure scenarios
        - test_chaos_network_partitions.py: Tests for network partition scenarios
        - test_chaos_timeouts.py: Tests for timeout scenarios
        - test_chaos_partial_failures.py: Tests for partial failure scenarios

    Failure Recovery:
        - test_recovery_restart_resume.py: Restart mid-workflow and resume tests
        - test_recovery_circuit_breaker.py: Circuit breaker state management tests
        - test_recovery_dlq.py: Dead Letter Queue capture tests

Related Tickets:
    - OMN-955: Implement chaos scenario tests and failure recovery tests
    - OMN-954: Effect Idempotency
    - OMN-945: Idempotency System
"""
