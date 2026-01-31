# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for agent_actions observability consumer.

This package contains unit tests for the agent_actions observability
consumer, including model validation, PostgreSQL writer, and Kafka consumer.

Test Categories:
    - test_models: Model validation (strict envelope, flexible payloads)
    - test_writer: PostgreSQL writer with idempotency and circuit breaker
    - test_consumer: Kafka consumer with offset tracking and health checks

Running Tests:
    pytest src/omnibase_infra/services/observability/agent_actions/tests/

Related Tickets:
    - OMN-1743: Migrate agent_actions_consumer to omnibase_infra
"""

__all__: list[str] = []
