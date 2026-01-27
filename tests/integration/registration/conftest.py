# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest configuration and fixtures for registration integration tests.

This conftest imports common fixtures from the handlers conftest to enable
Consul, PostgreSQL, and Kafka integration testing in registration tests.

Note:
    Fixtures from tests/integration/handlers/conftest.py are imported via
    pytest_plugins to make them available in registration tests.
"""

# Import fixtures from handlers conftest for Consul, DB, etc.
# This makes initialized_consul_handler, consul_config, etc. available here.
pytest_plugins = ["tests.integration.handlers.conftest"]
