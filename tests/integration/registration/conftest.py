# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest configuration and fixtures for registration integration tests.

This conftest provides registration-specific fixtures for integration tests.

Note:
    Fixtures from tests/integration/handlers/conftest.py (initialized_consul_handler,
    consul_config, db_config, etc.) are imported via pytest_plugins in the root
    tests/conftest.py, making them available throughout the test suite.
"""

# Handler fixtures (initialized_consul_handler, consul_config, db_config, etc.)
# are imported via pytest_plugins in tests/conftest.py (root level).
# This allows pytest to properly share fixtures across the test suite.
# See: https://docs.pytest.org/en/stable/deprecations.html#pytest-plugins-in-non-top-level-conftest-files
