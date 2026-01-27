# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest configuration and fixtures for registration integration tests.

This conftest imports fixtures from the handlers conftest to enable
Consul, PostgreSQL, Graph, Qdrant, Vault, and HTTP integration testing
in registration tests.

Available Fixtures
------------------
Infrastructure Availability Flags:
    CONSUL_AVAILABLE, POSTGRES_AVAILABLE, GRAPH_AVAILABLE,
    QDRANT_AVAILABLE, VAULT_AVAILABLE, POSTGRES_HOST
    - Boolean/string constants indicating if infrastructure is reachable.

Skip Marker Fixtures:
    consul_available, graph_available, qdrant_available, vault_available
    - Pytest fixtures that skip tests if infrastructure is unavailable.

Configuration Fixtures:
    consul_config, db_config, graph_config, qdrant_config, vault_config,
    http_handler_config, small_response_config
    - Return configuration dicts for handler initialization.

Initialized Handler Fixtures:
    initialized_consul_handler, initialized_db_handler, initialized_graph_handler,
    initialized_qdrant_handler, vault_handler
    - Provide ready-to-use handler instances with automatic cleanup.

Unique ID Fixtures (for test isolation):
    unique_kv_key, unique_service_name, unique_table_name,
    unique_node_label, unique_collection_name
    - Generate unique identifiers to prevent test interference.

Cleanup Fixtures:
    cleanup_table
    - Track resources for cleanup after tests complete.

Common Fixtures:
    mock_container
    - Provides a mock ModelONEXContainer for handler construction.

Fixture Import Pattern
----------------------
pytest discovers fixtures in conftest.py files through the directory hierarchy.
Since tests/integration/handlers/ is a sibling directory (not a parent),
fixtures from that conftest are NOT automatically available here.

To share fixtures across sibling directories, we use explicit Python imports
and re-export them. This makes pytest treat them as local fixtures.

Important: Do NOT use pytest_plugins for conftest.py files that are already
in the test tree, as this causes "Plugin already registered" errors when
pytest also discovers them during directory collection.

Reference:
    https://docs.pytest.org/en/stable/how-to/fixtures.html#using-fixtures-from-other-projects
"""

# =============================================================================
# Re-export fixtures from handlers conftest for registration tests
# =============================================================================
# These imports make the handler fixtures available to registration tests.
# pytest discovers fixtures by name, so importing them here is sufficient.
#
# Note: Using explicit imports instead of pytest_plugins avoids the
# "Plugin already registered under a different name" error that occurs
# when pytest_plugins references a conftest.py that's also in the test tree.
# =============================================================================

from tests.integration.handlers.conftest import (
    # Consul fixtures
    CONSUL_AVAILABLE,
    # Graph fixtures
    GRAPH_AVAILABLE,
    # Database fixtures
    POSTGRES_AVAILABLE,
    POSTGRES_HOST,
    # Qdrant fixtures
    QDRANT_AVAILABLE,
    # Vault fixtures
    VAULT_AVAILABLE,
    cleanup_table,
    consul_available,
    consul_config,
    db_config,
    graph_available,
    graph_config,
    # HTTP fixtures
    http_handler_config,
    initialized_consul_handler,
    initialized_db_handler,
    initialized_graph_handler,
    initialized_qdrant_handler,
    # Common fixtures
    mock_container,
    qdrant_available,
    qdrant_config,
    small_response_config,
    unique_collection_name,
    unique_kv_key,
    unique_node_label,
    unique_service_name,
    unique_table_name,
    vault_available,
    vault_config,
    vault_handler,
)
