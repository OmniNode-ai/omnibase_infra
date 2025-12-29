# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest configuration for all integration tests.

This conftest.py automatically applies the `integration` marker to all tests
in the tests/integration/ directory hierarchy, providing consistent test
categorization without requiring individual files to set pytestmark.

Marker Application:
    All tests under tests/integration/** are automatically marked with:
    - pytest.mark.integration

    NOTE: pytestmark at module-level in conftest.py does NOT automatically
    apply to tests in other files. We use pytest_collection_modifyitems hook
    instead to dynamically mark all tests in the integration directory.

This enables selective test execution:
    # Run only integration tests
    pytest -m integration

    # Run all except integration tests
    pytest -m "not integration"

Related:
    - pyproject.toml: Marker definitions
    - tests/conftest.py: Global test fixtures
"""

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Dynamically add integration marker to all tests in the integration directory.

    This hook runs after test collection and adds the 'integration' marker to any
    test whose file path contains 'tests/integration'. This is necessary because
    pytestmark defined in conftest.py does NOT automatically apply to tests
    in other files within the same directory.

    Args:
        config: Pytest configuration object.
        items: List of collected test items.

    Usage:
        Run only integration tests: pytest -m integration
        Exclude integration tests: pytest -m "not integration"
    """
    integration_marker = pytest.mark.integration

    for item in items:
        # Check if the test file is in the integration directory
        if "tests/integration" in str(item.fspath):
            # Only add marker if not already present
            if not any(marker.name == "integration" for marker in item.iter_markers()):
                item.add_marker(integration_marker)
