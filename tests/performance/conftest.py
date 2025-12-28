# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest configuration for all performance tests.

This conftest.py automatically applies the `performance` marker to all tests
in the tests/performance/ directory hierarchy, providing consistent test
categorization without requiring individual files to set pytestmark.

Marker Application:
    All tests under tests/performance/** are automatically marked with:
    - pytest.mark.performance

    NOTE: pytestmark at module-level in conftest.py does NOT automatically
    apply to tests in other files. We use pytest_collection_modifyitems hook
    instead to dynamically mark all tests in the performance directory.

This enables selective test execution:
    # Run only performance tests
    pytest -m performance

    # Run all except performance tests
    pytest -m "not performance"

Related:
    - pyproject.toml: Marker definitions
    - tests/conftest.py: Global test fixtures
"""

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Dynamically add performance marker to all tests in the performance directory.

    This hook runs after test collection and adds the 'performance' marker to any
    test whose file path contains 'tests/performance'. This is necessary because
    pytestmark defined in conftest.py does NOT automatically apply to tests
    in other files within the same directory.

    Args:
        config: Pytest configuration object.
        items: List of collected test items.

    Usage:
        Run only performance tests: pytest -m performance
        Exclude performance tests: pytest -m "not performance"
    """
    performance_marker = pytest.mark.performance

    for item in items:
        # Check if the test file is in the performance directory
        if "tests/performance" in str(item.fspath):
            # Only add marker if not already present
            if not any(marker.name == "performance" for marker in item.iter_markers()):
                item.add_marker(performance_marker)
