# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest configuration for all unit tests.

This conftest.py automatically applies the `unit` marker to all tests
in the tests/unit/ directory hierarchy, providing consistent test
categorization without requiring individual files to set pytestmark.

Marker Application:
    All tests under tests/unit/** are automatically marked with:
    - pytest.mark.unit

    NOTE: pytestmark at module-level in conftest.py does NOT automatically
    apply to tests in other files. We use pytest_collection_modifyitems hook
    instead to dynamically mark all tests in the unit directory.

This enables selective test execution:
    # Run only unit tests
    pytest -m unit

    # Run all except unit tests
    pytest -m "not unit"

Related:
    - pyproject.toml: Marker definitions
    - tests/conftest.py: Global test fixtures
"""

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Dynamically add unit marker to all tests in the unit directory.

    This hook runs after test collection and adds the 'unit' marker to any
    test whose file path contains 'tests/unit'. This is necessary because
    pytestmark defined in conftest.py does NOT automatically apply to tests
    in other files within the same directory.

    Args:
        config: Pytest configuration object.
        items: List of collected test items.

    Usage:
        Run only unit tests: pytest -m unit
        Exclude unit tests: pytest -m "not unit"
    """
    unit_marker = pytest.mark.unit

    for item in items:
        # Check if the test file is in the unit directory
        if "tests/unit" in str(item.fspath):
            # Only add marker if not already present
            if not any(marker.name == "unit" for marker in item.iter_markers()):
                item.add_marker(unit_marker)
