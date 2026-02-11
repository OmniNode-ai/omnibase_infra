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

from collections.abc import Generator
from unittest.mock import AsyncMock, patch

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


# =============================================================================
# Dependency Materialization Skip Fixture (unit tests only)
# =============================================================================
# RuntimeHostProcess._materialize_dependencies() requires OMNIBASE_INFRA_DB_URL
# and a live PostgreSQL connection. Unit tests exercise handler discovery,
# bootstrap, source mode resolution, or kernel lifecycle -- not dependency
# materialization (which has its own dedicated tests in
# test_dependency_materializer.py). This fixture patches the method to
# avoid requiring a live database in unrelated unit tests.
#
# Scoped to tests/unit/ only. Integration tests that need this mock should
# define their own local fixture.
# =============================================================================


@pytest.fixture(autouse=True)
def _skip_materialize_dependencies() -> Generator[None, None, None]:
    """Skip dependency materialization which requires OMNIBASE_INFRA_DB_URL.

    This fixture patches RuntimeHostProcess._materialize_dependencies with an
    AsyncMock so that unit tests exercising handler registration, source mode
    resolution, bootstrap flow, or kernel lifecycle do not need a live
    PostgreSQL connection.

    This fixture is ``autouse=True`` but scoped to ``tests/unit/`` only.
    Integration tests that need this mock should define a local fixture.

    Override:
        Tests that need real materialization behaviour (e.g.,
        ``test_dependency_materializer.py``) can override by defining a
        same-named fixture in their own module or conftest that yields
        without patching::

            @pytest.fixture(autouse=True)
            def _skip_materialize_dependencies():
                yield  # no-op — let real materialisation run
    """
    with patch(
        "omnibase_infra.runtime.service_runtime_host_process"
        ".RuntimeHostProcess._materialize_dependencies",
        new_callable=AsyncMock,
    ):
        yield
