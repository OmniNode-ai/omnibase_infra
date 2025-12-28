# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest configuration for all unit tests.

This conftest.py automatically applies the `unit` marker to all tests
in the tests/unit/ directory hierarchy, providing consistent test
categorization without requiring individual files to set pytestmark.

Marker Application:
    All tests under tests/unit/** are automatically marked with:
    - pytest.mark.unit

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

# Apply unit marker to all tests in this directory tree
pytestmark = [pytest.mark.unit]
