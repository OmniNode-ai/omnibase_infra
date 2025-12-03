"""Pytest configuration and shared fixtures for omnibase_infra tests."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_container() -> MagicMock:
    """Create mock ONEX container for testing."""
    container = MagicMock()
    container.get_config.return_value = {}
    return container
