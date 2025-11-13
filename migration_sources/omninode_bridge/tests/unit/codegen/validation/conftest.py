"""
Pytest configuration for validation tests.

Isolated from parent conftest to avoid omnibase_core dependency issues.
"""

import pytest


# Enable async test support
@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy."""
    import asyncio

    return asyncio.get_event_loop_policy()
