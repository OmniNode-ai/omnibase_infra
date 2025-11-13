"""
Conftest for workflow integration tests.

These tests are self-contained and don't require external infrastructure
(Kafka, PostgreSQL, Consul). They test workflow components in isolation
using mocks and in-memory implementations.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def verify_services_available():
    """
    Override parent fixture to skip service availability check.

    Workflow integration tests don't require external services - they use
    mocks and in-memory implementations to test workflow logic in isolation.
    """
    # No service check needed - these are isolated component tests
    pass


@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """
    Override parent fixture to skip database cleanup.

    Workflow tests don't use real database, so no cleanup needed.
    """
    # No cleanup needed - tests use mocks
    yield
