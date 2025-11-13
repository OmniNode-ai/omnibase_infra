"""
Conftest for production validation tests.

These tests validate production-readiness and don't require external
infrastructure (Kafka, PostgreSQL, Consul) as they use mocks.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def verify_services_available():
    """
    Override parent fixture to skip service availability check.

    Production tests use mocks and in-memory implementations to test
    production-readiness in isolation.
    """
    # No service check needed - these are isolated component tests
    pass


@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """
    Override parent fixture to skip database cleanup.

    Production tests don't use real database, so no cleanup needed.
    """
    # No cleanup needed - tests use mocks
    yield
