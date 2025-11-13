"""
Coordination integration test configuration.

These tests use in-memory components (ThreadSafeState, SignalCoordinator, etc.)
and do not require external services (Kafka, PostgreSQL, Consul).

We override the session-scoped verify_services_available fixture to skip
the service availability check since coordination tests are fully in-memory.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def verify_services_available():
    """Override the parent fixture to skip service checks.

    Coordination tests use in-memory components only:
    - ThreadSafeState (in-memory state)
    - SignalCoordinator (in-memory signals)
    - ContextDistributor (in-memory context)
    - DependencyResolver (in-memory resolution)
    - SmartRoutingOrchestrator (in-memory routing)

    No external services (Kafka, PostgreSQL, Consul) are required.
    """
    # Coordination tests run entirely in-memory, no service checks needed
    pass
