"""
Minimal test to debug fixture setup issues in cross-service integration tests.
"""

import pytest


def test_import_debug():
    """Debug import issues with fixtures."""
    print("🔍 Testing imports...")

    try:
        from tests.fixtures.integration_containers import full_integration_env

        _ = full_integration_env  # Mark as used
        print("✅ full_integration_env imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import full_integration_env: {e}")

    try:
        from tests.fixtures.test_data_isolation import isolated_integration_env

        _ = isolated_integration_env  # Mark as used
        print("✅ isolated_integration_env imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import isolated_integration_env: {e}")

    # Note: integrated_services is a class-level fixture, not importable at module level
    print(
        "info: integrated_services is a class-level fixture (not module-level importable)"
    )

    # Check if enhanced fixtures are available from conftest
    try:
        import tests.conftest as conftest_module

        enhanced_available = getattr(
            conftest_module, "ENHANCED_FIXTURES_AVAILABLE", False
        )
        print(f"🔧 ENHANCED_FIXTURES_AVAILABLE: {enhanced_available}")

        containers_available = getattr(conftest_module, "CONTAINERS_AVAILABLE", False)
        print(f"🔧 CONTAINERS_AVAILABLE: {containers_available}")
    except Exception as e:
        print(f"❌ Failed to check conftest status: {e}")


@pytest.mark.asyncio
async def test_isolated_integration_env_fixture_only(isolated_integration_env):
    """Test that only uses the isolated_integration_env fixture to verify it works."""
    print("✅ isolated_integration_env fixture loaded successfully")
    print(f"Database manager: {isolated_integration_env.db_manager}")
    print(f"Kafka manager: {isolated_integration_env.kafka_manager}")
    assert isolated_integration_env is not None
    assert isolated_integration_env.db_manager is not None
    assert isolated_integration_env.kafka_manager is not None


@pytest.mark.asyncio
async def test_full_integration_env_fixture_only(full_integration_env):
    """Test that only uses the full_integration_env fixture to verify it works."""
    print("✅ full_integration_env fixture loaded successfully")
    print(f"Integration env: {full_integration_env}")
    assert full_integration_env is not None


@pytest.mark.asyncio
async def test_integrated_services_fixture_only(integrated_services):
    """Test that only uses the integrated_services fixture to verify it works."""
    print("✅ integrated_services fixture loaded successfully")
    kafka_client, postgres_client, hook_service, workflow_service = integrated_services
    print(f"Kafka client: {kafka_client}")
    print(f"Postgres client: {postgres_client}")
    print(f"Hook service: {hook_service}")
    print(f"Workflow service: {workflow_service}")
    assert kafka_client is not None
    assert postgres_client is not None
    assert hook_service is not None
    assert workflow_service is not None
