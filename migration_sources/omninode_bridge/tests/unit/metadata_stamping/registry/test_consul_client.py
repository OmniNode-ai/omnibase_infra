"""Unit tests for RegistryConsulClient with type safety and exception handling validation."""

from unittest.mock import MagicMock, patch

import pytest

from src.metadata_stamping.registry.consul_client import (
    HealthCheckResult,
    RegistryConsulClient,
    ServiceInfo,
    ServiceMetadata,
)


class MockServiceSettings:
    """Mock implementation of ServiceSettings Protocol."""

    def __init__(
        self, host: str = "localhost", port: int = 8053, local_ip: str = "127.0.0.1"
    ):
        self.service_host = host
        self.service_port = port
        self.local_ip = local_ip


class TestConsulClientTypes:
    """Test type definitions and Protocol compliance."""

    def test_service_settings_protocol(self):
        """Test ServiceSettings Protocol duck typing."""
        mock_settings = MockServiceSettings()

        # Protocol compliance - should have required attributes
        assert hasattr(mock_settings, "service_host")
        assert hasattr(mock_settings, "service_port")
        assert hasattr(mock_settings, "local_ip")

        assert mock_settings.service_host == "localhost"
        assert mock_settings.service_port == 8053
        assert mock_settings.local_ip == "127.0.0.1"

    def test_service_info_typeddict(self):
        """Test ServiceInfo TypedDict structure."""
        service_info: ServiceInfo = {
            "id": "test-service-1",
            "address": "192.168.1.100",
            "port": 8080,
            "tags": ["api", "v1"],
            "meta": {"version": "1.0.0", "env": "dev"},
        }

        assert service_info["id"] == "test-service-1"
        assert service_info["address"] == "192.168.1.100"
        assert service_info["port"] == 8080
        assert len(service_info["tags"]) == 2
        assert service_info["meta"]["version"] == "1.0.0"

    def test_health_check_result_typeddict(self):
        """Test HealthCheckResult TypedDict structure."""
        health_result: HealthCheckResult = {
            "status": "healthy",
            "consul_connected": True,
            "consul_host": "consul.local",
            "consul_port": 8500,
            "service_id": "test-service-1",
        }

        assert health_result["status"] == "healthy"
        assert health_result["consul_connected"] is True
        assert health_result["consul_host"] == "consul.local"

    def test_service_metadata_typeddict(self):
        """Test ServiceMetadata TypedDict structure."""
        metadata: ServiceMetadata = {
            "id": "service-123",
            "name": "my-service",
            "tags": ["production"],
            "meta": {"owner": "team-a"},
            "address": "10.0.0.1",
            "port": 9000,
        }

        assert metadata["id"] == "service-123"
        assert metadata["name"] == "my-service"
        assert metadata["port"] == 9000


class TestConsulClientInitialization:
    """Test Consul client initialization and exception handling."""

    def test_successful_initialization(self):
        """Test successful Consul client initialization (integration test style)."""
        # Since consul library IS available in test env, test actual initialization
        client = RegistryConsulClient(consul_host="localhost", consul_port=8500)

        # Should have set these properties even if consul connection fails
        assert client.consul_host == "localhost"
        assert client.consul_port == 8500
        # Consul may or may not be available - that's ok

    def test_import_error_handling(self):
        """Test graceful handling when Consul initialization fails."""
        # Test that failure to initialize doesn't crash the application
        # This is more of an integration test verifying resilience
        client = RegistryConsulClient(consul_host="nonexistent-host", consul_port=99999)

        # Should have initialized without crashing
        assert client is not None
        assert client.consul_host == "nonexistent-host"
        assert client.consul_port == 99999
        # consul may or may not be None depending on connection

    def test_connection_error_handling(self):
        """Test handling of connection errors during initialization."""
        with patch("builtins.__import__") as mock_import:
            mock_consul_module = MagicMock()
            mock_consul_module.Consul.side_effect = ConnectionError(
                "Connection refused"
            )

            def import_side_effect(name, *args, **kwargs):
                if name == "consul":
                    return mock_consul_module
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            client = RegistryConsulClient()

            assert client.consul is None
            # Should log error but not crash

    def test_unexpected_error_handling(self):
        """Test handling of unexpected errors during initialization."""
        with patch("builtins.__import__") as mock_import:
            mock_consul_module = MagicMock()
            mock_consul_module.Consul.side_effect = RuntimeError("Unexpected error")

            def import_side_effect(name, *args, **kwargs):
                if name == "consul":
                    return mock_consul_module
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            client = RegistryConsulClient()

            assert client.consul is None
            # Should log exception with traceback


class TestConsulClientRegistration:
    """Test service registration with exception handling."""

    @pytest.mark.asyncio
    async def test_successful_registration(self):
        """Test successful service registration."""
        mock_consul_instance = MagicMock()

        client = RegistryConsulClient()
        client.consul = mock_consul_instance  # Directly set the consul instance

        mock_settings = MockServiceSettings(
            host="10.0.0.1", port=8053, local_ip="10.0.0.1"
        )

        result = await client.register_service(mock_settings)

        assert result is True
        assert client.service_id == "metadata-stamping-service-10.0.0.1-8053"
        mock_consul_instance.agent.service.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_registration_connection_error(self):
        """Test registration failure due to connection error."""
        mock_consul_instance = MagicMock()
        mock_consul_instance.agent.service.register.side_effect = ConnectionError(
            "Connection failed"
        )

        client = RegistryConsulClient()
        client.consul = mock_consul_instance
        mock_settings = MockServiceSettings()

        result = await client.register_service(mock_settings)

        assert result is False

    @pytest.mark.asyncio
    async def test_registration_validation_error(self):
        """Test registration failure due to invalid data."""
        mock_consul_instance = MagicMock()
        mock_consul_instance.agent.service.register.side_effect = TypeError(
            "Invalid port type"
        )

        client = RegistryConsulClient()
        client.consul = mock_consul_instance
        mock_settings = MockServiceSettings()

        result = await client.register_service(mock_settings)

        assert result is False

    @pytest.mark.asyncio
    async def test_registration_without_consul(self):
        """Test registration when Consul client is not available."""
        client = RegistryConsulClient()
        client.consul = None  # Simulate missing consul

        mock_settings = MockServiceSettings()

        result = await client.register_service(mock_settings)

        assert result is False


class TestConsulClientDiscovery:
    """Test service discovery with exception handling."""

    @pytest.mark.asyncio
    async def test_successful_discovery(self):
        """Test successful service discovery."""
        mock_consul_instance = MagicMock()
        mock_consul_instance.health.service.return_value = (
            None,
            [
                {
                    "Service": {
                        "ID": "service-1",
                        "Address": "10.0.0.1",
                        "Port": 8080,
                        "Tags": ["api"],
                        "Meta": {"version": "1.0"},
                    }
                }
            ],
        )

        client = RegistryConsulClient()
        client.consul = mock_consul_instance
        services = await client.discover_services("test-service")

        assert len(services) == 1
        assert services[0]["id"] == "service-1"
        assert services[0]["address"] == "10.0.0.1"
        assert services[0]["port"] == 8080

    @pytest.mark.asyncio
    async def test_discovery_connection_error(self):
        """Test discovery failure due to connection error."""
        mock_consul_instance = MagicMock()
        mock_consul_instance.health.service.side_effect = OSError("Network unreachable")

        client = RegistryConsulClient()
        client.consul = mock_consul_instance
        services = await client.discover_services("test-service")

        assert services == []

    @pytest.mark.asyncio
    async def test_discovery_invalid_response(self):
        """Test discovery failure due to invalid response data."""
        mock_consul_instance = MagicMock()
        # Missing required fields in response
        mock_consul_instance.health.service.return_value = (
            None,
            [{"Service": {"ID": "incomplete"}}],  # Missing Address, Port, etc.
        )

        client = RegistryConsulClient()
        client.consul = mock_consul_instance
        services = await client.discover_services("test-service")

        # Should return empty list on KeyError
        assert services == []


class TestConsulClientHealthCheck:
    """Test health check with exception handling."""

    @pytest.mark.asyncio
    async def test_healthy_check(self):
        """Test successful health check."""
        mock_consul_instance = MagicMock()
        mock_consul_instance.agent.self.return_value = {"Config": {}}

        client = RegistryConsulClient(consul_host="localhost", consul_port=8500)
        client.consul = mock_consul_instance
        client.service_id = "test-service-1"

        health = await client.health_check()

        assert health["status"] == "healthy"
        assert health["consul_connected"] is True
        assert health["consul_host"] == "localhost"
        assert health["consul_port"] == 8500
        assert health["service_id"] == "test-service-1"

    @pytest.mark.asyncio
    async def test_unhealthy_check_with_exception(self):
        """Test health check when Consul throws exception."""
        mock_consul_instance = MagicMock()
        mock_consul_instance.agent.self.side_effect = ConnectionError(
            "Connection timeout"
        )

        client = RegistryConsulClient()
        client.consul = mock_consul_instance
        health = await client.health_check()

        assert health["status"] == "unhealthy"
        assert health["consul_connected"] is False
        assert "error" in health

    @pytest.mark.asyncio
    async def test_unavailable_check_no_consul(self):
        """Test health check when Consul client not initialized."""
        client = RegistryConsulClient()
        client.consul = None
        health = await client.health_check()

        assert health["status"] == "unavailable"
        assert health["consul_connected"] is False
        assert health["message"] == "Consul client not initialized"


class TestConsulClientMetadata:
    """Test metadata operations with exception handling."""

    @pytest.mark.asyncio
    async def test_get_service_metadata_success(self):
        """Test successful metadata retrieval."""
        mock_consul_instance = MagicMock()
        mock_consul_instance.agent.service.return_value = (
            None,
            {
                "ID": "service-1",
                "Service": "my-service",
                "Tags": ["prod"],
                "Meta": {"owner": "team-a"},
                "Address": "10.0.0.1",
                "Port": 9000,
            },
        )

        client = RegistryConsulClient()
        client.consul = mock_consul_instance
        metadata = await client.get_service_metadata("service-1")

        assert metadata is not None
        assert metadata["id"] == "service-1"
        assert metadata["name"] == "my-service"
        assert metadata["address"] == "10.0.0.1"

    @pytest.mark.asyncio
    async def test_get_service_metadata_connection_error(self):
        """Test metadata retrieval failure due to connection error."""
        mock_consul_instance = MagicMock()
        mock_consul_instance.agent.service.side_effect = ConnectionError("Timeout")

        client = RegistryConsulClient()
        client.consul = mock_consul_instance
        metadata = await client.get_service_metadata("service-1")

        assert metadata is None

    @pytest.mark.asyncio
    async def test_update_service_metadata_success(self):
        """Test successful metadata update."""
        mock_consul_instance = MagicMock()
        mock_consul_instance.agent.service.return_value = (
            None,
            {
                "ID": "service-1",
                "Service": "my-service",
                "Tags": ["prod"],
                "Meta": {"owner": "team-a"},
                "Address": "10.0.0.1",
                "Port": 9000,
            },
        )

        client = RegistryConsulClient()
        client.consul = mock_consul_instance
        client.service_id = "service-1"

        result = await client.update_service_metadata({"version": "2.0"})

        assert result is True
        mock_consul_instance.agent.service.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_service_metadata_connection_error(self):
        """Test metadata update failure due to connection error."""
        mock_consul_instance = MagicMock()
        mock_consul_instance.agent.service.side_effect = OSError("Network error")

        client = RegistryConsulClient()
        client.consul = mock_consul_instance
        client.service_id = "service-1"

        result = await client.update_service_metadata({"version": "2.0"})

        assert result is False


class TestConsulClientDeregistration:
    """Test service deregistration with exception handling."""

    @pytest.mark.asyncio
    async def test_successful_deregistration(self):
        """Test successful service deregistration."""
        mock_consul_instance = MagicMock()

        client = RegistryConsulClient()
        client.consul = mock_consul_instance
        client.service_id = "test-service-1"

        result = await client.deregister_service()

        assert result is True
        mock_consul_instance.agent.service.deregister.assert_called_once_with(
            "test-service-1"
        )

    @pytest.mark.asyncio
    async def test_deregistration_connection_error(self):
        """Test deregistration failure due to connection error."""
        mock_consul_instance = MagicMock()
        mock_consul_instance.agent.service.deregister.side_effect = ConnectionError(
            "Timeout"
        )

        client = RegistryConsulClient()
        client.consul = mock_consul_instance
        client.service_id = "test-service-1"

        result = await client.deregister_service()

        assert result is False

    @pytest.mark.asyncio
    async def test_deregistration_no_consul(self):
        """Test deregistration when Consul not available."""
        client = RegistryConsulClient()
        client.consul = None

        result = await client.deregister_service()

        assert result is False
