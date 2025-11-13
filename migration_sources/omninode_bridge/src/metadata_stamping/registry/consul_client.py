"""
Consul registry client for O.N.E. v0.1 protocol compliance.

This module provides service registration and discovery capabilities
using Consul as the service registry backend.
"""

import logging
import os
from datetime import UTC, datetime
from typing import Optional, Protocol, TypedDict

logger = logging.getLogger(__name__)


class ServiceSettings(Protocol):
    """Protocol for service settings objects (duck typing for type safety)."""

    service_host: str
    service_port: int
    local_ip: str


class ServiceInfo(TypedDict):
    """Type definition for service information returned by discovery."""

    id: str
    address: str
    port: int
    tags: list[str]
    meta: dict[str, str]


class HealthCheckResult(TypedDict, total=False):
    """Type definition for health check results."""

    status: str
    consul_connected: bool
    consul_host: str
    consul_port: int
    service_id: Optional[str]
    message: str
    error: str


class ServiceMetadata(TypedDict, total=False):
    """Type definition for service metadata."""

    id: str
    name: str
    tags: list[str]
    meta: dict[str, str]
    address: str
    port: int


class RegistryConsulClient:
    """
    Consul registry client for service registration and discovery.

    Provides O.N.E. v0.1 protocol compliant service registration
    with health checks and metadata support.
    """

    def __init__(
        self, consul_host: Optional[str] = None, consul_port: Optional[int] = None
    ):
        """
        Initialize Consul registry client.

        Args:
            consul_host: Consul server hostname (defaults to CONSUL_HOST env var or "omninode-bridge-consul")
            consul_port: Consul server port (defaults to CONSUL_PORT env var or 8500)
        """
        self.consul_host = consul_host or os.getenv(
            "CONSUL_HOST", "omninode-bridge-consul"
        )
        self.consul_port = consul_port or int(os.getenv("CONSUL_PORT", "8500"))
        self.service_id = None
        self.consul = None
        self._initialize_consul_client()

    def _initialize_consul_client(self):
        """Initialize the Consul client connection."""
        try:
            import consul

            self.consul = consul.Consul(host=self.consul_host, port=self.consul_port)
            logger.info(
                f"Initialized Consul client: {self.consul_host}:{self.consul_port}"
            )
        except ImportError:
            logger.warning("python-consul2 not installed. Registry features disabled.")
            self.consul = None
        except (ConnectionError, OSError) as e:
            logger.error(f"Failed to connect to Consul: {e}")
            self.consul = None
        except Exception as e:
            # Unexpected errors - log and set to None but don't crash
            logger.exception(f"Unexpected error initializing Consul client: {e}")
            self.consul = None

    async def register_service(
        self, settings: Optional[ServiceSettings] = None
    ) -> bool:
        """
        Register MetadataStampingService with Consul.

        Args:
            settings: Service settings object with service_host, service_port, and local_ip

        Returns:
            bool: True if registration successful, False otherwise
        """
        if not self.consul:
            logger.warning(
                "Consul client not available. Skipping service registration."
            )
            return False

        service_name = "metadata-stamping-service"
        service_host = getattr(settings, "service_host", "localhost")
        service_port = getattr(settings, "service_port", 8053)
        # Use local_ip for health check URL (supports remote Consul)
        local_ip = getattr(settings, "local_ip", service_host)

        service_id = f"{service_name}-{service_host}-{service_port}"

        # Prepare service metadata
        service_meta = {
            "version": "0.1.0",
            "protocol": "O.N.E.v0.1",
            "namespace": "omninode.services.metadata",
            "capabilities": "hashing,stamping,validation",
            "registered_at": datetime.now(UTC).isoformat(),
        }

        # Prepare health check configuration
        # Note: Use local_ip for health check URL to support remote Consul
        health_check = {
            "http": f"http://{local_ip}:{service_port}/health",
            "interval": "10s",
            "timeout": "5s",
        }

        # Prepare service tags
        service_tags = [
            "omninode.services.metadata",
            "o.n.e.v0.1",
            "blake3-hashing",
            "metadata-stamping",
        ]

        try:
            # Register service with Consul using explicit parameters
            # Note: python-consul expects meta as a separate parameter, not in config dict
            self.consul.agent.service.register(
                name=service_name,
                service_id=service_id,
                address=service_host,
                port=service_port,
                tags=service_tags,
                meta=service_meta,
                check=health_check,
            )
            self.service_id = service_id
            logger.info(f"Service registered successfully: {service_id}")
            return True
        except (ConnectionError, OSError) as e:
            logger.error(f"Service registration failed (connection error): {e}")
            return False
        except (ValueError, TypeError) as e:
            logger.error(f"Service registration failed (invalid data): {e}")
            return False
        except Exception as e:
            # Unexpected errors - log with traceback
            logger.exception(f"Unexpected error during service registration: {e}")
            return False

    async def deregister_service(self) -> bool:
        """
        Deregister service from Consul.

        Returns:
            bool: True if deregistration successful, False otherwise
        """
        if not self.consul:
            return False

        if self.service_id:
            try:
                self.consul.agent.service.deregister(self.service_id)
                logger.info(f"Service deregistered: {self.service_id}")
                return True
            except (ConnectionError, OSError) as e:
                logger.error(f"Service deregistration failed (connection error): {e}")
            except Exception as e:
                # Unexpected errors - log with traceback
                logger.exception(f"Unexpected error during service deregistration: {e}")
        return False

    async def discover_services(self, service_name: str) -> list[ServiceInfo]:
        """
        Discover services by name.

        Args:
            service_name: Name of service to discover

        Returns:
            list: List of discovered service instances with ServiceInfo structure
        """
        if not self.consul:
            return []

        try:
            # Get healthy service instances
            _, services = self.consul.health.service(service_name, passing=True)

            result = []
            for service in services:
                service_info: ServiceInfo = {
                    "id": service["Service"]["ID"],
                    "address": service["Service"]["Address"],
                    "port": service["Service"]["Port"],
                    "tags": service["Service"]["Tags"],
                    "meta": service["Service"].get("Meta", {}),
                }
                result.append(service_info)

            logger.info(f"Discovered {len(result)} instances of {service_name}")
            return result

        except (ConnectionError, OSError) as e:
            logger.error(f"Service discovery failed (connection error): {e}")
            return []
        except (KeyError, TypeError) as e:
            logger.error(f"Service discovery failed (invalid response data): {e}")
            return []
        except Exception as e:
            # Unexpected errors - log with traceback
            logger.exception(f"Unexpected error during service discovery: {e}")
            return []

    async def health_check(self) -> HealthCheckResult:
        """
        Check registry client health.

        Returns:
            HealthCheckResult: Health check status with structured fields
        """
        try:
            if not self.consul:
                return HealthCheckResult(
                    status="unavailable",
                    consul_connected=False,
                    message="Consul client not initialized",
                )

            # Test connection to Consul
            self.consul.agent.self()

            return HealthCheckResult(
                status="healthy",
                consul_connected=True,
                consul_host=self.consul_host,
                consul_port=self.consul_port,
                service_id=self.service_id,
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                status="unhealthy", error=str(e), consul_connected=False
            )

    async def get_service_metadata(self, service_id: str) -> Optional[ServiceMetadata]:
        """
        Get metadata for a specific service.

        Args:
            service_id: Service ID to query

        Returns:
            ServiceMetadata: Service metadata or None if not found
        """
        if not self.consul:
            return None

        try:
            _, service = self.consul.agent.service(service_id)
            if service:
                return ServiceMetadata(
                    id=service["ID"],
                    name=service["Service"],
                    tags=service["Tags"],
                    meta=service.get("Meta", {}),
                    address=service["Address"],
                    port=service["Port"],
                )
        except (ConnectionError, OSError) as e:
            logger.error(f"Failed to get service metadata (connection error): {e}")
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to get service metadata (invalid response data): {e}")
        except Exception as e:
            # Unexpected errors - log with traceback
            logger.exception(f"Unexpected error getting service metadata: {e}")

        return None

    async def update_service_metadata(self, metadata: dict[str, str]) -> bool:
        """
        Update service metadata in Consul.

        Args:
            metadata: Metadata dictionary to update

        Returns:
            bool: True if update successful
        """
        if not self.consul or not self.service_id:
            return False

        try:
            # Re-register service with updated metadata
            _, service = self.consul.agent.service(self.service_id)
            if service:
                # Merge existing metadata with updates
                updated_meta = {**service.get("Meta", {}), **metadata}

                # Prepare health check
                # Note: Use Address from service for health check URL consistency
                health_check = {
                    "http": f"http://{service['Address']}:{service['Port']}/health",
                    "interval": "10s",
                    "timeout": "5s",
                }

                # Re-register service with updated metadata using explicit parameters
                self.consul.agent.service.register(
                    name=service["Service"],
                    service_id=service["ID"],
                    address=service["Address"],
                    port=service["Port"],
                    tags=service["Tags"],
                    meta=updated_meta,
                    check=health_check,
                )
                logger.info(f"Service metadata updated: {self.service_id}")
                return True

        except (ConnectionError, OSError) as e:
            logger.error(f"Failed to update service metadata (connection error): {e}")
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to update service metadata (invalid data): {e}")
        except Exception as e:
            # Unexpected errors - log with traceback
            logger.exception(f"Unexpected error updating service metadata: {e}")

        return False
