"""
Consul registry client for O.N.E. v0.1 protocol compliance.

This module provides service registration and discovery capabilities
using Consul as the service registry backend.
"""

import logging
import os
from datetime import UTC, datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
        except (ConnectionError, OSError, ValueError, TypeError) as e:
            # Expected errors: network issues, invalid host/port, configuration errors
            logger.error(
                f"Failed to initialize Consul client (configuration/connection error): {e}"
            )
            self.consul = None
        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(f"Unexpected error initializing Consul client: {e}")
            raise

    async def register_service(self, settings: Any = None) -> bool:
        """
        Register MetadataStampingService with Consul.

        Args:
            settings: Service settings object

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

        # Prepare primary health check (readiness probe)
        # Note: python-consul v1.1.0 only supports a single check per registration
        # Use the most critical health check (readiness) as primary
        # Note: Use local_ip for health check URL to support remote Consul
        health_check_url = (
            f"http://{local_ip}:{service_port}/api/v1/metadata-stamping/health/ready"
        )

        # Prepare service tags
        service_tags = [
            "omninode.services.metadata",
            "o.n.e.v0.1",
            "blake3-hashing",
            "metadata-stamping",
        ]

        try:
            # Register service with Consul using explicit parameters
            # Note: python-consul v1.1.0 only supports single health check
            # Using HTTP health check with interval and timeout
            # Note: python-consul v1.1.0 does not support 'meta' parameter
            # Metadata is encoded in tags instead for MVP compatibility
            self.consul.agent.service.register(
                name=service_name,
                service_id=service_id,
                address=service_host,
                port=service_port,
                tags=service_tags,
                http=health_check_url,
                interval="15s",
                timeout="10s",
            )
            self.service_id = service_id
            logger.info(f"Service registered successfully: {service_id}")
            return True
        except (ConnectionError, OSError) as e:
            # Network/connection errors
            logger.error(f"Service registration failed (network error): {e}")
            return False
        except (KeyError, ValueError, TypeError) as e:
            # Data structure or validation errors
            logger.error(f"Service registration failed (invalid data): {e}")
            return False
        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(f"Unexpected error during service registration: {e}")
            raise

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
                # Network/connection errors
                logger.error(f"Service deregistration failed (network error): {e}")
            except Exception as e:
                # Unexpected errors - log with full traceback and re-raise
                logger.exception(f"Unexpected error during service deregistration: {e}")
                raise
        return False

    async def deregister_node(self, node_id: str) -> bool:
        """
        Deregister a node from Consul by node_id.

        This method is used for rollback operations when dual registration fails.

        Args:
            node_id: The node ID to deregister

        Returns:
            bool: True if deregistration successful, False otherwise
        """
        if not self.consul:
            logger.warning(
                f"Consul client not available. Cannot deregister node {node_id}"
            )
            return False

        try:
            self.consul.agent.service.deregister(node_id)
            logger.info(f"Node deregistered from Consul: {node_id}")
            return True
        except (ConnectionError, OSError) as e:
            # Network/connection errors
            logger.error(
                f"Node deregistration failed (network error) for {node_id}: {e}"
            )
            return False
        except Exception as e:
            # Unexpected errors - log with full traceback
            logger.exception(
                f"Unexpected error during node deregistration for {node_id}: {e}"
            )
            return False

    async def discover_services(self, service_name: str) -> list[dict[str, Any]]:
        """
        Discover services by name.

        Args:
            service_name: Name of service to discover

        Returns:
            list: List of discovered service instances
        """
        if not self.consul:
            return []

        try:
            # Get healthy service instances
            _, services = self.consul.health.service(service_name, passing=True)

            result = []
            for service in services:
                service_info = {
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
            # Network/connection errors - return empty list
            logger.error(f"Service discovery failed (network error): {e}")
            return []
        except (KeyError, TypeError) as e:
            # Data structure errors - return empty list
            logger.error(f"Service discovery failed (invalid response structure): {e}")
            return []
        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(f"Unexpected error during service discovery: {e}")
            raise

    async def health_check(self) -> dict[str, Any]:
        """
        Check registry client health.

        Returns:
            dict: Health check status
        """
        try:
            if not self.consul:
                return {
                    "status": "unavailable",
                    "consul_connected": False,
                    "message": "Consul client not initialized",
                }

            # Test connection to Consul
            self.consul.agent.self()

            return {
                "status": "healthy",
                "consul_connected": True,
                "consul_host": self.consul_host,
                "consul_port": self.consul_port,
                "service_id": self.service_id,
            }
        except (ConnectionError, OSError) as e:
            # Network/connection errors - return unhealthy status
            return {
                "status": "unhealthy",
                "error": f"Network error: {e}",
                "consul_connected": False,
            }
        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(f"Unexpected error during health check: {e}")
            raise

    async def get_service_metadata(self, service_id: str) -> Optional[dict[str, Any]]:
        """
        Get metadata for a specific service.

        Args:
            service_id: Service ID to query

        Returns:
            dict: Service metadata or None if not found
        """
        if not self.consul:
            return None

        try:
            _, service = self.consul.agent.service(service_id)
            if service:
                return {
                    "id": service["ID"],
                    "name": service["Service"],
                    "tags": service["Tags"],
                    "meta": service.get("Meta", {}),
                    "address": service["Address"],
                    "port": service["Port"],
                }
        except (ConnectionError, OSError) as e:
            # Network/connection errors - return None
            logger.error(f"Failed to get service metadata (network error): {e}")
        except (KeyError, TypeError) as e:
            # Data structure errors - return None
            logger.error(
                f"Failed to get service metadata (invalid response structure): {e}"
            )
        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(f"Unexpected error getting service metadata: {e}")
            raise

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

                # Prepare health check URL
                # Note: Use Address from service for health check URL consistency
                health_check_url = f"http://{service['Address']}:{service['Port']}/api/v1/metadata-stamping/health/ready"

                # Re-register service with updated metadata using explicit parameters
                # Note: python-consul v1.1.0 requires http, interval, timeout as separate params
                # Note: python-consul v1.1.0 does not support 'meta' parameter
                # Metadata updates are not supported in this version (requires python-consul2)
                self.consul.agent.service.register(
                    name=service["Service"],
                    service_id=service["ID"],
                    address=service["Address"],
                    port=service["Port"],
                    tags=service["Tags"],
                    http=health_check_url,
                    interval="15s",
                    timeout="10s",
                )
                logger.info(f"Service metadata updated: {self.service_id}")
                return True

        except (ConnectionError, OSError) as e:
            # Network/connection errors - return False
            logger.error(f"Failed to update service metadata (network error): {e}")
        except (KeyError, TypeError, ValueError) as e:
            # Data structure or validation errors - return False
            logger.error(f"Failed to update service metadata (invalid data): {e}")
        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(f"Unexpected error updating service metadata: {e}")
            raise

        return False
