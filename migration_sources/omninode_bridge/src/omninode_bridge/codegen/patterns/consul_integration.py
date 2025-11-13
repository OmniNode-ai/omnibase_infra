#!/usr/bin/env python3
"""
Consul Service Discovery Integration Patterns for ONEX v2.0 Nodes.

This module provides production-ready code generation patterns for:
1. Service Registration - Register node with Consul on startup
2. Service Discovery - Find and connect to other services
3. Service Deregistration - Clean removal on shutdown

Part of Phase 2 codegen automation to reduce manual completion from 50% → 10%.

Design Principles:
- Graceful degradation when Consul unavailable
- Production-ready error handling
- omnibase_core ModelContainer integration
- Structured logging with emit_log_event
- Async/await patterns
- ONEX v2.0 compliance

Usage:
    >>> generator = ConsulPatternGenerator()
    >>> registration_code = generator.generate_registration(
    ...     node_type="effect",
    ...     service_name="postgres_crud",
    ...     port=8000
    ... )
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Pattern Templates
CONSUL_REGISTRATION_TEMPLATE = '''
    async def _register_with_consul(self) -> None:
        """
        Register this node with Consul service discovery.

        Gracefully handles Consul unavailability by logging warnings and continuing.
        This allows nodes to start even if Consul is temporarily unavailable.

        Registration includes:
        - Service name and unique ID
        - Network address and port
        - Health check endpoint with automatic deregistration
        - Metadata (node_type, node_id, version)
        - Tags for service categorization
        """
        if not self.container.consul_client:
            emit_log_event(
                LogLevel.WARNING,
                "Consul client not available, skipping registration",
                {
                    "node_id": str(self.node_id),
                    "service_name": "{{ service_name }}",
                    "reason": "consul_client_not_configured"
                }
            )
            return

        try:
            # Generate unique service ID (service_name + short node_id)
            service_id = f"{{ service_name }}-{str(self.node_id)[:8]}"

            # Register service with Consul
            await self.container.consul_client.agent.service.register(
                name="{{ service_name }}",
                service_id=service_id,
                address="0.0.0.0",
                port={{ port }},
                tags=[
                    "{{ node_type }}",
                    "onex-v2",
                    "generated",
                    "{{ domain }}"
                ],
                check={
                    "http": "http://0.0.0.0:{{ port }}{{ health_endpoint }}",
                    "interval": "10s",
                    "timeout": "5s",
                    "deregister_critical_service_after": "1m"
                },
                meta={
                    "node_type": "{{ node_type }}",
                    "node_id": str(self.node_id),
                    "version": "{{ version }}",
                    "domain": "{{ domain }}",
                    "generated_by": "omninode_codegen"
                }
            )

            # Store service ID for deregistration
            self._consul_service_id = service_id

            emit_log_event(
                LogLevel.INFO,
                f"Successfully registered with Consul: {service_id}",
                {
                    "node_id": str(self.node_id),
                    "service_id": service_id,
                    "service_name": "{{ service_name }}",
                    "port": {{ port }},
                    "health_endpoint": "{{ health_endpoint }}"
                }
            )

        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Consul registration failed: {e}",
                {
                    "node_id": str(self.node_id),
                    "service_name": "{{ service_name }}",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            # Don't raise - allow node to start without Consul
'''

CONSUL_DISCOVERY_TEMPLATE = '''
    async def _discover_service(self, service_name: str) -> Optional[str]:
        """
        Discover service endpoint via Consul.

        Performs health-aware service discovery with:
        - Automatic filtering of unhealthy instances
        - Random selection from healthy instances
        - Caching of discovered endpoints (5min TTL)
        - Graceful fallback when Consul unavailable

        Args:
            service_name: Name of service to discover (e.g., "postgres_crud")

        Returns:
            Service endpoint URL (e.g., "http://192.168.1.100:8000") or None
        """
        if not self.container.consul_client:
            emit_log_event(
                LogLevel.WARNING,
                f"Consul client not available, cannot discover {service_name}",
                {
                    "node_id": str(self.node_id),
                    "target_service": service_name,
                    "reason": "consul_client_not_configured"
                }
            )
            return None

        try:
            # Check cache first (5min TTL)
            cache_key = f"consul_service_{service_name}"
            if hasattr(self, '_service_cache'):
                cached = self._service_cache.get(cache_key)
                if cached and (datetime.now(UTC) - cached['timestamp']).seconds < 300:
                    emit_log_event(
                        LogLevel.DEBUG,
                        f"Using cached endpoint for {service_name}",
                        {
                            "node_id": str(self.node_id),
                            "target_service": service_name,
                            "endpoint": cached['endpoint']
                        }
                    )
                    return cached['endpoint']
            else:
                self._service_cache = {}

            # Query Consul for healthy service instances
            _, services = await self.container.consul_client.health.service(
                service_name,
                passing=True  # Only healthy instances
            )

            if not services:
                emit_log_event(
                    LogLevel.WARNING,
                    f"No healthy instances found for {service_name}",
                    {
                        "node_id": str(self.node_id),
                        "target_service": service_name
                    }
                )
                return None

            # Select random healthy instance (simple load balancing)
            import random
            service = random.choice(services)

            # Build endpoint URL
            address = service['Service']['Address']
            port = service['Service']['Port']
            endpoint = f"http://{address}:{port}"

            # Cache the endpoint
            self._service_cache[cache_key] = {
                'endpoint': endpoint,
                'timestamp': datetime.now(UTC)
            }

            emit_log_event(
                LogLevel.INFO,
                f"Discovered service {service_name}",
                {
                    "node_id": str(self.node_id),
                    "target_service": service_name,
                    "endpoint": endpoint,
                    "instance_id": service['Service']['ID']
                }
            )

            return endpoint

        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                f"Service discovery failed for {service_name}: {e}",
                {
                    "node_id": str(self.node_id),
                    "target_service": service_name,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            return None
'''

CONSUL_DEREGISTRATION_TEMPLATE = '''
    async def _deregister_from_consul(self) -> None:
        """
        Deregister this node from Consul service discovery.

        Called during node shutdown to cleanly remove service registration.
        Gracefully handles errors to prevent shutdown delays.
        """
        if not hasattr(self, '_consul_service_id'):
            # Not registered, nothing to do
            return

        if not self.container.consul_client:
            emit_log_event(
                LogLevel.WARNING,
                "Consul client not available, cannot deregister",
                {
                    "node_id": str(self.node_id),
                    "service_id": self._consul_service_id
                }
            )
            return

        try:
            await self.container.consul_client.agent.service.deregister(
                self._consul_service_id
            )

            emit_log_event(
                LogLevel.INFO,
                f"Successfully deregistered from Consul: {self._consul_service_id}",
                {
                    "node_id": str(self.node_id),
                    "service_id": self._consul_service_id
                }
            )

            # Clear stored service ID
            delattr(self, '_consul_service_id')

        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Consul deregistration failed: {e}",
                {
                    "node_id": str(self.node_id),
                    "service_id": self._consul_service_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            # Don't raise - allow shutdown to continue
'''


@dataclass
class ConsulRegistrationConfig:
    """Configuration for Consul service registration."""

    node_type: str  # effect, compute, reducer, orchestrator
    service_name: str  # e.g., "postgres_crud"
    port: int  # Service port (e.g., 8000)
    health_endpoint: str = "/health"  # Health check path
    version: str = "1.0.0"  # Service version
    domain: str = "default"  # Service domain/namespace


class ConsulPatternGenerator:
    """
    Generator for Consul integration patterns.

    Produces production-ready code for service registration, discovery,
    and deregistration with ONEX v2.0 compliance.

    Features:
    - Graceful degradation when Consul unavailable
    - Structured logging with emit_log_event
    - ModelContainer integration
    - Health-aware service discovery
    - Endpoint caching (5min TTL)
    - Proper async/await patterns
    """

    def __init__(self):
        """Initialize Consul pattern generator."""
        self._generated_patterns = []

    def generate_registration(
        self,
        node_type: str,
        service_name: str,
        port: int,
        health_endpoint: str = "/health",
        version: str = "1.0.0",
        domain: str = "default",
    ) -> str:
        """
        Generate Consul service registration code.

        Args:
            node_type: Node type (effect, compute, reducer, orchestrator)
            service_name: Service name (e.g., "postgres_crud")
            port: Service port (e.g., 8000)
            health_endpoint: Health check path (default: "/health")
            version: Service version (default: "1.0.0")
            domain: Service domain/namespace (default: "default")

        Returns:
            Generated Python code for Consul registration

        Example:
            >>> generator = ConsulPatternGenerator()
            >>> code = generator.generate_registration(
            ...     node_type="effect",
            ...     service_name="postgres_crud",
            ...     port=8000
            ... )
        """
        # Input validation
        if not node_type or not isinstance(node_type, str):
            raise ValueError(
                f"node_type must be a non-empty string, got: {node_type!r}. "
                f"Valid options: 'effect', 'compute', 'reducer', 'orchestrator'"
            )

        VALID_NODE_TYPES = {"effect", "compute", "reducer", "orchestrator"}
        if node_type.lower() not in VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node_type: {node_type!r}. "
                f"Valid options: {', '.join(sorted(VALID_NODE_TYPES))}"
            )

        if not service_name or not isinstance(service_name, str):
            raise ValueError(
                f"service_name must be a non-empty string, got: {service_name!r}. "
                f"Valid examples: 'postgres_crud', 'kafka_consumer', 'api_gateway'"
            )

        if not isinstance(port, int):
            raise TypeError(
                f"port must be an integer, got: {type(port).__name__}. "
                f"Valid examples: 8000, 8080, 9000"
            )

        if not (1 <= port <= 65535):
            raise ValueError(
                f"port must be between 1 and 65535, got: {port}. "
                f"Valid examples: 8000, 8080, 9000"
            )

        if not isinstance(health_endpoint, str):
            raise TypeError(
                f"health_endpoint must be a string, got: {type(health_endpoint).__name__}. "
                f"Valid examples: '/health', '/api/health', '/status'"
            )

        if not health_endpoint.startswith("/"):
            raise ValueError(
                f"health_endpoint must start with '/', got: {health_endpoint!r}. "
                f"Valid examples: '/health', '/api/health', '/status'"
            )

        code = CONSUL_REGISTRATION_TEMPLATE.replace("{{ service_name }}", service_name)
        code = code.replace("{{ port }}", str(port))
        code = code.replace("{{ health_endpoint }}", health_endpoint)
        code = code.replace("{{ node_type }}", node_type)
        code = code.replace("{{ version }}", version)
        code = code.replace("{{ domain }}", domain)

        self._generated_patterns.append(
            {"type": "registration", "service_name": service_name, "port": port}
        )

        return code

    def generate_discovery(self) -> str:
        """
        Generate Consul service discovery code.

        Returns:
            Generated Python code for service discovery

        Example:
            >>> generator = ConsulPatternGenerator()
            >>> code = generator.generate_discovery()
        """
        self._generated_patterns.append({"type": "discovery"})

        return CONSUL_DISCOVERY_TEMPLATE

    def generate_deregistration(self) -> str:
        """
        Generate Consul service deregistration code.

        Returns:
            Generated Python code for deregistration

        Example:
            >>> generator = ConsulPatternGenerator()
            >>> code = generator.generate_deregistration()
        """
        self._generated_patterns.append({"type": "deregistration"})

        return CONSUL_DEREGISTRATION_TEMPLATE

    def generate_all_patterns(
        self,
        node_type: str,
        service_name: str,
        port: int,
        health_endpoint: str = "/health",
        version: str = "1.0.0",
        domain: str = "default",
    ) -> dict[str, str]:
        """
        Generate all Consul integration patterns at once.

        Args:
            node_type: Node type (effect, compute, reducer, orchestrator)
            service_name: Service name (e.g., "postgres_crud")
            port: Service port (e.g., 8000)
            health_endpoint: Health check path (default: "/health")
            version: Service version (default: "1.0.0")
            domain: Service domain/namespace (default: "default")

        Returns:
            Dictionary with:
            - registration: Registration code
            - discovery: Discovery code
            - deregistration: Deregistration code

        Example:
            >>> generator = ConsulPatternGenerator()
            >>> patterns = generator.generate_all_patterns(
            ...     node_type="effect",
            ...     service_name="postgres_crud",
            ...     port=8000
            ... )
            >>> print(patterns['registration'])
        """
        # Input validation (reuse validation from generate_registration)
        if not node_type or not isinstance(node_type, str):
            raise ValueError(
                f"node_type must be a non-empty string, got: {node_type!r}. "
                f"Valid options: 'effect', 'compute', 'reducer', 'orchestrator'"
            )

        VALID_NODE_TYPES = {"effect", "compute", "reducer", "orchestrator"}
        if node_type.lower() not in VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node_type: {node_type!r}. "
                f"Valid options: {', '.join(sorted(VALID_NODE_TYPES))}"
            )

        if not service_name or not isinstance(service_name, str):
            raise ValueError(
                f"service_name must be a non-empty string, got: {service_name!r}. "
                f"Valid examples: 'postgres_crud', 'kafka_consumer', 'api_gateway'"
            )

        if not isinstance(port, int) or not (1 <= port <= 65535):
            raise ValueError(
                f"port must be an integer between 1 and 65535, got: {port}. "
                f"Valid examples: 8000, 8080, 9000"
            )

        return {
            "registration": self.generate_registration(
                node_type=node_type,
                service_name=service_name,
                port=port,
                health_endpoint=health_endpoint,
                version=version,
                domain=domain,
            ),
            "discovery": self.generate_discovery(),
            "deregistration": self.generate_deregistration(),
        }

    def get_required_imports(self) -> list[str]:
        """
        Get required imports for generated Consul code.

        Returns:
            List of import statements
        """
        return [
            "from datetime import UTC, datetime",
            "from typing import Optional",
            "from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel",
            "from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event",
        ]

    def get_generated_patterns(self) -> list[dict]:
        """
        Get list of generated patterns for debugging.

        Returns:
            List of pattern metadata dicts
        """
        return self._generated_patterns.copy()


# Convenience functions for quick generation
def generate_consul_registration(
    node_type: str,
    service_name: str,
    port: int,
    health_endpoint: str = "/health",
    version: str = "1.0.0",
    domain: str = "default",
) -> str:
    """
    Quick generate Consul service registration code.

    Args:
        node_type: Node type (effect, compute, reducer, orchestrator)
        service_name: Service name (e.g., "postgres_crud")
        port: Service port (e.g., 8000)
        health_endpoint: Health check path (default: "/health")
        version: Service version (default: "1.0.0")
        domain: Service domain/namespace (default: "default")

    Returns:
        Generated Python code for Consul registration

    Example:
        >>> code = generate_consul_registration(
        ...     node_type="effect",
        ...     service_name="postgres_crud",
        ...     port=8000
        ... )
    """
    generator = ConsulPatternGenerator()
    return generator.generate_registration(
        node_type=node_type,
        service_name=service_name,
        port=port,
        health_endpoint=health_endpoint,
        version=version,
        domain=domain,
    )


def generate_consul_discovery(target_service: str = "target_service") -> str:
    """
    Quick generate Consul service discovery code.

    Returns:
        Generated Python code for service discovery

    Example:
        >>> code = generate_consul_discovery()
    """
    generator = ConsulPatternGenerator()
    return generator.generate_discovery()


def generate_consul_deregistration() -> str:
    """
    Quick generate Consul service deregistration code.

    Returns:
        Generated Python code for deregistration

    Example:
        >>> code = generate_consul_deregistration()
    """
    generator = ConsulPatternGenerator()
    return generator.generate_deregistration()


# Example usage for documentation
if __name__ == "__main__":
    # Example 1: Generate all patterns
    generator = ConsulPatternGenerator()
    patterns = generator.generate_all_patterns(
        node_type="effect", service_name="postgres_crud", port=8000
    )

    print("=== REGISTRATION PATTERN ===")
    print(patterns["registration"])
    print("\n=== DISCOVERY PATTERN ===")
    print(patterns["discovery"])
    print("\n=== DEREGISTRATION PATTERN ===")
    print(patterns["deregistration"])

    # Example 2: Required imports
    print("\n=== REQUIRED IMPORTS ===")
    for imp in generator.get_required_imports():
        print(imp)

    # Example 3: Quick convenience functions
    print("\n=== QUICK GENERATION ===")
    quick_registration = generate_consul_registration(
        node_type="reducer", service_name="metrics_aggregator", port=8001
    )
    print(quick_registration[:200] + "...")
