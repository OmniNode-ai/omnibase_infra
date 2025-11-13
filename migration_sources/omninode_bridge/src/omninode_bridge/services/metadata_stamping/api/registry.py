"""
Registry API endpoints for service discovery.

This module provides API endpoints for registry health checks
and service discovery functionality.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..registry.consul_client import RegistryConsulClient

router = APIRouter(prefix="/registry", tags=["registry"])


@router.get("/health")
async def registry_health() -> dict[str, Any]:
    """
    Check registry client health.

    Returns:
        dict: Registry health status
    """
    settings = get_settings()
    if not settings.enable_registry:
        return {"status": "disabled", "message": "Registry not enabled"}

    # Create a temporary client for health check
    registry_client = RegistryConsulClient(settings.consul_host, settings.consul_port)
    health = await registry_client.health_check()
    return health


@router.get("/discover/{service_name}")
async def discover_service(service_name: str) -> dict[str, Any]:
    """
    Discover services by name.

    Args:
        service_name: Name of service to discover

    Returns:
        dict: Discovered services

    Raises:
        HTTPException: If registry is not enabled
    """
    settings = get_settings()
    if not settings.enable_registry:
        raise HTTPException(status_code=503, detail="Registry not enabled")

    registry_client = RegistryConsulClient(settings.consul_host, settings.consul_port)
    services = await registry_client.discover_services(service_name)

    return {"services": services, "count": len(services), "service_name": service_name}


@router.get("/services")
async def list_registered_services() -> dict[str, Any]:
    """
    List all registered services.

    Returns:
        dict: List of registered services

    Raises:
        HTTPException: If registry is not enabled
    """
    settings = get_settings()
    if not settings.enable_registry:
        raise HTTPException(status_code=503, detail="Registry not enabled")

    registry_client = RegistryConsulClient(settings.consul_host, settings.consul_port)

    # Discover our own service type
    services = await registry_client.discover_services("metadata-stamping-service")

    return {
        "services": services,
        "count": len(services),
        "registry_enabled": True,
        "consul_host": settings.consul_host,
        "consul_port": settings.consul_port,
    }
