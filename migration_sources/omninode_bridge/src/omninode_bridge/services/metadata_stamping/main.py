# === OmniNode:Tool_Metadata ===
# metadata_version: 0.1
# name: metadata_stamping_service_main
# title: MetadataStampingService Main Entry Point
# version: 0.1.0
# namespace: omninode.services.metadata
# category: service.infrastructure.stamping
# kind: service
# role: main_entry_point
# description: |
#   Main entry point for MetadataStampingService with FastAPI application
#   and service initialization for high-performance metadata stamping.
# tags: [service, metadata, blake3, hashing, stamping, fastapi]
# author: OmniNode Development Team
# license: MIT
# entrypoint: main.py
# protocols_supported: [O.N.E. v0.1]
# runtime_constraints: {sandboxed: false, privileged: false, requires_network: true, requires_gpu: false}
# dependencies: [{"name": "fastapi", "version": "^0.104.1"}, {"name": "uvicorn", "version": "^0.24.0"}, {"name": "asyncpg", "version": "^0.29.0"}]
# environment: [python>=3.11]
# === /OmniNode:Tool_Metadata ===

"""Main entry point for MetadataStampingService.

This module provides the FastAPI application and service initialization
for the metadata stamping service.
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from .api import router
from .api.router import set_service
from .config import get_settings

# Import logging configuration
from .config.logging_config import get_logger, setup_observability_logging
from .monitoring.integration import MonitoringIntegration, set_global_monitoring
from .registry.consul_client import RegistryConsulClient
from .security.middleware import ONESecurityMiddleware
from .service import MetadataStampingService

# Setup structured logging early
setup_observability_logging()

# Get structured logger
logger = get_logger("metadata_stamping.main")

# Global service instance
service_instance: MetadataStampingService = None
monitoring_instance: MonitoringIntegration = None
registry_client: RegistryConsulClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle.

    Args:
        app: FastAPI application instance

    Yields:
        Control back to FastAPI
    """
    global service_instance, monitoring_instance, registry_client

    # Startup
    logger.info("Starting MetadataStampingService...")
    settings = get_settings()

    # Logging level is configured by structured logging setup

    # Create service instance
    config = {
        "database": settings.get_database_config() if settings.db_password else None,
        "hash_generator": {
            "pool_size": settings.hash_generator_pool_size,
            "max_workers": settings.hash_generator_max_workers,
        },
        "events": settings.get_event_config(),
    }

    service_instance = MetadataStampingService(config)

    # Initialize monitoring if performance metrics enabled
    if settings.enable_performance_metrics:
        logger.info("Initializing performance monitoring...")
        monitoring_instance = MonitoringIntegration(
            enable_resource_monitoring=True,
            enable_alerts=True,
            enable_dashboard=True,
            resource_sample_interval=5.0,
        )
        set_global_monitoring(monitoring_instance)

    # Initialize service
    if await service_instance.initialize():
        logger.info("Service initialized successfully")

        # Initialize monitoring with database client
        if monitoring_instance:
            if await monitoring_instance.initialize(service_instance.db_client):
                logger.info("Performance monitoring initialized successfully")
                # Add monitoring endpoints to the app
                monitoring_router = monitoring_instance.get_router()
                if monitoring_router:
                    app.include_router(monitoring_router)
                    logger.info("Monitoring dashboard endpoints enabled")
            else:
                logger.warning(
                    "Failed to initialize performance monitoring - continuing without monitoring"
                )
                monitoring_instance = None

        # Set service for dependency injection
        set_service(service_instance)

        # Initialize registry client if enabled
        if settings.enable_registry:
            logger.info("Initializing Consul registry client...")
            registry_client = RegistryConsulClient(
                consul_host=settings.consul_host, consul_port=settings.consul_port
            )

            if settings.service_registration_enabled:
                registration_success = await registry_client.register_service(settings)
                if registration_success:
                    logger.info("Service registered with Consul successfully")
                else:
                    logger.warning("Failed to register service with Consul")
    else:
        logger.error("Failed to initialize service")
        sys.exit(1)

    yield

    # Shutdown
    logger.info("Shutting down MetadataStampingService...")

    # Cleanup registry first
    if registry_client:
        await registry_client.deregister_service()
        logger.info("Service deregistered from Consul")

    # Cleanup monitoring
    if monitoring_instance:
        await monitoring_instance.cleanup()
        logger.info("Performance monitoring shutdown complete")

    # Cleanup service
    if service_instance:
        await service_instance.cleanup()

    logger.info("Service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="MetadataStampingService",
    description="High-performance metadata stamping service with BLAKE3 hashing",
    version="0.1.0",
    lifespan=lifespan,
)

# Get settings
settings = get_settings()

# Add CORS middleware if enabled
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add O.N.E. security middleware
if settings.enable_security:
    app.add_middleware(ONESecurityMiddleware, enable_security=True)

# Include API router
app.include_router(router)

# Monitoring router will be added during startup in lifespan function

# Add Prometheus metrics endpoint if enabled
if settings.enable_prometheus_metrics:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint.

    Returns:
        Service information
    """
    return {
        "service": "MetadataStampingService",
        "version": "0.1.0",
        "status": "running",
    }


# Simple health endpoint for Docker/Kubernetes health checks
@app.get("/health")
async def simple_health():
    """Simple health check endpoint for container orchestration.

    This is a fast, lightweight endpoint that returns 200 OK if the service
    is alive and responsive. It bypasses all middleware and doesn't require
    full service initialization, making it suitable for Docker HEALTHCHECK
    and Kubernetes liveness probes.

    Returns:
        dict: Simple health status
    """
    return {"status": "ok", "service": "metadata-stamping"}


def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully.

    Args:
        signum: Signal number
        frame: Stack frame
    """
    logger.info(f"Received signal {signum}, initiating shutdown...")
    if service_instance:
        asyncio.create_task(service_instance.cleanup())
    sys.exit(0)


def main():
    """Main entry point for running the service."""
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Get settings
    settings = get_settings()

    logger.info(
        f"Starting MetadataStampingService on {settings.service_host}:{settings.service_port}"
    )

    # Run the service
    uvicorn.run(
        "omninode_bridge.services.metadata_stamping.main:app",
        host=settings.service_host,
        port=settings.service_port,
        workers=settings.service_workers if not settings.service_reload else 1,
        reload=settings.service_reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
