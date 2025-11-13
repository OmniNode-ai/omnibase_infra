#!/usr/bin/env python3
"""
Startup script for MetadataStampingService.

This script provides robust startup handling with:
- Environment validation
- Database connectivity checks
- Graceful error handling
- Health checks
"""

import asyncio
import logging
import os
import sys
from typing import Optional

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from omninode_bridge.services.metadata_stamping.config.settings import get_settings
from omninode_bridge.services.metadata_stamping.database.client import (
    MetadataStampingPostgresClient,
)
from omninode_bridge.services.metadata_stamping.service import MetadataStampingService

logger = logging.getLogger(__name__)


async def validate_environment() -> bool:
    """Validate required environment variables and configuration.

    Returns:
        True if environment is valid, False otherwise
    """
    try:
        settings = get_settings()

        # Check required environment variables
        required_vars = [
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
        ]

        missing_vars = []
        for var in required_vars:
            if not getattr(settings, var.lower(), None):
                missing_vars.append(var)

        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False

        logger.info("Environment validation passed")
        return True

    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False


async def check_database_connectivity(
    max_retries: int = 5, retry_delay: int = 5
) -> bool:
    """Check database connectivity with retries.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        True if database is accessible, False otherwise
    """
    for attempt in range(max_retries):
        try:
            settings = get_settings()
            client = MetadataStampingPostgresClient(settings.get_database_config())

            # Try to establish connection
            if await client.initialize():
                logger.info("Database connectivity check passed")
                await client.cleanup()
                return True
            else:
                logger.warning(
                    f"Database connectivity check failed (attempt {attempt + 1}/{max_retries})"
                )

        except Exception as e:
            logger.warning(
                f"Database connectivity check failed (attempt {attempt + 1}/{max_retries}): {e}"
            )

        if attempt < max_retries - 1:
            logger.info(f"Retrying database connection in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)

    logger.error("Database connectivity check failed after all retries")
    return False


async def initialize_service() -> Optional[MetadataStampingService]:
    """Initialize the metadata stamping service.

    Returns:
        Initialized service instance or None if initialization failed
    """
    try:
        settings = get_settings()

        # Create service configuration
        config = {
            "database": settings.get_database_config(),
            "hash_generator": {
                "pool_size": settings.hash_generator_pool_size,
                "max_workers": getattr(settings, "hash_generator_max_workers", 4),
            },
            "events": (
                settings.get_event_config()
                if hasattr(settings, "get_event_config")
                else {}
            ),
        }

        service = MetadataStampingService(config)

        if await service.initialize():
            logger.info("Service initialization successful")
            return service
        else:
            logger.error("Service initialization failed")
            return None

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return None


async def startup_checks() -> bool:
    """Run all startup checks.

    Returns:
        True if all checks pass, False otherwise
    """
    logger.info("Starting MetadataStampingService startup checks...")

    # Environment validation
    if not await validate_environment():
        return False

    # Database connectivity
    if not await check_database_connectivity():
        return False

    logger.info("All startup checks passed")
    return True


def setup_logging():
    """Setup logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


async def main():
    """Main startup function."""
    setup_logging()

    logger.info("Starting MetadataStampingService...")

    # Run startup checks
    if not await startup_checks():
        logger.error("Startup checks failed, exiting...")
        sys.exit(1)

    # Initialize service
    service = await initialize_service()
    if not service:
        logger.error("Service initialization failed, exiting...")
        sys.exit(1)

    logger.info("MetadataStampingService startup completed successfully")

    # Start the main application
    from .main import main as run_main

    run_main()


if __name__ == "__main__":
    asyncio.run(main())
