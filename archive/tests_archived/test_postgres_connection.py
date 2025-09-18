#!/usr/bin/env python3
"""Integration test to verify PostgreSQL connection works in Docker environment."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnibase_infra.infrastructure.postgres_connection_manager import (
    PostgresConnectionManager,
)

# Configure logging following omnibase_3 infrastructure pattern
logger = logging.getLogger(__name__)


async def test_postgres_connection():
    """Test PostgreSQL connection and basic operations."""
    logger.info("Starting PostgreSQL connection test...")

    try:
        # Create connection manager with environment configuration
        manager = PostgresConnectionManager()

        logger.info("Initializing connection manager...")
        await manager.initialize()

        logger.info("Running health check...")
        health = await manager.health_check()
        logger.info(f"Health check result: {health}")

        logger.info("Testing simple query...")
        result = await manager.execute_query("SELECT version();")
        logger.info(f"PostgreSQL version: {result}")

        logger.info("Testing infrastructure schema query...")
        result = await manager.execute_query(
            "SELECT COUNT(*) as service_count FROM infrastructure.service_registry;",
        )
        logger.info(f"Service registry entries: {result}")

        logger.info("Closing connection...")
        await manager.close()

        logger.info("✅ PostgreSQL connection test successful!")

    except Exception as e:
        logger.error(f"❌ PostgreSQL connection test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Configure logging for test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    success = asyncio.run(test_postgres_connection())
    exit(0 if success else 1)
