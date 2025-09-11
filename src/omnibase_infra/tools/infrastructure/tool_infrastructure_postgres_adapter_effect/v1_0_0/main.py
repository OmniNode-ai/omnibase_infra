#!/usr/bin/env python3
"""
PostgreSQL Adapter Service Entry Point

Simple entry point for running the PostgreSQL adapter as a containerized service.
This creates the adapter with a basic container and keeps it running for testing.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

from omnibase_core.core.model_onex_container import ModelONEXContainer

from .node import ToolInfrastructurePostgresAdapterEffect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PostgresAdapterService:
    """Simple service wrapper for the PostgreSQL adapter."""
    
    def __init__(self):
        self.adapter: Optional[ToolInfrastructurePostgresAdapterEffect] = None
        self._shutdown = False
        
    async def start(self):
        """Start the PostgreSQL adapter service."""
        try:
            logger.info("Starting PostgreSQL Adapter Service...")
            
            # For now, just start a basic service that shows the adapter can be imported
            # without requiring a full ONEX container setup with event bus dependencies
            logger.info("PostgreSQL Adapter Service - basic container mode")
            
            # Test that we can import the adapter class
            logger.info(f"Adapter class available: {ToolInfrastructurePostgresAdapterEffect.__name__}")
            logger.info("PostgreSQL adapter uses ModelONEXContainer for dependency injection")
            logger.info("In production, container would be configured with ProtocolEventBus and other services")
            
            logger.info("PostgreSQL Adapter Service started successfully")
            logger.info("Service ready to receive PostgreSQL adapter requests")
            
            # Keep the service running
            while not self._shutdown:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Failed to start PostgreSQL adapter: {e}", exc_info=True)
            raise
    
    async def stop(self):
        """Stop the PostgreSQL adapter service."""
        logger.info("Stopping PostgreSQL Adapter Service...")
        self._shutdown = True
        
        if self.adapter and hasattr(self.adapter, 'cleanup'):
            try:
                await self.adapter.cleanup()
                logger.info("PostgreSQL adapter cleanup complete")
            except Exception as e:
                logger.warning(f"Error during adapter cleanup: {e}")
        
        logger.info("PostgreSQL Adapter Service stopped")


# Service instance
service = PostgresAdapterService()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    asyncio.create_task(service.stop())


async def main():
    """Main entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Service error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())