"""
Workflow Coordinator Service - OmniNode Bridge (Refactored)
===========================================================

This module provides the main entry point for the workflow coordinator service
using the new modular architecture.

The large monolithic workflow_coordinator.py has been broken down into:
- workflow/enums.py - Status and type enums
- workflow/models.py - Pydantic data models
- workflow/events.py - Kafka event definitions
- workflow/execution.py - WorkflowExecution class
- workflow/coordinator.py - WorkflowCoordinator class
- workflow/app.py - FastAPI application setup
"""

import asyncio
import logging
import os
from typing import Any

from fastapi import FastAPI

# Re-export commonly used classes for backward compatibility
from ..workflow.app import create_workflow_app
from ..workflow.coordinator import WorkflowCoordinator

# Re-export KafkaClient for backward compatibility with tests

# Re-export PostgresClient for backward compatibility with tests

# Re-export smart_responder_client for backward compatibility with tests

# Configure logging
logger = logging.getLogger(__name__)

# Global coordinator instance
workflow_coordinator: WorkflowCoordinator | None = None


async def start_workflow_coordinator(
    config: dict[str, Any] = None,
) -> WorkflowCoordinator:
    """
    Start the workflow coordinator service.

    Args:
        config: Optional configuration dictionary

    Returns:
        WorkflowCoordinator: The initialized coordinator instance
    """
    global workflow_coordinator

    logger.info("Starting workflow coordinator service...")

    try:
        # Create and initialize coordinator
        workflow_coordinator = WorkflowCoordinator(config)
        await workflow_coordinator.initialize()

        logger.info("Workflow coordinator service started successfully")
        return workflow_coordinator

    except Exception as e:
        logger.error(
            "Failed to start workflow coordinator service",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise


def create_app() -> FastAPI:
    """
    Create FastAPI application for workflow coordinator.

    This function handles automatic initialization if needed to support
    uvicorn factory pattern in containerized environments.

    Returns:
        FastAPI: The configured application
    """
    global workflow_coordinator

    # Auto-initialize if not already initialized (for container startup)
    if not workflow_coordinator:
        logger.info("Auto-initializing workflow coordinator for factory pattern")

        # Get configuration from environment variables
        config = {
            "host": os.getenv("WORKFLOW_COORDINATOR_HOST", "127.0.0.1"),
            "port": int(os.getenv("WORKFLOW_COORDINATOR_PORT", "8006")),
        }

        # Create coordinator instance (clients will be initialized on app startup)
        workflow_coordinator = WorkflowCoordinator(config)

    # Create the FastAPI app
    app = create_workflow_app(workflow_coordinator)

    @app.on_event("startup")
    async def startup_event():
        """Initialize workflow coordinator components on application startup."""
        logger.info("Initializing workflow coordinator on application startup")
        try:
            await workflow_coordinator.initialize()
            logger.info("Workflow coordinator initialization completed")
        except Exception as e:
            logger.error(f"Failed to initialize workflow coordinator: {e}")
            raise

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup workflow coordinator on application shutdown."""
        logger.info("Shutting down workflow coordinator")
        try:
            if workflow_coordinator:
                await workflow_coordinator.cleanup()
        except Exception as e:
            logger.error(f"Error during workflow coordinator cleanup: {e}")

    return app


async def main():
    """Main entry point for running the workflow coordinator service."""
    try:
        # Get configuration
        config = {
            "host": os.getenv("WORKFLOW_COORDINATOR_HOST", "127.0.0.1"),
            "port": int(os.getenv("WORKFLOW_COORDINATOR_PORT", "8006")),
        }

        # Start coordinator
        coordinator = await start_workflow_coordinator(config)

        # Create and configure app
        app = create_workflow_app(coordinator)

        # Import uvicorn for running the server
        import uvicorn

        logger.info(
            "Starting workflow coordinator HTTP server",
            extra={
                "host": config["host"],
                "port": config["port"],
            },
        )

        # Run the server
        await uvicorn.run(
            app,
            host=config["host"],
            port=config["port"],
            log_level="info",
        )

    except KeyboardInterrupt:
        logger.info("Workflow coordinator service interrupted by user")
    except Exception as e:
        logger.error(
            "Workflow coordinator service failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise
    finally:
        # Cleanup
        if "coordinator" in locals() and coordinator:
            await coordinator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
