"""Graceful shutdown handler for OmniNode Bridge services."""

import asyncio
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class GracefulShutdownHandler:
    """
    Handles graceful shutdown of OmniNode Bridge services.

    Manages cleanup of resources, ongoing tasks, and external connections
    to ensure zero-downtime deployments and proper resource cleanup.
    """

    def __init__(self, service_name: str, shutdown_timeout: int = 30):
        """
        Initialize graceful shutdown handler.

        Args:
            service_name: Name of the service
            shutdown_timeout: Maximum time to wait for graceful shutdown
        """
        self.service_name = service_name
        self.shutdown_timeout = shutdown_timeout
        self.shutdown_requested = False
        self.shutdown_event = threading.Event()
        self.cleanup_tasks: list[Callable] = []
        self.async_cleanup_tasks: list[Callable] = []
        self.resource_managers: dict[str, Any] = {}
        self.graceful_shutdown_complete = False

        # Register signal handlers
        self._register_signal_handlers()

        logger.info(f"Graceful shutdown handler initialized for {service_name}")

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.info(
                f"Received {signal_name} signal, initiating graceful shutdown...",
            )
            self.initiate_shutdown()

        # Register handlers for common termination signals
        signal.signal(signal.SIGTERM, signal_handler)  # Kubernetes termination
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C

        # Handle SIGUSR1 for custom graceful restart
        signal.signal(signal.SIGUSR1, signal_handler)

    def register_cleanup_task(self, cleanup_func: Callable, description: str = ""):
        """
        Register a synchronous cleanup task.

        Args:
            cleanup_func: Function to call during shutdown
            description: Description of the cleanup task
        """
        self.cleanup_tasks.append(
            {"func": cleanup_func, "description": description or cleanup_func.__name__},
        )
        logger.debug(f"Registered cleanup task: {description or cleanup_func.__name__}")

    def register_async_cleanup_task(
        self,
        cleanup_func: Callable,
        description: str = "",
    ):
        """
        Register an asynchronous cleanup task.

        Args:
            cleanup_func: Async function to call during shutdown
            description: Description of the cleanup task
        """
        self.async_cleanup_tasks.append(
            {"func": cleanup_func, "description": description or cleanup_func.__name__},
        )
        logger.debug(
            f"Registered async cleanup task: {description or cleanup_func.__name__}",
        )

    def register_resource_manager(self, name: str, manager: Any):
        """
        Register a resource manager for cleanup.

        Args:
            name: Name of the resource manager
            manager: Object with cleanup methods
        """
        self.resource_managers[name] = manager
        logger.debug(f"Registered resource manager: {name}")

    def initiate_shutdown(self):
        """Initiate graceful shutdown process."""
        if self.shutdown_requested:
            logger.warning("Shutdown already in progress...")
            return

        self.shutdown_requested = True
        logger.info(f"🛑 Initiating graceful shutdown for {self.service_name}")

        # Set shutdown event for any waiting threads
        self.shutdown_event.set()

        # Start shutdown process in separate thread to avoid blocking signal handler
        shutdown_thread = threading.Thread(target=self._execute_shutdown, daemon=True)
        shutdown_thread.start()

    def _execute_shutdown(self):
        """Execute the shutdown process."""
        start_time = time.time()

        try:
            # Step 1: Stop accepting new requests
            logger.info("Step 1: Stopping acceptance of new requests...")
            self._stop_accepting_requests()

            # Step 2: Wait for ongoing requests to complete
            logger.info("Step 2: Waiting for ongoing requests to complete...")
            self._wait_for_ongoing_requests()

            # Step 3: Clean up resources
            logger.info("Step 3: Cleaning up resources...")
            self._cleanup_resources()

            # Step 4: Close external connections
            logger.info("Step 4: Closing external connections...")
            self._close_external_connections()

            # Step 5: Final cleanup
            logger.info("Step 5: Performing final cleanup...")
            self._final_cleanup()

            self.graceful_shutdown_complete = True
            elapsed_time = time.time() - start_time

            logger.info(f"✅ Graceful shutdown completed in {elapsed_time:.2f}s")

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"❌ Error during graceful shutdown: {e!s}")
            logger.error(f"Shutdown time: {elapsed_time:.2f}s")

        finally:
            # Force exit if shutdown takes too long
            elapsed_time = time.time() - start_time
            if elapsed_time > self.shutdown_timeout:
                logger.critical(
                    f"⚠️  Graceful shutdown timeout ({self.shutdown_timeout}s), forcing exit",
                )
                sys.exit(1)
            else:
                logger.info("Exiting gracefully...")
                sys.exit(0)

    def _stop_accepting_requests(self):
        """Stop accepting new requests."""
        # Mark service as unavailable for health checks
        for name, manager in self.resource_managers.items():
            if hasattr(manager, "stop_accepting_requests"):
                try:
                    manager.stop_accepting_requests()
                    logger.info(f"Stopped accepting requests for {name}")
                except Exception as e:
                    logger.error(f"Error stopping requests for {name}: {e!s}")

    def _wait_for_ongoing_requests(self):
        """Wait for ongoing requests to complete."""
        max_wait_time = min(
            15,
            self.shutdown_timeout // 2,
        )  # Max 15 seconds or half timeout
        wait_start = time.time()

        while time.time() - wait_start < max_wait_time:
            active_requests = 0

            # Check active requests in resource managers
            for name, manager in self.resource_managers.items():
                if hasattr(manager, "get_active_request_count"):
                    try:
                        count = manager.get_active_request_count()
                        active_requests += count
                    except Exception as e:
                        logger.warning(
                            f"Could not get active request count for {name}: {e!s}",
                        )

            if active_requests == 0:
                logger.info("All ongoing requests completed")
                break

            logger.info(f"Waiting for {active_requests} active requests to complete...")
            time.sleep(1)

        if active_requests > 0:
            logger.warning(
                f"Proceeding with shutdown despite {active_requests} active requests",
            )

    def _cleanup_resources(self):
        """Clean up application resources."""
        # Execute synchronous cleanup tasks
        for task in self.cleanup_tasks:
            try:
                logger.info(f"Executing cleanup task: {task['description']}")
                task["func"]()
            except Exception as e:
                logger.error(f"Error in cleanup task {task['description']}: {e!s}")

        # Execute asynchronous cleanup tasks
        if self.async_cleanup_tasks:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def run_async_cleanup():
                    for task in self.async_cleanup_tasks:
                        try:
                            logger.info(
                                f"Executing async cleanup task: {task['description']}",
                            )
                            await task["func"]()
                        except Exception as e:
                            logger.error(
                                f"Error in async cleanup task {task['description']}: {e!s}",
                            )

                loop.run_until_complete(run_async_cleanup())
                loop.close()

            except Exception as e:
                logger.error(f"Error running async cleanup tasks: {e!s}")

    def _close_external_connections(self):
        """Close external connections."""
        for name, manager in self.resource_managers.items():
            try:
                # Database connections
                if hasattr(manager, "disconnect"):
                    logger.info(f"Disconnecting {name}...")
                    manager.disconnect()

                # Close method for general resources
                elif hasattr(manager, "close"):
                    logger.info(f"Closing {name}...")
                    manager.close()

                # Cleanup method for general cleanup
                elif hasattr(manager, "cleanup"):
                    logger.info(f"Cleaning up {name}...")
                    manager.cleanup()

            except Exception as e:
                logger.error(f"Error closing {name}: {e!s}")

    def _final_cleanup(self):
        """Perform final cleanup operations."""
        try:
            # Cleanup any remaining asyncio tasks
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    pending_tasks = [
                        task for task in asyncio.all_tasks(loop) if not task.done()
                    ]
                    if pending_tasks:
                        logger.info(
                            f"Cancelling {len(pending_tasks)} pending asyncio tasks...",
                        )
                        for task in pending_tasks:
                            task.cancel()
            except RuntimeError:
                # No event loop running
                pass

            # Final resource cleanup
            logger.info("Final resource cleanup completed")

        except Exception as e:
            logger.error(f"Error in final cleanup: {e!s}")

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_requested

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        """
        Wait for shutdown to be requested.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if shutdown was requested, False if timeout
        """
        return self.shutdown_event.wait(timeout)

    @property
    def is_shutting_down(self) -> bool:
        """Check if service is currently shutting down."""
        return self.shutdown_requested and not self.graceful_shutdown_complete


# Convenience function for FastAPI applications
def setup_fastapi_graceful_shutdown(app, service_name: str) -> GracefulShutdownHandler:
    """
    Set up graceful shutdown for FastAPI applications.

    Args:
        app: FastAPI application instance
        service_name: Name of the service

    Returns:
        GracefulShutdownHandler instance
    """
    shutdown_handler = GracefulShutdownHandler(service_name)

    # Add health check endpoint that returns 503 during shutdown
    @app.get("/health")
    async def health_check():
        if shutdown_handler.is_shutting_down:
            return {"status": "shutting_down", "code": 503}
        return {"status": "healthy", "service": service_name}

    # Register FastAPI shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        if not shutdown_handler.shutdown_requested:
            shutdown_handler.initiate_shutdown()

    return shutdown_handler


# Context manager for graceful shutdown
class GracefulShutdownContext:
    """Context manager for graceful shutdown handling."""

    def __init__(self, service_name: str, shutdown_timeout: int = 30):
        self.shutdown_handler = GracefulShutdownHandler(service_name, shutdown_timeout)

    def __enter__(self):
        return self.shutdown_handler

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.shutdown_handler.graceful_shutdown_complete:
            self.shutdown_handler.initiate_shutdown()
        return False
