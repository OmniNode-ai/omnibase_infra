"""
Background cleanup task for action deduplication service.

Periodically removes expired deduplication entries to maintain database
performance and prevent unbounded growth of the action_dedup_log table.

ONEX v2.0 Compliance:
- Suffix-based naming: DedupCleanupTask
- Graceful shutdown handling
- Comprehensive error handling and logging
- Configurable cleanup interval

Features:
- Periodic cleanup of expired dedup entries
- Configurable cleanup interval (default: 1 hour)
- Graceful shutdown support
- Error recovery and retry
- Metrics reporting

Example:
    >>> from omninode_bridge.services.postgres_client import PostgresClient
    >>> from omninode_bridge.services.action_dedup import ActionDedupService
    >>>
    >>> postgres_client = PostgresClient()
    >>> await postgres_client.connect()
    >>>
    >>> dedup_service = ActionDedupService(postgres_client)
    >>>
    >>> # Start cleanup task
    >>> task = DedupCleanupTask(dedup_service, cleanup_interval=3600)
    >>> await task.start()
    >>>
    >>> # Later, shutdown gracefully
    >>> await task.stop()
"""

import asyncio
import logging
from typing import Optional

from ..services.action_dedup import ActionDedupService

logger = logging.getLogger(__name__)


class DedupCleanupTask:
    """
    Background task for periodic cleanup of expired deduplication entries.

    This task runs in the background and periodically calls cleanup_expired()
    on the ActionDedupService to remove expired entries from the database.

    Features:
    - Configurable cleanup interval (default: 1 hour)
    - Graceful shutdown support
    - Error recovery with exponential backoff
    - Metrics reporting

    Performance:
    - Typically completes in <100ms for thousands of entries
    - Does not block application operations
    - Safe to run concurrently from multiple instances

    Thread Safety:
    - This task is designed for asyncio and is not thread-safe
    - Use separate instances for different event loops
    """

    def __init__(
        self,
        dedup_service: ActionDedupService,
        cleanup_interval: int = 3600,
        max_retry_delay: int = 300,
    ):
        """
        Initialize the cleanup task.

        Args:
            dedup_service: ActionDedupService instance to clean up
            cleanup_interval: Interval between cleanups in seconds (default: 1 hour)
            max_retry_delay: Maximum retry delay on error in seconds (default: 5 min)
        """
        self.dedup_service = dedup_service
        self.cleanup_interval = cleanup_interval
        self.max_retry_delay = max_retry_delay

        self._task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._running = False

    async def start(self) -> None:
        """
        Start the background cleanup task.

        This method starts a background asyncio task that periodically
        cleans up expired deduplication entries.

        Example:
            >>> task = DedupCleanupTask(dedup_service)
            >>> await task.start()
        """
        if self._running:
            logger.warning("Cleanup task is already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._run_cleanup_loop())

        logger.info(
            f"Started dedup cleanup task with {self.cleanup_interval}s interval"
        )

    async def stop(self) -> None:
        """
        Stop the background cleanup task gracefully.

        This method signals the cleanup task to stop and waits for it
        to complete its current iteration.

        Example:
            >>> await task.stop()
        """
        if not self._running:
            logger.warning("Cleanup task is not running")
            return

        logger.info("Stopping dedup cleanup task...")
        self._shutdown_event.set()

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=30.0)
            except TimeoutError:
                logger.warning("Cleanup task did not stop gracefully, cancelling...")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        self._running = False
        logger.info("Dedup cleanup task stopped")

    async def _run_cleanup_loop(self) -> None:
        """
        Main cleanup loop that runs periodically.

        This method runs in the background and performs cleanup at regular
        intervals. It handles errors and implements exponential backoff
        on failures.
        """
        retry_delay = 0
        consecutive_errors = 0

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Perform cleanup
                    deleted_count = await self.dedup_service.cleanup_expired()

                    # Log results
                    if deleted_count > 0:
                        logger.info(
                            f"Dedup cleanup completed: deleted {deleted_count} expired entries"
                        )
                    else:
                        logger.debug("Dedup cleanup completed: no expired entries")

                    # Reset error tracking on success
                    consecutive_errors = 0
                    retry_delay = 0

                    # Wait for next cleanup interval or shutdown
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(), timeout=self.cleanup_interval
                        )
                        # If we get here, shutdown was signaled
                        break
                    except TimeoutError:
                        # Normal timeout, continue to next iteration
                        continue

                except asyncio.CancelledError:
                    # Task was cancelled, exit gracefully
                    logger.info("Dedup cleanup task cancelled")
                    break

                except Exception as e:
                    consecutive_errors += 1
                    retry_delay = min(
                        2**consecutive_errors, self.max_retry_delay
                    )  # Exponential backoff

                    logger.error(
                        f"Error during dedup cleanup (attempt {consecutive_errors}): {e}",
                        exc_info=True,
                    )
                    logger.info(f"Retrying cleanup in {retry_delay} seconds...")

                    # Wait before retry or shutdown
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(), timeout=retry_delay
                        )
                        # If we get here, shutdown was signaled
                        break
                    except TimeoutError:
                        # Normal timeout, continue to next iteration
                        continue

        except Exception as e:
            logger.critical(f"Fatal error in dedup cleanup loop: {e}", exc_info=True)
            raise

        finally:
            logger.info("Dedup cleanup loop exited")

    @property
    def is_running(self) -> bool:
        """Check if the cleanup task is currently running."""
        return self._running


async def periodic_cleanup(
    dedup_service: ActionDedupService, interval: int = 3600
) -> None:
    """
    Simple periodic cleanup function for standalone usage.

    This is a convenience function for running cleanup in a simple loop
    without the full DedupCleanupTask class.

    Args:
        dedup_service: ActionDedupService instance to clean up
        interval: Interval between cleanups in seconds (default: 1 hour)

    Example:
        >>> # In main application startup
        >>> asyncio.create_task(periodic_cleanup(dedup_service, interval=3600))
    """
    logger.info(f"Starting periodic dedup cleanup with {interval}s interval")

    while True:
        try:
            deleted_count = await dedup_service.cleanup_expired()

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired dedup entries")

            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info("Periodic cleanup cancelled")
            break

        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}", exc_info=True)
            # Wait before retry
            await asyncio.sleep(min(60, interval / 10))
