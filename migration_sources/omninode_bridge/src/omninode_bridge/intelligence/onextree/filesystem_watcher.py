"""
Filesystem watcher for automatic tree updates.

Uses watchdog library to detect filesystem changes and trigger regeneration.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import structlog
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = structlog.get_logger(__name__)


class OnexTreeWatcher(FileSystemEventHandler):
    """
    Watches filesystem for changes and triggers tree regeneration.

    Update latency target: < 1 second

    Implements debouncing to batch rapid filesystem changes and avoid
    excessive regeneration.
    """

    def __init__(
        self,
        project_root: Path,
        on_change_callback: Callable[[], None],
        debounce_seconds: float = 0.5,
    ):
        """
        Initialize filesystem watcher.

        Args:
            project_root: Root directory to watch
            on_change_callback: Async function to call on changes
            debounce_seconds: Debounce period to batch changes
        """
        super().__init__()
        self.project_root = Path(project_root)
        self.on_change_callback = on_change_callback
        self.debounce_seconds = debounce_seconds

        self._observer: Optional[Observer] = None
        self._debounce_task: Optional[asyncio.Task] = None
        self._pending_changes = False
        self._is_running = False

    def start(self) -> None:
        """
        Start watching filesystem.

        Creates observer and schedules recursive watching of project_root.
        """
        if self._is_running:
            logger.info("Watcher already running")
            return

        self._observer = Observer()
        self._observer.schedule(self, str(self.project_root), recursive=True)
        self._observer.start()
        self._is_running = True
        logger.info("Started watching filesystem", project_root=str(self.project_root))

    def stop(self) -> None:
        """
        Stop watching filesystem.

        Gracefully stops observer and waits for shutdown.
        """
        if not self._is_running:
            return

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        # Cancel pending debounce task
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()

        self._is_running = False
        logger.info("Stopped watching filesystem")

    def on_any_event(self, event: FileSystemEvent) -> None:
        """
        Handle any filesystem event.

        Args:
            event: Filesystem event from watchdog
        """
        # Ignore directory events (we'll catch file events)
        if event.is_directory:
            return

        # Ignore events for paths we should exclude
        if self._should_ignore_event(event):
            return

        # Mark changes pending and trigger debounced update
        self._pending_changes = True
        try:
            # Try to create task in running loop
            asyncio.create_task(self._debounced_update())
        except RuntimeError:
            # No running event loop - schedule for later
            # This can happen during test teardown or when watcher runs in thread
            pass

    async def _debounced_update(self) -> None:
        """
        Debounced tree update to batch rapid changes.

        Waits for debounce period, then calls the update callback if
        changes are still pending.
        """
        # Cancel existing debounce task
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()

        # Create new debounce task
        self._debounce_task = asyncio.create_task(self._execute_debounced_update())

    async def _execute_debounced_update(self) -> None:
        """Execute the actual debounced update after waiting."""
        try:
            # Wait for debounce period
            await asyncio.sleep(self.debounce_seconds)

            # Execute callback if still pending
            if self._pending_changes:
                self._pending_changes = False
                await self.on_change_callback()

        except asyncio.CancelledError:
            # Task was cancelled, ignore
            pass
        except Exception as e:
            logger.error("Error in debounced update", error=str(e), exc_info=True)

    def _should_ignore_event(self, event: FileSystemEvent) -> bool:
        """
        Check if event should be ignored.

        Args:
            event: Filesystem event

        Returns:
            True if event should be ignored
        """
        ignore_patterns = [
            ".git",
            "__pycache__",
            ".DS_Store",
            ".pyc",
            ".swp",
            ".swo",
            "~",
            ".tmp",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "4913",  # vim temp files
        ]

        path_str = str(event.src_path)

        # Check if any ignore pattern is in the path
        return any(pattern in path_str for pattern in ignore_patterns)

    def is_running(self) -> bool:
        """
        Check if watcher is currently running.

        Returns:
            True if watcher is active
        """
        return self._is_running

    async def __aenter__(self):
        """
        Context manager entry - start the watcher.

        Returns:
            Self for use in context
        """
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - stop the watcher.

        Args:
            exc_type: Exception type if any
            exc_val: Exception value if any
            exc_tb: Exception traceback if any

        Returns:
            False to not suppress exceptions
        """
        self.stop()
        return False
