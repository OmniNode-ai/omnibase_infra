# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for watchdog-based contract file change detection with MD5 hash tracking.

Watches ``contract.yaml`` files under a configurable root directory using the
``watchdog`` filesystem event library. Changes are detected via MD5 hash
comparison (not mtime alone) to avoid false positives from touch/rsync.

Design:
    - Synchronous watchdog observer runs in a background thread.
    - A 1-second debounce window prevents duplicate events for rapid writes.
    - Hash state is maintained in-memory (``self._file_hashes``).
    - On hash change: emits ``ModelUpdateTrigger`` with ``trigger_type="contract_changed"``.
    - The ``changed_files`` list contains the relative path (from ``watch_root``) of each
      modified contract file.

Handler Purity:
    This handler does NOT publish events directly. Callers should await
    ``start()`` and then iterate ``get_pending_triggers()`` to drain pending
    trigger queue.

    The pattern for integration with the ONEX runtime:
        1. Instantiate and ``await handler.start()``
        2. Poll ``await handler.get_pending_triggers()`` in the event loop
        3. Publish each ``ModelUpdateTrigger`` to ``onex.evt.artifact.change-detected.v1``
        4. ``await handler.stop()`` on shutdown

Dependencies:
    ``watchdog`` must be installed. Import errors surface as
    ``ImportError`` with a descriptive message.

Related Tickets:
    - OMN-3940: Task 5 — Change Detector EFFECT Node
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer as WatchdogObserver

    _WATCHDOG_AVAILABLE = True
except ImportError:
    _WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object  # type: ignore[assignment,misc]
    WatchdogObserver = None  # type: ignore[assignment]
    FileSystemEvent = object  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from watchdog.observers.api import BaseObserver as _ObserverType

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
    ModelUpdateTrigger,
)

logger = logging.getLogger(__name__)

# Default glob pattern for watched contract files
_CONTRACT_GLOB = "contract.yaml"

__all__ = ["HandlerContractFileWatcher"]


def _md5_of_file(path: Path) -> str | None:
    """Return the MD5 hex digest of a file's contents, or None if unreadable."""
    try:
        digest = hashlib.md5(usedforsecurity=False)
        data = path.read_bytes()
        digest.update(data)
        return digest.hexdigest()
    except OSError:
        return None


class HandlerContractFileEvent(FileSystemEventHandler):
    """Internal watchdog event handler that enqueues changed contract paths."""

    def __init__(
        self,
        watch_root: Path,
        contract_glob: str,
        pending: list[Path],
        lock: threading.Lock,
    ) -> None:
        super().__init__()
        self._watch_root = watch_root
        self._contract_glob = contract_glob
        self._pending = pending
        self._lock = lock

    def _is_contract_file(self, src_path: str) -> bool:
        path = Path(src_path)
        return path.name == self._contract_glob

    def on_modified(self, event: object) -> None:
        if not getattr(event, "is_directory", True) and self._is_contract_file(
            str(getattr(event, "src_path", ""))
        ):
            with self._lock:
                self._pending.append(Path(str(getattr(event, "src_path", ""))))

    def on_created(self, event: object) -> None:
        if not getattr(event, "is_directory", True) and self._is_contract_file(
            str(getattr(event, "src_path", ""))
        ):
            with self._lock:
                self._pending.append(Path(str(getattr(event, "src_path", ""))))

    def on_moved(self, event: object) -> None:
        # Handle renames — treat as a creation of the dest path
        dest_path = getattr(event, "dest_path", "")
        if not getattr(event, "is_directory", True) and self._is_contract_file(
            str(dest_path)
        ):
            with self._lock:
                self._pending.append(Path(str(dest_path)))


class HandlerContractFileWatcher:
    """Watchdog-based handler for detecting contract.yaml file changes.

    Watches all ``contract.yaml`` files under ``watch_root`` (recursively).
    Uses MD5 hash tracking to detect actual content changes (not just
    filesystem event noise).

    Lifecycle:
        ``start()`` → ``get_pending_triggers()`` (poll loop) → ``stop()``

    Args:
        watch_root: Root directory to watch (e.g. ``src/omnibase_infra/nodes``).
        source_repo: Repository identifier placed in ``ModelUpdateTrigger.source_repo``.
        debounce_seconds: Seconds to wait after the last FS event before
            processing (coalesces rapid writes). Default: 1.0.
        contract_glob: Filename to watch for. Default: ``"contract.yaml"``.

    Raises:
        ImportError: If ``watchdog`` is not installed.
        FileNotFoundError: If ``watch_root`` does not exist.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role: NODE_HANDLER."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification: EFFECT (filesystem I/O)."""
        return EnumHandlerTypeCategory.EFFECT

    def __init__(
        self,
        watch_root: Path,
        source_repo: str,
        debounce_seconds: float = 1.0,
        contract_glob: str = _CONTRACT_GLOB,
    ) -> None:
        if not _WATCHDOG_AVAILABLE:
            raise ImportError(
                "watchdog is required for HandlerContractFileWatcher. "
                "Install it with: uv add 'watchdog'"
            )
        self._watch_root = watch_root
        self._source_repo = source_repo
        self._debounce_seconds = debounce_seconds
        self._contract_glob = contract_glob

        # MD5 hash table: absolute path → last known hash
        self._file_hashes: dict[Path, str] = {}
        # Pending FS event paths (written by watchdog thread, read by async loop)
        self._pending_paths: list[Path] = []
        self._pending_lock = threading.Lock()
        # Trigger queue: populated by _process_pending, consumed by get_pending_triggers
        self._trigger_queue: asyncio.Queue[ModelUpdateTrigger] = asyncio.Queue()
        # Internal state
        self._observer: _ObserverType | None = None
        self._running = False
        self._debounce_task: asyncio.Task[None] | None = None

    def _seed_hashes(self) -> None:
        """Compute initial MD5 hashes for all existing contract files."""
        for path in self._watch_root.rglob(self._contract_glob):
            h = _md5_of_file(path)
            if h is not None:
                self._file_hashes[path.resolve()] = h
        logger.debug(
            "HandlerContractFileWatcher: seeded %d contract files under %s",
            len(self._file_hashes),
            self._watch_root,
        )

    def _build_trigger(self, changed: list[Path]) -> ModelUpdateTrigger:
        """Build a ModelUpdateTrigger for the given list of changed file paths."""
        # Use relative paths from watch_root for portability
        relative_paths: list[str] = []
        for p in changed:
            try:
                relative_paths.append(str(p.relative_to(self._watch_root.resolve())))
            except ValueError:
                relative_paths.append(str(p))

        return ModelUpdateTrigger(
            trigger_id=uuid4(),
            trigger_type="contract_changed",
            source_repo=self._source_repo,
            source_ref=None,
            changed_files=relative_paths,
            ticket_ids=[],
            actor=None,
            reason=f"File watcher detected {len(changed)} contract change(s)",
            timestamp=datetime.now(tz=UTC),
        )

    def _check_for_changes(self, paths: list[Path]) -> list[Path]:
        """Filter paths to those with actual content changes (MD5 comparison).

        Updates ``self._file_hashes`` for changed files.
        """
        changed: list[Path] = []
        for path in paths:
            abs_path = path.resolve()
            new_hash = _md5_of_file(abs_path)
            if new_hash is None:
                # File deleted or unreadable — treat as change
                if abs_path in self._file_hashes:
                    del self._file_hashes[abs_path]
                    changed.append(abs_path)
                continue
            old_hash = self._file_hashes.get(abs_path)
            if old_hash != new_hash:
                self._file_hashes[abs_path] = new_hash
                changed.append(abs_path)
        return changed

    async def _debounce_and_process(self) -> None:
        """Async debounce: wait for quiet period, then process accumulated events."""
        await asyncio.sleep(self._debounce_seconds)

        with self._pending_lock:
            paths = list(self._pending_paths)
            self._pending_paths.clear()

        if not paths:
            return

        changed = self._check_for_changes(paths)
        if changed:
            trigger = self._build_trigger(changed)
            await self._trigger_queue.put(trigger)
            logger.info(
                "HandlerContractFileWatcher: emitted trigger %s for %d changed contracts",
                trigger.trigger_id,
                len(changed),
            )

    def _on_watchdog_event(self) -> None:
        """Called by the watchdog thread to schedule async debounce processing."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        # Cancel pending debounce and reschedule (sliding window)
        if self._debounce_task is not None and not self._debounce_task.done():
            self._debounce_task.cancel()
        # run_coroutine_threadsafe returns a concurrent.futures.Future,
        # not an asyncio.Task — store as Any to avoid type confusion here.
        # The debounce_task field is only used for cancellation checks via .done()/.cancel().
        future = asyncio.run_coroutine_threadsafe(self._debounce_and_process(), loop)
        self._debounce_task = None  # Can't cancel a threadsafe future from here
        _ = future  # Suppress unused-variable warning; future runs in event loop

    async def start(self) -> None:
        """Start the watchdog observer and seed initial file hashes.

        Raises:
            FileNotFoundError: If ``watch_root`` does not exist.
        """
        if not self._watch_root.exists():
            raise FileNotFoundError(
                f"HandlerContractFileWatcher: watch_root does not exist: {self._watch_root}"
            )

        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._seed_hashes()

        event_handler = HandlerContractFileEvent(
            watch_root=self._watch_root,
            contract_glob=self._contract_glob,
            pending=self._pending_paths,
            lock=self._pending_lock,
        )

        self._observer = WatchdogObserver()
        self._observer.schedule(event_handler, str(self._watch_root), recursive=True)
        self._observer.start()
        self._running = True

        logger.info(
            "HandlerContractFileWatcher: started watching %s (debounce=%.1fs)",
            self._watch_root,
            self._debounce_seconds,
        )

    async def stop(self) -> None:
        """Stop the watchdog observer and cancel any pending debounce tasks."""
        self._running = False
        if self._debounce_task is not None and not self._debounce_task.done():
            self._debounce_task.cancel()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        logger.info("HandlerContractFileWatcher: stopped")

    async def get_pending_triggers(self) -> list[ModelUpdateTrigger]:
        """Drain all pending triggers from the internal queue.

        Returns an empty list if no triggers are queued.
        """
        triggers: list[ModelUpdateTrigger] = []
        while not self._trigger_queue.empty():
            try:
                triggers.append(self._trigger_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return triggers

    def notify_event(self) -> None:
        """Programmatically notify the handler of a pending filesystem event.

        Useful for testing without a running watchdog observer. Schedules
        the debounce coroutine on the running event loop.
        """
        loop = getattr(self, "_loop", None)
        if loop is None or not loop.is_running():
            return
        if self._debounce_task is not None and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = loop.create_task(self._debounce_and_process())
