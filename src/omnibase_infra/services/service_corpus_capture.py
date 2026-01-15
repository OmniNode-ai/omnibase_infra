# SPDX-FileCopyrightText: 2025 OmniNode Team <info@omninode.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Corpus Capture Service for recording production executions.

This service observes live execution flow and selectively captures executions
based on configurable filters, building a replayable test corpus.

.. versionadded:: 0.5.0
    Added for OMN-1203
"""

import asyncio
import hashlib
import json
import logging
import random
import threading
import warnings
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

logger = logging.getLogger(__name__)

from omnibase_core.models.manifest.model_execution_manifest import (
    ModelExecutionManifest,
)
from omnibase_core.models.replay.model_execution_corpus import ModelExecutionCorpus

from omnibase_infra.enums.enum_capture_outcome import EnumCaptureOutcome
from omnibase_infra.enums.enum_capture_state import EnumCaptureState
from omnibase_infra.enums.enum_dedupe_strategy import EnumDedupeStrategy
from omnibase_infra.models.corpus.model_capture_config import ModelCaptureConfig
from omnibase_infra.models.corpus.model_capture_result import ModelCaptureResult


class ProtocolManifestPersistence(Protocol):
    """Protocol for manifest persistence handlers."""

    async def execute(self, envelope: dict[str, object]) -> object:
        """Execute a persistence operation."""
        ...


# =============================================================================
# Module-Level Helper Functions (extracted to reduce class method count)
# =============================================================================


def _compute_dedupe_hash(
    manifest: ModelExecutionManifest,
    strategy: EnumDedupeStrategy,
) -> str | None:
    """
    Compute hash for deduplication based on configured strategy.

    Args:
        manifest: The manifest to hash.
        strategy: The deduplication strategy to apply.

    Returns:
        The hash string, or None if no deduplication.

    Raises:
        ValueError: If strategy is unknown.
    """
    if strategy == EnumDedupeStrategy.NONE:
        return None

    if strategy == EnumDedupeStrategy.INPUT_HASH:
        # Hash based on handler identity (input fingerprint)
        data = (
            f"{manifest.node_identity.node_id}:{manifest.contract_identity.contract_id}"
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    if strategy == EnumDedupeStrategy.FULL_MANIFEST_HASH:
        # Hash based on manifest content, excluding unique identifiers
        # (manifest_id, created_at, correlation_id) that would defeat deduplication
        manifest_data = manifest.model_dump(mode="json")
        # Remove unique per-execution fields to enable content-based deduplication
        manifest_data.pop("manifest_id", None)
        manifest_data.pop("created_at", None)
        manifest_data.pop("correlation_id", None)
        # Sort keys for deterministic hashing
        data = json.dumps(manifest_data, sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    # Explicit error for unknown strategies - fail fast rather than silently returning None
    raise ValueError(f"Unknown dedupe strategy: {strategy}")


def _create_capture_result(
    manifest_id: UUID,
    outcome: EnumCaptureOutcome,
    start_time: datetime,
    dedupe_hash: str | None = None,
    error_message: str | None = None,
) -> ModelCaptureResult:
    """
    Create a capture result with timing.

    Module-level helper to keep parameter count within limits.

    Args:
        manifest_id: ID of the manifest processed.
        outcome: Result of the capture attempt.
        start_time: When capture processing started.
        dedupe_hash: Hash used for deduplication (if computed).
        error_message: Error message if capture failed.

    Returns:
        A ModelCaptureResult with computed duration.
    """
    end_time = datetime.now(UTC)
    duration_ms = (end_time - start_time).total_seconds() * 1000
    was_captured = outcome == EnumCaptureOutcome.CAPTURED

    return ModelCaptureResult(
        manifest_id=manifest_id,
        outcome=outcome,
        captured_at=end_time if was_captured else None,
        dedupe_hash=dedupe_hash,
        duration_ms=duration_ms,
        error_message=error_message,
    )


# =============================================================================
# State Machine Helper Class (extracted to reduce service class method count)
# =============================================================================


class CaptureLifecycleFSM:
    """
    Internal state machine for capture lifecycle management.

    Encapsulates state transition logic to keep the main service class focused
    on capture operations.

    Note:
        This is an internal helper class and should not be imported directly.
        Use ServiceCorpusCapture as the public API.

    .. versionadded:: 0.5.0
        Extracted from ServiceCorpusCapture for OMN-1203 refactoring.
    """

    # Valid state transitions map
    _VALID_TRANSITIONS: dict[EnumCaptureState, set[EnumCaptureState]] = {
        EnumCaptureState.IDLE: {EnumCaptureState.READY},
        EnumCaptureState.READY: {
            EnumCaptureState.CAPTURING,
            EnumCaptureState.CLOSED,
        },
        EnumCaptureState.CAPTURING: {
            EnumCaptureState.PAUSED,
            EnumCaptureState.FULL,
            EnumCaptureState.CLOSED,
        },
        EnumCaptureState.PAUSED: {
            EnumCaptureState.CAPTURING,
            EnumCaptureState.CLOSED,
        },
        EnumCaptureState.FULL: {EnumCaptureState.CLOSED},
        EnumCaptureState.CLOSED: set(),  # Terminal state
    }

    def __init__(self) -> None:
        """Initialize state machine in IDLE state."""
        self._state: EnumCaptureState = EnumCaptureState.IDLE

    @property
    def state(self) -> EnumCaptureState:
        """Get the current capture state."""
        return self._state

    @state.setter
    def state(self, value: EnumCaptureState) -> None:
        """Set state directly (for internal use in critical sections)."""
        self._state = value

    def transition_to(self, target: EnumCaptureState) -> None:
        """
        Transition to a new state with validation.

        Args:
            target: The target state to transition to.

        Raises:
            ValueError: If the transition is not valid.
        """
        valid_targets = self._VALID_TRANSITIONS.get(self._state, set())
        if target not in valid_targets:
            raise ValueError(
                f"Invalid state transition: {self._state.value} -> {target.value}"
            )
        self._state = target

    def can_capture(self) -> bool:
        """Check if current state allows capture."""
        return self._state.can_capture()


# =============================================================================
# Main Service Class
# =============================================================================


class ServiceCorpusCapture:
    """
    Service for capturing production executions into a corpus.

    This service provides:
    - Configurable filtering (handler whitelist, time window, sample rate)
    - Deduplication strategies (input hash, full manifest hash)
    - Lifecycle management (create, start, pause, resume, close)
    - Max executions enforcement with automatic state transitions

    Example:
        >>> config = ModelCaptureConfig(
        ...     corpus_display_name="regression-suite-v1",
        ...     max_executions=50,
        ...     sample_rate=0.5,
        ...     handler_filter=["compute-handler"],
        ... )
        >>> service = ServiceCorpusCapture()
        >>> service.create_corpus(config)
        >>> service.start_capture()
        >>>
        >>> # In production pipeline callback
        >>> def on_manifest_built(manifest: ModelExecutionManifest) -> None:
        ...     service.capture(manifest)
        >>>
        >>> # When done
        >>> corpus = service.close_corpus()

    .. versionadded:: 0.5.0
        Added for OMN-1203
    """

    def __init__(
        self,
        persistence: ProtocolManifestPersistence | None = None,
    ) -> None:
        """
        Initialize the corpus capture service.

        Args:
            persistence: Optional persistence handler for flushing manifests.
                If provided, manifests will be persisted via this handler
                when close_corpus() is called or max_executions is reached.
        """
        self._state_machine = CaptureLifecycleFSM()
        self._config: ModelCaptureConfig | None = None
        self._corpus: ModelExecutionCorpus | None = None
        self._seen_hashes: set[str] = set()
        self._persistence = persistence

        # Lock for thread-safe access to shared mutable state in async methods
        self._async_lock = asyncio.Lock()
        # Lock for thread-safe access when capture() runs in thread pool
        self._sync_lock = threading.Lock()

        # Metrics for monitoring
        self._capture_count: int = 0
        self._capture_missed_count: int = 0
        self._capture_timeout_count: int = 0

    @property
    def state(self) -> EnumCaptureState:
        """Get the current capture state."""
        with self._sync_lock:
            return self._state_machine.state

    def create_corpus(self, config: ModelCaptureConfig) -> ModelExecutionCorpus:
        """
        Initialize a new corpus for capture.

        Args:
            config: Configuration for the capture session.

        Returns:
            The newly created (empty) corpus.

        Raises:
            ValueError: If not in IDLE state.
        """
        with self._sync_lock:
            self._state_machine.transition_to(EnumCaptureState.READY)
            self._config = config
            self._seen_hashes = set()

            # Create empty corpus
            self._corpus = ModelExecutionCorpus(
                name=config.corpus_display_name,
                version="1.0.0",
                source="capture",
            )

            return self._corpus

    def start_capture(self) -> None:
        """
        Begin capturing executions.

        Raises:
            ValueError: If not in READY state.
        """
        with self._sync_lock:
            self._state_machine.transition_to(EnumCaptureState.CAPTURING)

    def pause_capture(self) -> None:
        """
        Pause capture without closing corpus.

        Raises:
            ValueError: If not in CAPTURING state.
        """
        with self._sync_lock:
            self._state_machine.transition_to(EnumCaptureState.PAUSED)

    def resume_capture(self) -> None:
        """
        Resume capture after pause.

        Raises:
            ValueError: If not in PAUSED state.
        """
        with self._sync_lock:
            self._state_machine.transition_to(EnumCaptureState.CAPTURING)

    def close_corpus(self) -> ModelExecutionCorpus:
        """
        Finalize and seal the corpus.

        Returns:
            The finalized corpus.

        Raises:
            ValueError: If no corpus is active.
        """
        with self._sync_lock:
            if self._corpus is None:
                raise ValueError("No corpus to close")

            self._state_machine.transition_to(EnumCaptureState.CLOSED)

            corpus = self._corpus
            self._corpus = None
            self._config = None
            self._seen_hashes = set()

        return corpus

    def get_active_corpus(self) -> ModelExecutionCorpus | None:
        """
        Get the currently active corpus.

        Returns:
            The active corpus, or None if no corpus is active.
        """
        return self._corpus

    def capture(self, manifest: ModelExecutionManifest) -> ModelCaptureResult:
        """
        Attempt to capture an execution.

        This method applies all configured filters and deduplication
        to determine if the manifest should be added to the corpus.

        Args:
            manifest: The execution manifest to potentially capture.

        Returns:
            Result indicating whether the capture succeeded or was skipped.
        """
        start_time = datetime.now(UTC)

        # Snapshot config early for thread-safety - prevents race with close_corpus()
        # which may set _config to None between our checks and usage.
        # This read is safe because Python's GIL ensures atomic reference reads.
        config = self._config

        # Check state allows capture
        if self._state_machine.state == EnumCaptureState.FULL:
            return _create_capture_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_CORPUS_FULL,
                start_time,
            )

        if not self._state_machine.can_capture():
            return _create_capture_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_NOT_CAPTURING,
                start_time,
            )

        if config is None or self._corpus is None:
            return _create_capture_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_NOT_CAPTURING,
                start_time,
            )

        # Apply handler filter (using snapshot)
        handler_id = manifest.node_identity.node_id
        if not config.is_handler_allowed(handler_id):
            return _create_capture_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_HANDLER_FILTER,
                start_time,
            )

        # Apply time window filter (using snapshot)
        if not config.is_in_time_window(manifest.created_at):
            return _create_capture_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_TIME_WINDOW,
                start_time,
            )

        # Apply sample rate filter (using snapshot)
        if config.sample_rate < 1.0:
            if random.random() > config.sample_rate:
                return _create_capture_result(
                    manifest.manifest_id,
                    EnumCaptureOutcome.SKIPPED_SAMPLE_RATE,
                    start_time,
                )

        # Apply deduplication - compute hash outside lock (pure computation)
        dedupe_hash = _compute_dedupe_hash(manifest, config.dedupe_strategy)

        # Thread-safe critical section for shared state access
        # Protects: _seen_hashes (check + add), _corpus (update), _state (update)
        with self._sync_lock:
            # Re-check state under lock (may have changed)
            if self._state_machine.state == EnumCaptureState.FULL:
                return _create_capture_result(
                    manifest.manifest_id,
                    EnumCaptureOutcome.SKIPPED_CORPUS_FULL,
                    start_time,
                )

            if dedupe_hash is not None and dedupe_hash in self._seen_hashes:
                return _create_capture_result(
                    manifest.manifest_id,
                    EnumCaptureOutcome.SKIPPED_DUPLICATE,
                    start_time,
                    dedupe_hash=dedupe_hash,
                )

            # Add to corpus (requires _corpus not None, checked above)
            if self._corpus is None:
                return _create_capture_result(
                    manifest.manifest_id,
                    EnumCaptureOutcome.SKIPPED_NOT_CAPTURING,
                    start_time,
                )

            self._corpus = self._corpus.with_execution(manifest)
            if dedupe_hash is not None:
                self._seen_hashes.add(dedupe_hash)

            # Check if we hit max_executions (using snapshot for thread-safety)
            if self._corpus.execution_count >= config.max_executions:
                self._state_machine.state = EnumCaptureState.FULL

        return _create_capture_result(
            manifest.manifest_id,
            EnumCaptureOutcome.CAPTURED,
            start_time,
            dedupe_hash=dedupe_hash,
        )

    async def capture_async(
        self,
        manifest: ModelExecutionManifest,
        timeout_ms: float | None = None,
    ) -> ModelCaptureResult:
        """
        Async capture with configurable timeout.

        This method provides bounded latency for production use. If the capture
        operation exceeds the timeout, it returns a SKIPPED_TIMEOUT result
        rather than blocking the caller.

        Args:
            manifest: The execution manifest to potentially capture.
            timeout_ms: Timeout in milliseconds. If None, uses config value.
                Defaults to 50ms from config.

        Returns:
            Result indicating whether the capture succeeded, was skipped,
            or timed out.
        """
        start_time = datetime.now(UTC)
        effective_timeout = timeout_ms
        if effective_timeout is None and self._config is not None:
            effective_timeout = self._config.capture_timeout_ms
        if effective_timeout is None:
            effective_timeout = 50.0  # Default 50ms

        try:
            # Run capture in executor to avoid blocking if serialization is slow
            # Use get_running_loop() as the modern approach (get_event_loop() is deprecated)
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self.capture, manifest),
                timeout=effective_timeout / 1000.0,  # Convert to seconds
            )

            async with self._async_lock:
                if result.was_captured:
                    self._capture_count += 1

            return result

        except TimeoutError:
            async with self._async_lock:
                self._capture_timeout_count += 1
                self._capture_missed_count += 1

            warnings.warn(
                f"Capture timed out after {effective_timeout}ms for manifest "
                f"{manifest.manifest_id}. Consider increasing capture_timeout_ms.",
                stacklevel=2,
            )

            return _create_capture_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_TIMEOUT,
                start_time,
                error_message=f"Capture timed out after {effective_timeout}ms",
            )

        except Exception as e:
            async with self._async_lock:
                self._capture_missed_count += 1

            # Log detailed exception internally for debugging
            logger.exception(
                "Capture failed for manifest %s: %s",
                manifest.manifest_id,
                e,
            )

            # Sanitize error message for external exposure - use exception type only
            sanitized_error = f"Capture failed: {type(e).__name__}"

            return _create_capture_result(
                manifest.manifest_id,
                EnumCaptureOutcome.FAILED,
                start_time,
                error_message=sanitized_error,
            )

    async def flush_to_persistence(self) -> int:
        """
        Flush captured manifests to persistence layer.

        This method stores all captured manifests via the persistence handler
        if one was provided. Called automatically on close_corpus() if
        persistence is configured.

        Returns:
            Number of manifests successfully persisted.

        Raises:
            RuntimeError: If no persistence handler is configured.
        """
        if self._persistence is None:
            raise RuntimeError("No persistence handler configured")

        if self._corpus is None or self._corpus.execution_count == 0:
            return 0

        persisted_count = 0
        for manifest in self._corpus.executions:
            try:
                await self._persistence.execute(
                    {
                        "operation": "manifest.store",
                        "payload": {
                            "manifest": manifest.model_dump(mode="json"),
                        },
                        "correlation_id": str(manifest.correlation_id)
                        if manifest.correlation_id
                        else None,
                    }
                )
                persisted_count += 1
            except Exception as e:
                # Log detailed exception internally for debugging
                logger.exception(
                    "Failed to persist manifest %s: %s",
                    manifest.manifest_id,
                    e,
                )
                # Emit sanitized warning (no raw exception details)
                warnings.warn(
                    f"Failed to persist manifest {manifest.manifest_id}: "
                    f"{type(e).__name__}",
                    stacklevel=2,
                )

        return persisted_count

    async def close_corpus_async(
        self,
        flush: bool = True,
    ) -> tuple[ModelExecutionCorpus, int]:
        """
        Async close and optionally flush corpus.

        Args:
            flush: If True and persistence is configured, flush all manifests
                before closing.

        Returns:
            Tuple of (closed corpus, number of manifests persisted).
        """
        persisted_count = 0

        if flush and self._persistence is not None:
            persisted_count = await self.flush_to_persistence()

        corpus = self.close_corpus()
        return corpus, persisted_count

    def get_metrics(self) -> dict[str, int | float]:
        """
        Get capture metrics.

        Returns:
            Dict with capture_count, capture_missed_count, capture_timeout_count,
            and corpus_size.
        """
        with self._sync_lock:
            return {
                "capture_count": self._capture_count,
                "capture_missed_count": self._capture_missed_count,
                "capture_timeout_count": self._capture_timeout_count,
                "corpus_size": self._corpus.execution_count if self._corpus else 0,
            }
