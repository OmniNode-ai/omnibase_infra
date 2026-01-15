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
import random
import warnings
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

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
        self._state: EnumCaptureState = EnumCaptureState.IDLE
        self._config: ModelCaptureConfig | None = None
        self._corpus: ModelExecutionCorpus | None = None
        self._seen_hashes: set[str] = set()
        self._persistence = persistence

        # Metrics for monitoring
        self._capture_count: int = 0
        self._capture_missed_count: int = 0
        self._capture_timeout_count: int = 0

    @property
    def state(self) -> EnumCaptureState:
        """Get the current capture state."""
        return self._state

    def _transition_state(self, target: EnumCaptureState) -> None:
        """
        Transition to a new state with validation.

        Args:
            target: The target state to transition to.

        Raises:
            ValueError: If the transition is not valid.
        """
        if not self._state.can_transition_to(target):
            raise ValueError(
                f"Invalid state transition: {self._state.value} -> {target.value}"
            )
        self._state = target

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
        self._transition_state(EnumCaptureState.READY)
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
        self._transition_state(EnumCaptureState.CAPTURING)

    def pause_capture(self) -> None:
        """
        Pause capture without closing corpus.

        Raises:
            ValueError: If not in CAPTURING state.
        """
        self._transition_state(EnumCaptureState.PAUSED)

    def resume_capture(self) -> None:
        """
        Resume capture after pause.

        Raises:
            ValueError: If not in PAUSED state.
        """
        self._transition_state(EnumCaptureState.CAPTURING)

    def close_corpus(self) -> ModelExecutionCorpus:
        """
        Finalize and seal the corpus.

        Returns:
            The finalized corpus.

        Raises:
            ValueError: If no corpus is active.
        """
        if self._corpus is None:
            raise ValueError("No corpus to close")

        self._transition_state(EnumCaptureState.CLOSED)

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

        # Check state allows capture
        if self._state == EnumCaptureState.FULL:
            return self._make_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_CORPUS_FULL,
                start_time,
            )

        if not self._state.can_capture():
            return self._make_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_NOT_CAPTURING,
                start_time,
            )

        if self._config is None or self._corpus is None:
            return self._make_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_NOT_CAPTURING,
                start_time,
            )

        # Apply handler filter
        handler_id = manifest.node_identity.node_id
        if not self._config.is_handler_allowed(handler_id):
            return self._make_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_HANDLER_FILTER,
                start_time,
            )

        # Apply time window filter
        if not self._config.is_in_time_window(manifest.created_at):
            return self._make_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_TIME_WINDOW,
                start_time,
            )

        # Apply sample rate filter
        if self._config.sample_rate < 1.0:
            if random.random() > self._config.sample_rate:
                return self._make_result(
                    manifest.manifest_id,
                    EnumCaptureOutcome.SKIPPED_SAMPLE_RATE,
                    start_time,
                )

        # Apply deduplication
        dedupe_hash = self._compute_dedupe_hash(manifest)
        if dedupe_hash is not None and dedupe_hash in self._seen_hashes:
            return self._make_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_DUPLICATE,
                start_time,
                dedupe_hash=dedupe_hash,
            )

        # Add to corpus
        self._corpus = self._corpus.with_execution(manifest)
        if dedupe_hash is not None:
            self._seen_hashes.add(dedupe_hash)

        # Check if we hit max_executions
        if self._corpus.execution_count >= self._config.max_executions:
            self._state = EnumCaptureState.FULL

        return self._make_result(
            manifest.manifest_id,
            EnumCaptureOutcome.CAPTURED,
            start_time,
            dedupe_hash=dedupe_hash,
        )

    def _compute_dedupe_hash(self, manifest: ModelExecutionManifest) -> str | None:
        """
        Compute hash for deduplication based on configured strategy.

        Args:
            manifest: The manifest to hash.

        Returns:
            The hash string, or None if no deduplication.
        """
        if self._config is None:
            return None

        strategy = self._config.dedupe_strategy

        if strategy == EnumDedupeStrategy.NONE:
            return None

        if strategy == EnumDedupeStrategy.INPUT_HASH:
            # Hash based on handler identity (input fingerprint)
            data = f"{manifest.node_identity.node_id}:{manifest.contract_identity.contract_id}"
            return hashlib.sha256(data.encode()).hexdigest()[:16]

        if strategy == EnumDedupeStrategy.FULL_MANIFEST_HASH:
            # Hash based on full manifest (includes manifest_id, so unique)
            data = manifest.model_dump_json()
            return hashlib.sha256(data.encode()).hexdigest()[:16]

        return None

    def _make_result(
        self,
        manifest_id: UUID,
        outcome: EnumCaptureOutcome,
        start_time: datetime,
        dedupe_hash: str | None = None,
        error_message: str | None = None,
    ) -> ModelCaptureResult:
        """Create a capture result with timing."""
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

    # === Async Capture Methods ===

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
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.capture, manifest),
                timeout=effective_timeout / 1000.0,  # Convert to seconds
            )

            if result.was_captured:
                self._capture_count += 1

            return result

        except TimeoutError:
            self._capture_timeout_count += 1
            self._capture_missed_count += 1

            warnings.warn(
                f"Capture timed out after {effective_timeout}ms for manifest "
                f"{manifest.manifest_id}. Consider increasing capture_timeout_ms.",
                stacklevel=2,
            )

            return self._make_result(
                manifest.manifest_id,
                EnumCaptureOutcome.SKIPPED_TIMEOUT,
                start_time,
                error_message=f"Capture timed out after {effective_timeout}ms",
            )

        except Exception as e:
            self._capture_missed_count += 1

            return self._make_result(
                manifest.manifest_id,
                EnumCaptureOutcome.FAILED,
                start_time,
                error_message=str(e),
            )

    # === Persistence Methods ===

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
                warnings.warn(
                    f"Failed to persist manifest {manifest.manifest_id}: {e}",
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

    # === Metrics ===

    def get_metrics(self) -> dict[str, int | float]:
        """
        Get capture metrics.

        Returns:
            Dict with capture_count, capture_missed_count, capture_timeout_count,
            and corpus_size.
        """
        return {
            "capture_count": self._capture_count,
            "capture_missed_count": self._capture_missed_count,
            "capture_timeout_count": self._capture_timeout_count,
            "corpus_size": self._corpus.execution_count if self._corpus else 0,
        }
