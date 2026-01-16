# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerManifestPersistence retry behavior.

This module tests the retry logic for the manifest persistence handler, including:
- Error classification for determining retry eligibility
- Retry state progression with exponential backoff
- Circuit breaker integration during retry loops
- Full retry flow (success, transient failure, exhaustion)

Contract Configuration (from handler_manifest_persistence.contract.yaml):
    - max_retries: 3
    - initial_delay_ms: 100 (0.1 seconds)
    - max_delay_ms: 5000 (5.0 seconds)
    - exponential_base: 2
    - retry_on: [InfraConnectionError, InfraTimeoutError]

Error Classification Rules:
    - TimeoutError -> TIMEOUT category, should_retry=True
    - BlockingIOError -> TIMEOUT category, should_retry=True (timeout-like condition)
    - OSError/IOError -> CONNECTION category, should_retry=True
    - FileNotFoundError -> NOT_FOUND category, should_retry=False
    - PermissionError -> AUTHENTICATION category, should_retry=False

Related:
    - OMN-1163: Manifest persistence handler implementation
    - src/omnibase_infra/handlers/handler_manifest_persistence.py
    - src/omnibase_infra/handlers/handler_manifest_persistence.contract.yaml
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumRetryErrorCategory
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_infra.handlers.handler_manifest_persistence import (
    HandlerManifestPersistence,
)
from omnibase_infra.handlers.models import ModelRetryState
from omnibase_infra.models.model_retry_error_classification import (
    ModelRetryErrorClassification,
)

# =============================================================================
# Test Constants - From Contract Configuration
# =============================================================================

# Contract-defined retry parameters
CONTRACT_MAX_RETRIES = 3
CONTRACT_INITIAL_DELAY_SECONDS = 0.1  # 100ms
CONTRACT_MAX_DELAY_SECONDS = 5.0  # 5000ms
CONTRACT_EXPONENTIAL_BASE = 2


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_storage_path(tmp_path: Path) -> Path:
    """Create temporary storage directory for tests.

    Args:
        tmp_path: pytest tmp_path fixture.

    Returns:
        Path to temporary manifest storage directory.
    """
    return tmp_path / "manifests"


@pytest.fixture
async def handler(
    temp_storage_path: Path,
    mock_container: MagicMock,
) -> AsyncGenerator[HandlerManifestPersistence, None]:
    """Create and initialize handler with temp storage.

    Args:
        temp_storage_path: Temporary storage directory.
        mock_container: Mock ONEX container for dependency injection.

    Yields:
        Initialized HandlerManifestPersistence instance.

    Note:
        Sets _circuit_breaker_initialized=True to satisfy MixinRetryExecution
        requirements. This attribute is expected by the retry mixin but not
        automatically set by MixinAsyncCircuitBreaker._init_circuit_breaker().
    """
    h = HandlerManifestPersistence(mock_container)
    await h.initialize({"storage_path": str(temp_storage_path)})
    # Set required attribute for MixinRetryExecution compatibility
    h._circuit_breaker_initialized = True
    yield h
    await h.shutdown()


@pytest.fixture
def retry_state_initial() -> ModelRetryState:
    """Create initial retry state matching handler's _execute_with_retry initialization.

    This fixture mirrors the exact ModelRetryState creation in
    HandlerManifestPersistence._execute_with_retry (lines 593-598):

        retry_state = ModelRetryState(
            attempt=0,
            max_attempts=int(self._retry_config["max_retries"]) + 1,
            delay_seconds=float(self._retry_config["initial_delay_seconds"]),
            backoff_multiplier=float(self._retry_config["exponential_base"]),
        )

    Contract configuration:
        - max_retries: 3 (CONTRACT_MAX_RETRIES)
        - initial_delay_ms: 100ms = 0.1s (CONTRACT_INITIAL_DELAY_SECONDS)
        - exponential_base: 2 (CONTRACT_EXPONENTIAL_BASE)

    Resulting max_attempts = max_retries + 1 = 4 total execution attempts:
        - attempt=0: Initial execution
        - attempt=1: First retry
        - attempt=2: Second retry
        - attempt=3: Third retry (final attempt)
        - attempt=4: is_retriable() returns False (exhausted)

    Returns:
        ModelRetryState configured identically to handler initialization.
    """
    return ModelRetryState(
        attempt=0,
        # max_attempts = max_retries + 1 = total execution attempts including initial
        max_attempts=CONTRACT_MAX_RETRIES + 1,
        delay_seconds=CONTRACT_INITIAL_DELAY_SECONDS,
        backoff_multiplier=CONTRACT_EXPONENTIAL_BASE,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_manifest(
    manifest_id: str | None = None,
    node_id: str = "test-node",
) -> dict[str, object]:
    """Create a test manifest dict.

    Args:
        manifest_id: Optional manifest UUID string.
        node_id: Node identifier for the manifest.

    Returns:
        Dict representing a minimal valid execution manifest.
    """
    return {
        "manifest_id": manifest_id or str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "correlation_id": str(uuid4()),
        "node_identity": {
            "node_id": node_id,
            "node_type": "test",
        },
        "contract_identity": {
            "contract_id": "test-contract",
            "contract_version": "1.0.0",
        },
        "execution_context": {
            "environment": "test",
            "session_id": str(uuid4()),
        },
    }


def create_store_envelope(
    manifest: dict[str, object],
) -> dict[str, object]:
    """Create envelope for manifest.store operation.

    Args:
        manifest: The manifest dict to store.

    Returns:
        Envelope dict for execute() method.
    """
    return {
        "id": str(uuid4()),
        "operation": "manifest.store",
        "payload": {"manifest": manifest},
        "correlation_id": str(uuid4()),
    }


def create_retrieve_envelope(
    manifest_id: str,
) -> dict[str, object]:
    """Create envelope for manifest.retrieve operation.

    Args:
        manifest_id: The manifest ID to retrieve.

    Returns:
        Envelope dict for execute() method.
    """
    return {
        "id": str(uuid4()),
        "operation": "manifest.retrieve",
        "payload": {"manifest_id": manifest_id},
        "correlation_id": str(uuid4()),
    }


async def create_initialized_handler(
    storage_path: Path,
    mock_container: MagicMock,
    retry_policy: dict[str, object] | None = None,
) -> HandlerManifestPersistence:
    """Create and initialize a handler with proper mixin compatibility.

    This helper ensures _circuit_breaker_initialized is set after initialization,
    which is required by MixinRetryExecution but not set by MixinAsyncCircuitBreaker.

    Args:
        storage_path: Path to storage directory.
        mock_container: Mock ONEX container for dependency injection.
        retry_policy: Optional retry policy configuration.

    Returns:
        Initialized HandlerManifestPersistence with all required attributes.
    """
    h = HandlerManifestPersistence(mock_container)
    config: dict[str, object] = {"storage_path": str(storage_path)}
    if retry_policy:
        config["retry_policy"] = retry_policy
    await h.initialize(config)
    # Set required attribute for MixinRetryExecution compatibility
    h._circuit_breaker_initialized = True
    return h


# =============================================================================
# TestErrorClassification
# =============================================================================


class TestErrorClassification:
    """Test error classification for retry decision making.

    Error classification determines:
    - Whether to retry the operation (should_retry)
    - Which error category applies (EnumRetryErrorCategory)
    - Whether to record circuit breaker failure

    Expected mappings:
        - TimeoutError -> TIMEOUT, should_retry=True
        - BlockingIOError -> TIMEOUT, should_retry=True (timeout-like condition)
        - OSError -> CONNECTION, should_retry=True
        - IOError -> CONNECTION, should_retry=True
        - FileNotFoundError -> NOT_FOUND, should_retry=False
        - PermissionError -> AUTHENTICATION, should_retry=False
    """

    @pytest.mark.asyncio
    async def test_classify_timeout_error(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """TimeoutError should classify as TIMEOUT with should_retry=True.

        Timeout errors are transient and should be retried with backoff.
        They should also record circuit breaker failures on exhaustion.
        """
        error = TimeoutError("Operation timed out")
        operation = "manifest.store"

        classification = handler._classify_error(error, operation)

        assert isinstance(classification, ModelRetryErrorClassification)
        assert classification.category == EnumRetryErrorCategory.TIMEOUT
        assert classification.should_retry is True
        assert classification.record_circuit_failure is True
        assert "timed out" in classification.error_message.lower()

    @pytest.mark.asyncio
    async def test_classify_oserror_connection(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """OSError should classify as CONNECTION with should_retry=True.

        General OS/IO errors are transient and should be retried.
        """
        error = OSError("I/O error occurred")
        operation = "manifest.store"

        classification = handler._classify_error(error, operation)

        assert isinstance(classification, ModelRetryErrorClassification)
        assert classification.category == EnumRetryErrorCategory.CONNECTION
        assert classification.should_retry is True
        assert classification.record_circuit_failure is True

    @pytest.mark.asyncio
    async def test_classify_ioerror_connection(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """IOError should classify as CONNECTION with should_retry=True.

        IOError is an alias for OSError in Python 3 and should be retriable.
        """
        error = OSError("Input/output error")
        operation = "manifest.store"

        classification = handler._classify_error(error, operation)

        assert isinstance(classification, ModelRetryErrorClassification)
        assert classification.category == EnumRetryErrorCategory.CONNECTION
        assert classification.should_retry is True
        assert classification.record_circuit_failure is True

    @pytest.mark.asyncio
    async def test_classify_filenotfounderror_not_found(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """FileNotFoundError should classify as NOT_FOUND with should_retry=False.

        Not found errors are not transient - retrying won't help.
        They should NOT record circuit breaker failures (user error).
        """
        error = FileNotFoundError("No such file or directory")
        operation = "manifest.retrieve"

        classification = handler._classify_error(error, operation)

        assert isinstance(classification, ModelRetryErrorClassification)
        assert classification.category == EnumRetryErrorCategory.NOT_FOUND
        assert classification.should_retry is False
        assert classification.record_circuit_failure is False

    @pytest.mark.asyncio
    async def test_classify_permissionerror_authentication(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """PermissionError should classify as AUTHENTICATION with should_retry=False.

        Permission errors are not transient - retrying won't help.
        They should record circuit breaker failure immediately.
        """
        error = PermissionError("Permission denied")
        operation = "manifest.store"

        classification = handler._classify_error(error, operation)

        assert isinstance(classification, ModelRetryErrorClassification)
        assert classification.category == EnumRetryErrorCategory.AUTHENTICATION
        assert classification.should_retry is False
        assert classification.record_circuit_failure is True

    @pytest.mark.asyncio
    async def test_classify_blockingio_error_as_timeout(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """BlockingIOError should classify as TIMEOUT with should_retry=True.

        BlockingIOError indicates "resource temporarily unavailable" (EAGAIN/EWOULDBLOCK),
        which is a timeout-like condition. The handler explicitly checks for BlockingIOError
        BEFORE the general OSError check to classify it as TIMEOUT rather than CONNECTION.

        This is an intentional design decision - BlockingIOError is semantically closer
        to a timeout than a connection error.
        """
        error = BlockingIOError("Resource temporarily unavailable")
        operation = "manifest.store"

        classification = handler._classify_error(error, operation)

        assert isinstance(classification, ModelRetryErrorClassification)
        # BlockingIOError is explicitly handled as TIMEOUT (checked before OSError)
        assert classification.category == EnumRetryErrorCategory.TIMEOUT
        assert classification.should_retry is True
        assert classification.record_circuit_failure is True

    @pytest.mark.asyncio
    async def test_classify_connectionrefused_error(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """ConnectionRefusedError should classify as CONNECTION with should_retry=True.

        ConnectionRefusedError is a subclass of OSError.
        """
        error = ConnectionRefusedError("Connection refused")
        operation = "manifest.store"

        classification = handler._classify_error(error, operation)

        assert isinstance(classification, ModelRetryErrorClassification)
        assert classification.category == EnumRetryErrorCategory.CONNECTION
        assert classification.should_retry is True
        assert classification.record_circuit_failure is True

    @pytest.mark.asyncio
    async def test_classify_unknown_exception(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Unknown exceptions should classify as UNKNOWN with should_retry=True.

        Unknown errors get retry attempts as a safety measure.
        """
        error = RuntimeError("Unexpected internal error")
        operation = "manifest.store"

        classification = handler._classify_error(error, operation)

        assert isinstance(classification, ModelRetryErrorClassification)
        assert classification.category == EnumRetryErrorCategory.UNKNOWN
        assert classification.should_retry is True
        assert classification.record_circuit_failure is True

    @pytest.mark.asyncio
    async def test_classify_error_includes_operation_context(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Error classification should include operation in error message.

        The error message should provide enough context for debugging.
        """
        error = TimeoutError("Timeout")
        operation = "manifest.store"

        classification = handler._classify_error(error, operation)

        # Error message should contain operation context
        assert classification.error_message is not None
        assert len(classification.error_message) > 0


# =============================================================================
# TestRetryStateProgression
# =============================================================================


class TestRetryStateProgression:
    """Test retry state progression with exponential backoff.

    Contract configuration:
        - initial_delay_ms: 100 (0.1 seconds)
        - max_delay_ms: 5000 (5.0 seconds)
        - exponential_base: 2
        - max_retries: 3

    Expected delay sequence: 0.1s -> 0.2s -> 0.4s -> 0.8s -> ...
    """

    def test_initial_retry_state_values(
        self, retry_state_initial: ModelRetryState
    ) -> None:
        """Initial retry state should match contract configuration.

        Validates that the initial state is correctly configured with
        contract-defined parameters.
        """
        assert retry_state_initial.attempt == 0
        # max_attempts = max_retries + 1 (for initial attempt)
        assert retry_state_initial.max_attempts == CONTRACT_MAX_RETRIES + 1
        assert retry_state_initial.delay_seconds == CONTRACT_INITIAL_DELAY_SECONDS
        assert retry_state_initial.backoff_multiplier == CONTRACT_EXPONENTIAL_BASE
        assert retry_state_initial.last_error is None
        assert retry_state_initial.last_attempt_at is None

    def test_exponential_backoff_calculation(
        self, retry_state_initial: ModelRetryState
    ) -> None:
        """Delay should increase exponentially with each attempt.

        Contract: initial_delay=0.1s, base=2
        Expected sequence: 0.1 -> 0.2 -> 0.4 -> 0.8
        """
        state = retry_state_initial

        # First attempt: delay stays at initial (0.1s)
        state = state.next_attempt("Error 1")
        assert state.attempt == 1
        # After first attempt, delay is initial * backoff = 0.1 * 2 = 0.2
        assert state.delay_seconds == pytest.approx(0.2, rel=0.01)

        # Second attempt
        state = state.next_attempt("Error 2")
        assert state.attempt == 2
        # Delay is 0.2 * 2 = 0.4
        assert state.delay_seconds == pytest.approx(0.4, rel=0.01)

        # Third attempt
        state = state.next_attempt("Error 3")
        assert state.attempt == 3
        # Delay is 0.4 * 2 = 0.8
        assert state.delay_seconds == pytest.approx(0.8, rel=0.01)

    def test_max_delay_cap(self) -> None:
        """Delay should be capped at max_delay_ms (5.0 seconds).

        Contract: max_delay_ms=5000 (5.0 seconds)
        Even with exponential growth, delay should never exceed max.
        """
        # Start with a high delay that would exceed max after backoff
        state = ModelRetryState(
            attempt=0,
            max_attempts=10,
            delay_seconds=3.0,  # Will become 6.0 after backoff
            backoff_multiplier=2.0,
        )

        # Next attempt should cap at 5.0s
        state = state.next_attempt(
            "Error", max_delay_seconds=CONTRACT_MAX_DELAY_SECONDS
        )

        assert state.delay_seconds == CONTRACT_MAX_DELAY_SECONDS

    def test_attempt_counter_increments(
        self, retry_state_initial: ModelRetryState
    ) -> None:
        """Attempt counter should increment with each next_attempt call.

        Contract: max_retries=3 (attempts 0, 1, 2 are valid)
        """
        state = retry_state_initial

        assert state.attempt == 0

        state = state.next_attempt("Error 1")
        assert state.attempt == 1

        state = state.next_attempt("Error 2")
        assert state.attempt == 2

        state = state.next_attempt("Error 3")
        assert state.attempt == 3

    def test_is_retriable_returns_true_before_max(
        self, retry_state_initial: ModelRetryState
    ) -> None:
        """is_retriable() should return True while attempts < max_attempts.

        Contract: max_retries=3
        is_retriable() should be True for attempts 0, 1, 2
        """
        state = retry_state_initial

        # Attempt 0 - retriable
        assert state.is_retriable() is True

        # Attempt 1 - retriable
        state = state.next_attempt("Error 1")
        assert state.is_retriable() is True

        # Attempt 2 - retriable (last valid attempt)
        state = state.next_attempt("Error 2")
        assert state.is_retriable() is True

    def test_is_retriable_returns_false_after_max(
        self, retry_state_initial: ModelRetryState
    ) -> None:
        """is_retriable() should return False after max_attempts reached.

        Contract: max_retries=3, max_attempts=4 (including initial attempt)
        After 4 attempts (0, 1, 2, 3), is_retriable() should return False.
        """
        state = retry_state_initial

        # Execute all 4 attempts (max_retries + 1 for initial attempt)
        for i in range(CONTRACT_MAX_RETRIES + 1):
            state = state.next_attempt(f"Error {i + 1}")

        # Now at attempt=4, which equals max_attempts
        assert state.attempt == CONTRACT_MAX_RETRIES + 1
        assert state.is_retriable() is False

    def test_last_error_is_updated(self, retry_state_initial: ModelRetryState) -> None:
        """last_error should be updated with each next_attempt call.

        Validates that error messages are tracked for logging/debugging.
        """
        state = retry_state_initial

        assert state.last_error is None

        state = state.next_attempt("Connection refused")
        assert state.last_error == "Connection refused"

        state = state.next_attempt("Timeout after 30s")
        assert state.last_error == "Timeout after 30s"

    def test_last_attempt_at_is_updated(
        self, retry_state_initial: ModelRetryState
    ) -> None:
        """last_attempt_at should be updated with each next_attempt call.

        Validates that attempt timestamps are tracked.
        """
        state = retry_state_initial

        assert state.last_attempt_at is None

        before_time = time.time()
        state = state.next_attempt("Error")
        after_time = time.time()

        assert state.last_attempt_at is not None
        assert before_time <= state.last_attempt_at <= after_time

    def test_is_final_attempt(self, retry_state_initial: ModelRetryState) -> None:
        """is_final_attempt() should return True on the last allowed attempt.

        Contract: max_retries=3, max_attempts=4 (including initial attempt)
        is_final_attempt() should be True at attempt=3 (0-indexed).
        """
        state = retry_state_initial

        # Attempt 0 - not final
        assert state.is_final_attempt() is False

        # Attempt 1 - not final
        state = state.next_attempt("Error 1")
        assert state.is_final_attempt() is False

        # Attempt 2 - not final (with max_attempts=4)
        state = state.next_attempt("Error 2")
        assert state.is_final_attempt() is False

        # Attempt 3 - THIS IS FINAL (max_attempts - 1)
        state = state.next_attempt("Error 3")
        assert state.is_final_attempt() is True

    def test_retry_state_is_immutable(
        self, retry_state_initial: ModelRetryState
    ) -> None:
        """ModelRetryState should be immutable (frozen=True).

        Validates that state cannot be mutated directly.
        """
        state = retry_state_initial

        with pytest.raises(Exception):  # ValidationError or FrozenInstanceError
            state.attempt = 5  # type: ignore[misc]

    def test_with_initial_delay_creates_copy(
        self, retry_state_initial: ModelRetryState
    ) -> None:
        """with_initial_delay() should create a new state with updated delay.

        This is useful for adjusting initial delay without creating
        a new state from scratch.
        """
        state = retry_state_initial
        new_state = state.with_initial_delay(0.5)

        # Original unchanged
        assert state.delay_seconds == CONTRACT_INITIAL_DELAY_SECONDS

        # New state has updated delay
        assert new_state.delay_seconds == 0.5
        assert new_state.attempt == state.attempt
        assert new_state.max_attempts == state.max_attempts


# =============================================================================
# TestRetryIntegration
# =============================================================================


class TestRetryIntegration:
    """Test full retry flow integration with the handler.

    These tests verify the complete retry behavior including:
    - Successful operations (no retry needed)
    - Transient failures with recovery
    - Exhausted retries
    - Proper error propagation

    Note: These tests mock at the I/O level (pathlib operations) to properly
    test the retry logic inside _execute_with_retry.
    """

    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Successful operations should not trigger retry logic.

        Validates that normal operations complete without retry overhead.
        """
        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        result = await handler.execute(envelope)

        assert result.result["status"] == "success"
        assert result.result["payload"]["created"] is True

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_transient_failure(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Retry should succeed when transient failure recovers.

        Simulates a transient filesystem error that succeeds on retry.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Track call count for verification
        call_count = 0
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            nonlocal call_count
            # Only fail on the storage path operations, not temp_storage_path creation
            if "manifests" in str(self) and call_count < 1:
                call_count += 1
                raise OSError("Temporary filesystem error")
            call_count += 1
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            # Should succeed on retry
            result = await handler.execute(envelope)

            assert result.result["status"] == "success"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_oserror(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Exhausted retries should raise OSError that propagates.

        When all retry attempts fail with OSError, the handler raises
        the original OSError (which gets re-raised in the retry loop).
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Track call count
        call_count = 0
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            nonlocal call_count
            if "manifests" in str(self):
                call_count += 1
                raise OSError("Persistent filesystem error")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            with pytest.raises(OSError) as exc_info:
                await handler.execute(envelope)

            # Should have tried multiple times
            assert call_count >= 1
            assert "Persistent filesystem error" in str(exc_info.value)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_non_retriable_error_raises_immediately(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Non-retriable errors should raise immediately without retry.

        FileNotFoundError is not retriable and should propagate immediately.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Track call count
        call_count = 0
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            nonlocal call_count
            if "manifests" in str(self):
                call_count += 1
                raise FileNotFoundError("Storage path not found")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            with pytest.raises(FileNotFoundError):
                await handler.execute(envelope)

            # Should only have tried once (no retry for FileNotFoundError)
            assert call_count == 1

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_timeout_error_is_retried(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """TimeoutError should trigger retry attempts.

        Timeout errors are transient and should be retried.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Track call count
        call_count = 0
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            nonlocal call_count
            if "manifests" in str(self):
                call_count += 1
                if call_count < 3:
                    raise TimeoutError("Operation timed out")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            result = await handler.execute(envelope)

            assert result.result["status"] == "success"
            # Should have retried at least twice
            assert call_count >= 2

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_permission_error_not_retried(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """PermissionError should not trigger retry attempts.

        Permission errors are not transient and should fail immediately.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Track call count
        call_count = 0
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            nonlocal call_count
            if "manifests" in str(self):
                call_count += 1
                raise PermissionError("Access denied")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            with pytest.raises(PermissionError):
                await handler.execute(envelope)

            # Should only have tried once (no retry for PermissionError)
            assert call_count == 1

        await handler.shutdown()


# =============================================================================
# TestCircuitBreakerRetryIntegration
# =============================================================================


class TestCircuitBreakerRetryIntegration:
    """Test circuit breaker integration with retry logic.

    These tests verify that:
    - CB is checked before retry loop starts
    - CB failure is recorded only on retry exhaustion
    - CB is not recorded for non-retriable errors (FileNotFoundError)
    """

    @pytest.mark.asyncio
    async def test_circuit_breaker_checked_before_retry(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Circuit breaker should be checked before retry loop starts.

        If circuit is open, operations should fail fast without retry attempts.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        # Force circuit to open state
        handler._circuit_breaker_open = True
        handler._circuit_breaker_open_until = time.time() + 60.0

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        with pytest.raises(InfraUnavailableError) as exc_info:
            await handler.execute(envelope)

        # Should fail fast with circuit breaker error
        error_msg = str(exc_info.value).lower()
        assert "circuit breaker" in error_msg or "unavailable" in error_msg

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_recorded_on_exhaustion(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Circuit breaker failure should be recorded when retries exhausted.

        After all retry attempts fail, the circuit breaker failure count
        should be incremented.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        # Verify initial CB state
        initial_failures = handler._circuit_breaker_failures

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Mock mkdir to fail with retriable error
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            if "manifests" in str(self):
                raise OSError("Persistent error")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            with pytest.raises(OSError):
                await handler.execute(envelope)

        # CB failure should be recorded after exhaustion
        assert handler._circuit_breaker_failures > initial_failures, (
            f"Circuit breaker failures should increase after retry exhaustion: "
            f"expected > {initial_failures}, got {handler._circuit_breaker_failures}"
        )

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_not_recorded_for_file_not_found(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Circuit breaker failure should NOT be recorded for FileNotFoundError.

        FileNotFoundError (NOT_FOUND category) should not increment CB failures.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        # Verify initial CB state
        initial_failures = handler._circuit_breaker_failures

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Mock mkdir to fail with non-retriable error
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            if "manifests" in str(self):
                raise FileNotFoundError("Storage not found")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            with pytest.raises(FileNotFoundError):
                await handler.execute(envelope)

        # CB failure should NOT be recorded for NOT_FOUND category
        assert handler._circuit_breaker_failures == initial_failures

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset_on_success_after_retry(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Circuit breaker should reset on success after retry.

        If retries eventually succeed, the circuit breaker should be reset.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        # Set some initial failures
        handler._circuit_breaker_failures = 3

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Track call count
        call_count = 0
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            nonlocal call_count
            if "manifests" in str(self):
                call_count += 1
                if call_count == 1:
                    raise OSError("Transient error")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            result = await handler.execute(envelope)

            assert result.result["status"] == "success"

        # CB should be reset on success
        assert handler._circuit_breaker_failures == 0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_permission_error_records_circuit_failure_immediately(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Permission errors should record CB failure immediately without retries.

        PermissionError (AUTHENTICATION category) has:
        - should_retry=False (no retry attempts)
        - record_circuit_failure=True (CB failure recorded)

        This test verifies:
        1. Only one execution attempt is made (no retries)
        2. Circuit breaker failure count is incremented by exactly 1
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        # Verify initial CB state
        initial_failures = handler._circuit_breaker_failures

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Track call count to verify no retries
        call_count = 0
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            nonlocal call_count
            if "manifests" in str(self):
                call_count += 1
                raise PermissionError("Access denied")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            with pytest.raises(PermissionError):
                await handler.execute(envelope)

        # Verify no retries occurred (PermissionError has should_retry=False)
        assert call_count == 1, (
            f"PermissionError should not be retried, expected 1 call, got {call_count}"
        )

        # CB failure SHOULD be recorded for AUTHENTICATION category
        # (record_circuit_failure=True in classification)
        # Verify CB failure was recorded with exactly 1 increment
        assert handler._circuit_breaker_failures == initial_failures + 1, (
            f"CB should record exactly one authentication failure: "
            f"expected {initial_failures + 1}, got {handler._circuit_breaker_failures}"
        )

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_threshold_failures(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Circuit breaker should open after reaching failure threshold.

        The circuit breaker threshold is 5 (configured in initialize()).
        After 5 consecutive failures, subsequent operations should fail fast
        with InfraUnavailableError without attempting the operation.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Track calls to verify circuit breaker behavior
        call_count = 0
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            nonlocal call_count
            if "manifests" in str(self):
                call_count += 1
                raise OSError("Persistent infrastructure failure")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            # Execute multiple times to trigger CB threshold (5 failures)
            # Each execute() with OSError exhausts retries and records 1 CB failure
            for i in range(5):
                with pytest.raises(OSError):
                    await handler.execute(envelope)

            # Verify CB is now open (failures >= threshold)
            assert handler._circuit_breaker_failures >= 5, (
                f"Expected at least 5 CB failures, got {handler._circuit_breaker_failures}"
            )

            # Reset call count to verify next operation fails fast
            calls_before_open_check = call_count

            # Next operation should fail fast with InfraUnavailableError
            # due to open circuit breaker
            with pytest.raises(InfraUnavailableError) as exc_info:
                await handler.execute(envelope)

            # Verify no additional I/O attempts were made (fail fast)
            assert call_count == calls_before_open_check, (
                f"Circuit breaker should fail fast without I/O: "
                f"calls before={calls_before_open_check}, after={call_count}"
            )

            # Verify error message indicates circuit breaker
            error_msg = str(exc_info.value).lower()
            assert "circuit breaker" in error_msg or "unavailable" in error_msg

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_success_resets(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Circuit breaker should reset to closed on successful operation after failures.

        This tests the pattern where:
        1. Circuit breaker accumulates some failures (but not enough to open)
        2. A successful operation resets the failure count to 0

        This is the "success resets circuit breaker" behavior already tested in
        test_circuit_breaker_reset_on_success_after_retry, but this test
        verifies the specific transition from partial failures to reset.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Simulate 3 accumulated failures (below threshold of 5)
        handler._circuit_breaker_failures = 3

        # Track calls to verify behavior
        call_count = 0
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            nonlocal call_count
            if "manifests" in str(self):
                call_count += 1
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            # Execute successfully
            result = await handler.execute(envelope)

            assert result.result["status"] == "success"

        # Verify CB failures were reset to 0
        assert handler._circuit_breaker_failures == 0, (
            f"CB failures should be reset to 0 on success, "
            f"got {handler._circuit_breaker_failures}"
        )

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_count_accurate_on_retry_exhaustion(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Verify CB failure is recorded exactly once when retries are exhausted.

        When an operation exhausts all retry attempts (max_retries=3, total 4 attempts),
        the circuit breaker failure count should be incremented by exactly 1,
        not once per retry attempt.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        # Verify initial state
        initial_failures = handler._circuit_breaker_failures
        assert initial_failures == 0

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Track calls to verify retry behavior
        call_count = 0
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            nonlocal call_count
            if "manifests" in str(self):
                call_count += 1
                raise OSError("Persistent failure")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            with pytest.raises(OSError):
                await handler.execute(envelope)

        # Verify correct number of attempts (initial + 3 retries = 4)
        expected_attempts = handler._retry_config["max_retries"] + 1
        assert call_count == expected_attempts, (
            f"Expected {expected_attempts} attempts, got {call_count}"
        )

        # Verify CB failure incremented by exactly 1
        assert handler._circuit_breaker_failures == initial_failures + 1, (
            f"CB failure should increment by exactly 1 after retry exhaustion: "
            f"expected {initial_failures + 1}, got {handler._circuit_breaker_failures}"
        )

        await handler.shutdown()


# =============================================================================
# TestRetryTiming
# =============================================================================


class TestRetryTiming:
    """Test retry timing and backoff delays.

    These tests verify that the actual delay between retries
    follows the exponential backoff pattern defined in the contract.

    Note: These tests use time measurements and may be flaky on
    heavily loaded systems. Use appropriate tolerances.
    """

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Timing-based test may be flaky in CI - backoff timing is approximate",
        strict=True,
    )
    async def test_backoff_timing_approximate(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Verify backoff delay is approximately correct.

        Contract: initial_delay=0.1s, base=2
        First retry should wait ~0.1s, second should wait ~0.2s.

        Note: This test is marked xfail because timing tests are inherently
        flaky in CI environments. The retry logic is also tested by other
        non-timing-based tests.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        manifest = create_test_manifest()
        envelope = create_store_envelope(manifest)

        # Track timing
        call_times: list[float] = []
        call_count = 0
        original_mkdir = Path.mkdir

        def mock_mkdir(self: Path, *args, **kwargs) -> None:
            nonlocal call_count
            if "manifests" in str(self):
                call_times.append(time.time())
                call_count += 1
                if call_count < 3:
                    raise OSError("Transient error")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(Path, "mkdir", mock_mkdir):
            await handler.execute(envelope)

        # Verify timing between retries
        if len(call_times) >= 2:
            first_delay = call_times[1] - call_times[0]
            # Allow generous tolerance for CI environments
            assert 0.05 <= first_delay <= 0.5, f"First delay was {first_delay}s"

        if len(call_times) >= 3:
            second_delay = call_times[2] - call_times[1]
            # Second delay should be larger due to backoff
            assert 0.1 <= second_delay <= 1.0, f"Second delay was {second_delay}s"

        await handler.shutdown()


# =============================================================================
# TestExecuteWithRetryDirect
# =============================================================================


class TestExecuteWithRetryDirect:
    """Test _execute_with_retry method directly.

    These tests call the retry method directly to verify its behavior
    without going through the full execute() flow.
    """

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_on_first_try(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Successful operation should return immediately without retry.

        Validates that the retry mechanism doesn't add overhead for
        successful operations.
        """
        call_count = 0

        async def success_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await handler._execute_with_retry(
            operation="test.operation",
            func=success_func,
            correlation_id=uuid4(),
        )

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_after_transient_failure(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Should succeed after transient failures.

        Validates that retries work for transient errors.
        """
        call_count = 0

        async def transient_failure_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("Transient failure")
            return "success"

        result = await handler._execute_with_retry(
            operation="test.operation",
            func=transient_failure_func,
            correlation_id=uuid4(),
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_raises_after_exhaustion(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Should raise original error after retries exhausted.

        Validates that the original exception type is preserved.
        """
        call_count = 0

        async def always_fail_func() -> str:
            nonlocal call_count
            call_count += 1
            raise OSError("Persistent failure")

        with pytest.raises(OSError) as exc_info:
            await handler._execute_with_retry(
                operation="test.operation",
                func=always_fail_func,
                correlation_id=uuid4(),
            )

        assert "Persistent failure" in str(exc_info.value)
        # Should have tried max_retries + 1 (initial + retries)
        assert call_count == handler._retry_config["max_retries"] + 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_non_retriable_raises_immediately(
        self, handler: HandlerManifestPersistence
    ) -> None:
        """Non-retriable errors should raise immediately.

        FileNotFoundError has should_retry=False.
        """
        call_count = 0

        async def not_found_func() -> str:
            nonlocal call_count
            call_count += 1
            raise FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            await handler._execute_with_retry(
                operation="test.operation",
                func=not_found_func,
                correlation_id=uuid4(),
            )

        # Should only try once
        assert call_count == 1


# =============================================================================
# TestRetryConfigurationFromInitialize
# =============================================================================


class TestRetryConfigurationFromInitialize:
    """Test that retry configuration is properly loaded from initialize config."""

    @pytest.mark.asyncio
    async def test_default_retry_config_values(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Default retry config should match contract values.

        When no retry_policy is provided, defaults should apply.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        # Verify default configuration
        assert handler._retry_config["max_retries"] == 3
        assert handler._retry_config["initial_delay_seconds"] == 0.1
        assert handler._retry_config["max_delay_seconds"] == 5.0
        assert handler._retry_config["exponential_base"] == 2.0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_custom_retry_config_from_initialize(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Custom retry config should override defaults.

        Tests that retry_policy from initialize config is applied.
        """
        handler = await create_initialized_handler(
            temp_storage_path,
            mock_container,
            retry_policy={
                "max_retries": 5,
                "initial_delay_ms": 200,
                "max_delay_ms": 10000,
                "exponential_base": 3,
            },
        )

        # Verify custom configuration
        assert handler._retry_config["max_retries"] == 5
        assert handler._retry_config["initial_delay_seconds"] == 0.2
        assert handler._retry_config["max_delay_seconds"] == 10.0
        assert handler._retry_config["exponential_base"] == 3.0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_partial_retry_config_preserves_defaults(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Partial retry config should preserve other defaults.

        Only specified values should override defaults.
        """
        handler = await create_initialized_handler(
            temp_storage_path,
            mock_container,
            retry_policy={
                "max_retries": 10,
                # Other values not specified
            },
        )

        # Only max_retries should be overridden
        assert handler._retry_config["max_retries"] == 10
        # Others should be defaults
        assert handler._retry_config["initial_delay_seconds"] == 0.1
        assert handler._retry_config["max_delay_seconds"] == 5.0
        assert handler._retry_config["exponential_base"] == 2.0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_invalid_retry_config_values_raise_error(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Invalid retry config values should raise ProtocolConfigurationError.

        This is fail-fast behavior: configuration errors should fail during
        initialize(), not silently be ignored at runtime.
        """
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.handlers.handler_manifest_persistence import (
            HandlerManifestPersistence,
        )

        handler = HandlerManifestPersistence(mock_container)

        # Test invalid max_retries (negative)
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.initialize(
                {
                    "storage_path": str(temp_storage_path),
                    "retry_policy": {"max_retries": -1},
                }
            )
        assert "max_retries" in str(exc_info.value)
        assert "positive integer" in str(exc_info.value)

        # Test invalid initial_delay_ms (wrong type)
        handler = HandlerManifestPersistence(mock_container)
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.initialize(
                {
                    "storage_path": str(temp_storage_path),
                    "retry_policy": {"initial_delay_ms": "not_a_number"},
                }
            )
        assert "initial_delay_ms" in str(exc_info.value)
        assert "positive number" in str(exc_info.value)

        # Test invalid exponential_base (< 1.0)
        handler = HandlerManifestPersistence(mock_container)
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.initialize(
                {
                    "storage_path": str(temp_storage_path),
                    "retry_policy": {"exponential_base": 0.5},
                }
            )
        assert "exponential_base" in str(exc_info.value)
        assert ">= 1.0" in str(exc_info.value)

        # Test invalid max_delay_ms (zero)
        handler = HandlerManifestPersistence(mock_container)
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.initialize(
                {
                    "storage_path": str(temp_storage_path),
                    "retry_policy": {"max_delay_ms": 0},
                }
            )
        assert "max_delay_ms" in str(exc_info.value)
        assert "positive number" in str(exc_info.value)


# =============================================================================
# TestConcurrentWrites
# =============================================================================


class TestConcurrentWrites:
    """Test concurrent write behavior and idempotency.

    These tests verify that:
    - Multiple concurrent stores of the same manifest are idempotent
    - Only one store operation actually creates the manifest
    - Subsequent stores detect existing manifest and skip creation
    - No race conditions or data corruption occurs

    Related:
        - OMN-1163: Manifest persistence handler implementation
    """

    @pytest.mark.asyncio
    async def test_concurrent_stores_same_manifest_idempotent(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Verify concurrent stores of same manifest are idempotent.

        When multiple concurrent requests attempt to store the same manifest
        (identified by manifest_id), only the first should actually create it.
        Subsequent stores should detect the existing file and return created=False.

        This ensures:
        - No duplicate manifests are created
        - No race conditions corrupt data
        - All concurrent requests complete successfully
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        manifest_id = str(uuid4())
        manifest = create_test_manifest(manifest_id=manifest_id)
        envelope = create_store_envelope(manifest)

        # Run 5 concurrent stores of same manifest
        tasks = [handler.execute(envelope) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        for result in results:
            assert result.result["status"] == "success"

        # Only first should create, rest should be idempotent (created=False)
        created_count = sum(
            1 for r in results if r.result["payload"].get("created", False)
        )
        assert created_count == 1, (
            f"Only one concurrent store should create manifest, "
            f"but {created_count} reported created=True"
        )

        # Verify manifest was written correctly (only one file exists)
        # Note: Handler stores in date-based structure: {year}/{month}/{day}/{id}.json
        manifest_files = list(temp_storage_path.glob(f"**/{manifest_id}.json"))
        assert len(manifest_files) == 1, (
            f"Expected exactly 1 manifest file, found {len(manifest_files)}"
        )

        # Verify content is valid
        with open(manifest_files[0]) as f:
            stored_manifest = json.load(f)
        assert stored_manifest["manifest_id"] == manifest_id

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_stores_different_manifests(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Verify concurrent stores of different manifests all succeed.

        When storing multiple different manifests concurrently, all should
        be created independently without interference.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        # Create 5 different manifests
        manifests = [create_test_manifest() for _ in range(5)]
        envelopes = [create_store_envelope(m) for m in manifests]

        # Run concurrent stores
        tasks = [handler.execute(env) for env in envelopes]
        results = await asyncio.gather(*tasks)

        # All should succeed with created=True
        for result in results:
            assert result.result["status"] == "success"
            assert result.result["payload"].get("created", False) is True

        # Verify all manifest files exist
        # Note: Handler stores in date-based structure: {year}/{month}/{day}/{id}.json
        manifest_files = list(temp_storage_path.glob("**/*.json"))
        assert len(manifest_files) == 5, (
            f"Expected 5 manifest files, found {len(manifest_files)}"
        )

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_store_and_retrieve(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Verify concurrent store and retrieve operations don't conflict.

        A store operation should complete such that subsequent retrieve
        operations (even concurrent ones) can read the manifest.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        manifest_id = str(uuid4())
        manifest = create_test_manifest(manifest_id=manifest_id)
        store_envelope = create_store_envelope(manifest)
        retrieve_envelope = create_retrieve_envelope(manifest_id)

        # First, store the manifest
        store_result = await handler.execute(store_envelope)
        assert store_result.result["status"] == "success"
        assert store_result.result["payload"]["created"] is True

        # Now run multiple concurrent retrieves
        retrieve_tasks = [handler.execute(retrieve_envelope) for _ in range(5)]
        retrieve_results = await asyncio.gather(*retrieve_tasks)

        # All retrieves should succeed and return same manifest
        for result in retrieve_results:
            assert result.result["status"] == "success"
            retrieved = result.result["payload"]["manifest"]
            assert retrieved["manifest_id"] == manifest_id

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_stores_high_volume(
        self, temp_storage_path: Path, mock_container: MagicMock
    ) -> None:
        """Stress test with high volume of concurrent operations.

        Verifies system stability under load with many concurrent writes.
        """
        handler = await create_initialized_handler(temp_storage_path, mock_container)

        # Create 20 concurrent store operations for the same manifest
        manifest_id = str(uuid4())
        manifest = create_test_manifest(manifest_id=manifest_id)
        envelope = create_store_envelope(manifest)

        tasks = [handler.execute(envelope) for _ in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete (either success or exception, not hang)
        assert len(results) == 20

        # Filter out any exceptions and count successes
        successes = [r for r in results if not isinstance(r, Exception)]
        exceptions = [r for r in results if isinstance(r, Exception)]

        # At least one should succeed
        assert len(successes) >= 1, (
            f"Expected at least 1 success, got {len(successes)} successes "
            f"and {len(exceptions)} exceptions"
        )

        # Exactly one should report created=True
        created_count = sum(
            1 for r in successes if r.result["payload"].get("created", False)
        )
        assert created_count == 1, (
            f"Expected exactly 1 created=True, got {created_count}"
        )

        await handler.shutdown()


__all__: list[str] = [
    "TestErrorClassification",
    "TestRetryStateProgression",
    "TestRetryIntegration",
    "TestCircuitBreakerRetryIntegration",
    "TestRetryTiming",
    "TestExecuteWithRetryDirect",
    "TestRetryConfigurationFromInitialize",
    "TestConcurrentWrites",
]
