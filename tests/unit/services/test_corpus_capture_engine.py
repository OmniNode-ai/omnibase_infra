# SPDX-FileCopyrightText: 2025 OmniNode Team <info@omninode.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
TDD tests for CorpusCaptureEngine.

Tests filtering, deduplication, and lifecycle state transitions.
Written test-first per OMN-1203 requirements.

.. versionadded:: 0.5.0
    Added for CorpusCaptureEngine (OMN-1203)
"""

import random
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.errors import OnexError
from omnibase_core.models.manifest.model_contract_identity import ModelContractIdentity
from omnibase_core.models.manifest.model_execution_manifest import (
    ModelExecutionManifest,
)
from omnibase_core.models.manifest.model_node_identity import ModelNodeIdentity
from omnibase_core.models.primitives.model_semver import ModelSemVer

from omnibase_infra.enums.enum_capture_outcome import EnumCaptureOutcome
from omnibase_infra.enums.enum_capture_state import EnumCaptureState
from omnibase_infra.enums.enum_dedupe_strategy import EnumDedupeStrategy
from omnibase_infra.models.corpus import ModelCaptureConfig

# Import will fail until service is implemented - that's expected for TDD
try:
    from omnibase_infra.services.corpus_capture_engine import CorpusCaptureEngine
except ImportError:
    CorpusCaptureEngine = None  # type: ignore[misc, assignment]


# === Test Fixtures ===


@pytest.fixture
def sample_node_identity() -> ModelNodeIdentity:
    """Create a sample node identity for testing."""
    return ModelNodeIdentity(
        node_id="test-handler-001",
        node_kind=EnumNodeKind.COMPUTE,
        node_version=ModelSemVer(major=1, minor=0, patch=0),
    )


@pytest.fixture
def sample_contract_identity() -> ModelContractIdentity:
    """Create a sample contract identity for testing."""
    return ModelContractIdentity(contract_id="test-contract-001")


@pytest.fixture
def sample_manifest(
    sample_node_identity: ModelNodeIdentity,
    sample_contract_identity: ModelContractIdentity,
) -> ModelExecutionManifest:
    """Create a sample execution manifest for testing."""
    return ModelExecutionManifest(
        manifest_id=uuid4(),
        created_at=datetime.now(UTC),
        node_identity=sample_node_identity,
        contract_identity=sample_contract_identity,
    )


def create_manifest_with_handler(
    handler_id: str, created_at: datetime | None = None
) -> ModelExecutionManifest:
    """Helper to create a manifest with specific handler ID and timestamp."""
    return ModelExecutionManifest(
        manifest_id=uuid4(),
        created_at=created_at or datetime.now(UTC),
        node_identity=ModelNodeIdentity(
            node_id=handler_id,
            node_kind=EnumNodeKind.COMPUTE,
            node_version=ModelSemVer(major=1, minor=0, patch=0),
        ),
        contract_identity=ModelContractIdentity(contract_id="test-contract"),
    )


@pytest.fixture
def basic_config() -> ModelCaptureConfig:
    """Create a basic capture config for testing."""
    return ModelCaptureConfig(
        corpus_display_name="test-corpus",
        max_executions=50,
        sample_rate=1.0,
        dedupe_strategy=EnumDedupeStrategy.INPUT_HASH,
    )


# === Filtering Logic Tests ===


@pytest.mark.skipif(CorpusCaptureEngine is None, reason="Service not yet implemented")
class TestHandlerFiltering:
    """Tests for handler whitelist filtering."""

    def test_capture_allowed_when_no_filter_configured(
        self, sample_manifest: ModelExecutionManifest
    ) -> None:
        """All handlers should be captured when handler_filter is None."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            handler_filter=None,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        result = service.capture(sample_manifest)

        assert result.was_captured
        assert result.outcome == EnumCaptureOutcome.CAPTURED

    def test_capture_allowed_when_handler_in_whitelist(self) -> None:
        """Handler in whitelist should be captured."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            handler_filter=["handler-a", "handler-b", "handler-c"],
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        manifest = create_manifest_with_handler("handler-b")
        result = service.capture(manifest)

        assert result.was_captured
        assert result.outcome == EnumCaptureOutcome.CAPTURED

    def test_capture_skipped_when_handler_not_in_whitelist(self) -> None:
        """Handler not in whitelist should be skipped."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            handler_filter=["handler-a", "handler-b"],
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        manifest = create_manifest_with_handler("handler-z")
        result = service.capture(manifest)

        assert result.was_skipped
        assert result.outcome == EnumCaptureOutcome.SKIPPED_HANDLER_FILTER


@pytest.mark.skipif(CorpusCaptureEngine is None, reason="Service not yet implemented")
class TestTimeWindowFiltering:
    """Tests for time window filtering."""

    def test_capture_allowed_when_no_time_window_configured(
        self, sample_manifest: ModelExecutionManifest
    ) -> None:
        """All timestamps should be allowed when no time window configured."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            time_window_start=None,
            time_window_end=None,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        result = service.capture(sample_manifest)

        assert result.was_captured

    def test_capture_allowed_when_within_time_window(self) -> None:
        """Manifest within time window should be captured."""
        now = datetime.now(UTC)
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            time_window_start=now - timedelta(hours=1),
            time_window_end=now + timedelta(hours=1),
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        manifest = create_manifest_with_handler("handler-a", created_at=now)
        result = service.capture(manifest)

        assert result.was_captured

    def test_capture_skipped_when_before_time_window(self) -> None:
        """Manifest before time window should be skipped."""
        now = datetime.now(UTC)
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            time_window_start=now,
            time_window_end=now + timedelta(hours=1),
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        manifest = create_manifest_with_handler(
            "handler-a", created_at=now - timedelta(minutes=30)
        )
        result = service.capture(manifest)

        assert result.was_skipped
        assert result.outcome == EnumCaptureOutcome.SKIPPED_TIME_WINDOW

    def test_capture_skipped_when_after_time_window(self) -> None:
        """Manifest after time window should be skipped."""
        now = datetime.now(UTC)
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            time_window_start=now - timedelta(hours=1),
            time_window_end=now,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        manifest = create_manifest_with_handler(
            "handler-a", created_at=now + timedelta(minutes=30)
        )
        result = service.capture(manifest)

        assert result.was_skipped
        assert result.outcome == EnumCaptureOutcome.SKIPPED_TIME_WINDOW


@pytest.mark.skipif(CorpusCaptureEngine is None, reason="Service not yet implemented")
class TestSampleRateFiltering:
    """Tests for sample rate filtering."""

    def test_all_captured_when_sample_rate_is_1(self) -> None:
        """All executions should be captured when sample_rate is 1.0."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            sample_rate=1.0,
            max_executions=100,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        captured_count = 0
        for i in range(20):
            manifest = create_manifest_with_handler(f"handler-{i}")
            result = service.capture(manifest)
            if result.was_captured:
                captured_count += 1

        assert captured_count == 20

    def test_none_captured_when_sample_rate_is_0(self) -> None:
        """No executions should be captured when sample_rate is 0."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            sample_rate=0.0,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        captured_count = 0
        for i in range(20):
            manifest = create_manifest_with_handler(f"handler-{i}")
            result = service.capture(manifest)
            if result.was_captured:
                captured_count += 1

        assert captured_count == 0

    def test_approximately_half_captured_when_sample_rate_is_half(self) -> None:
        """Approximately half should be captured when sample_rate is 0.5."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            sample_rate=0.5,
            max_executions=200,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        # Use fixed seed for reproducibility
        random.seed(42)

        captured_count = 0
        for i in range(100):
            manifest = create_manifest_with_handler(f"handler-{i}")
            result = service.capture(manifest)
            if result.was_captured:
                captured_count += 1

        # Allow 20% tolerance due to randomness
        assert 30 <= captured_count <= 70, f"Expected ~50, got {captured_count}"


# === Deduplication Tests ===


@pytest.mark.skipif(CorpusCaptureEngine is None, reason="Service not yet implemented")
class TestDeduplication:
    """Tests for deduplication strategies."""

    def test_no_dedup_when_strategy_is_none(self) -> None:
        """All executions should be captured when dedupe_strategy is NONE."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            dedupe_strategy=EnumDedupeStrategy.NONE,
            max_executions=100,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        # Capture same handler multiple times
        results = []
        for _ in range(5):
            manifest = create_manifest_with_handler("handler-a")
            results.append(service.capture(manifest))

        captured_count = sum(1 for r in results if r.was_captured)
        assert captured_count == 5

    def test_input_hash_dedup_skips_duplicate_inputs(self) -> None:
        """Duplicate inputs should be skipped with INPUT_HASH strategy."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            dedupe_strategy=EnumDedupeStrategy.INPUT_HASH,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        # Create two manifests with same handler (same "input")
        manifest1 = create_manifest_with_handler("handler-a")
        manifest2 = create_manifest_with_handler("handler-a")

        result1 = service.capture(manifest1)
        result2 = service.capture(manifest2)

        assert result1.was_captured
        assert result2.was_skipped
        assert result2.outcome == EnumCaptureOutcome.SKIPPED_DUPLICATE

    def test_input_hash_allows_different_handlers(self) -> None:
        """Different handlers should not be considered duplicates."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            dedupe_strategy=EnumDedupeStrategy.INPUT_HASH,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        manifest_a = create_manifest_with_handler("handler-a")
        manifest_b = create_manifest_with_handler("handler-b")

        result_a = service.capture(manifest_a)
        result_b = service.capture(manifest_b)

        assert result_a.was_captured
        assert result_b.was_captured

    def test_full_manifest_hash_deduplicates_same_content(self) -> None:
        """Identical content manifests should be deduplicated with FULL_MANIFEST_HASH.

        FULL_MANIFEST_HASH excludes unique identifiers (manifest_id, created_at,
        correlation_id) to enable content-based deduplication.
        """
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            dedupe_strategy=EnumDedupeStrategy.FULL_MANIFEST_HASH,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        # Two manifests with same content (only manifest_id/created_at differ)
        manifest1 = create_manifest_with_handler("handler-a")
        manifest2 = create_manifest_with_handler("handler-a")

        result1 = service.capture(manifest1)
        result2 = service.capture(manifest2)

        # First captured, second deduplicated (same content)
        assert result1.was_captured
        assert result2.was_skipped
        assert result2.outcome == EnumCaptureOutcome.SKIPPED_DUPLICATE

    def test_full_manifest_hash_allows_different_handlers(self) -> None:
        """Different handlers should not be deduplicated with FULL_MANIFEST_HASH."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            dedupe_strategy=EnumDedupeStrategy.FULL_MANIFEST_HASH,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        # Two manifests with different handlers (different content)
        manifest1 = create_manifest_with_handler("handler-a")
        manifest2 = create_manifest_with_handler("handler-b")

        result1 = service.capture(manifest1)
        result2 = service.capture(manifest2)

        # Both captured since content differs
        assert result1.was_captured
        assert result2.was_captured


# === Lifecycle State Transition Tests ===


@pytest.mark.skipif(CorpusCaptureEngine is None, reason="Service not yet implemented")
class TestLifecycleTransitions:
    """Tests for corpus lifecycle state transitions."""

    def test_initial_state_is_idle(self) -> None:
        """Service should start in IDLE state."""
        service = CorpusCaptureEngine()
        assert service.state == EnumCaptureState.IDLE

    def test_create_corpus_transitions_to_ready(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """create_corpus() should transition from IDLE to READY."""
        service = CorpusCaptureEngine()
        assert service.state == EnumCaptureState.IDLE

        service.create_corpus(basic_config)

        assert service.state == EnumCaptureState.READY

    def test_start_capture_transitions_to_capturing(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """start_capture() should transition from READY to CAPTURING."""
        service = CorpusCaptureEngine()
        service.create_corpus(basic_config)

        service.start_capture()

        assert service.state == EnumCaptureState.CAPTURING

    def test_pause_capture_transitions_to_paused(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """pause_capture() should transition from CAPTURING to PAUSED."""
        service = CorpusCaptureEngine()
        service.create_corpus(basic_config)
        service.start_capture()

        service.pause_capture()

        assert service.state == EnumCaptureState.PAUSED

    def test_resume_capture_transitions_to_capturing(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """resume_capture() should transition from PAUSED to CAPTURING."""
        service = CorpusCaptureEngine()
        service.create_corpus(basic_config)
        service.start_capture()
        service.pause_capture()

        service.resume_capture()

        assert service.state == EnumCaptureState.CAPTURING

    def test_close_corpus_transitions_to_closed(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """close_corpus() should transition to CLOSED from any active state."""
        service = CorpusCaptureEngine()
        service.create_corpus(basic_config)
        service.start_capture()

        corpus = service.close_corpus()

        assert service.state == EnumCaptureState.CLOSED
        assert corpus is not None

    def test_max_executions_triggers_full_state(self) -> None:
        """Reaching max_executions should transition to FULL."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            max_executions=3,
            dedupe_strategy=EnumDedupeStrategy.NONE,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        # Capture up to max
        for i in range(3):
            manifest = create_manifest_with_handler(f"handler-{i}")
            service.capture(manifest)

        assert service.state == EnumCaptureState.FULL

    def test_capture_skipped_when_not_capturing(
        self, sample_manifest: ModelExecutionManifest, basic_config: ModelCaptureConfig
    ) -> None:
        """Capture should be skipped when not in CAPTURING state."""
        service = CorpusCaptureEngine()
        service.create_corpus(basic_config)
        # Note: NOT calling start_capture()

        result = service.capture(sample_manifest)

        assert result.was_skipped
        assert result.outcome == EnumCaptureOutcome.SKIPPED_NOT_CAPTURING

    def test_capture_skipped_when_corpus_full(self) -> None:
        """Capture should be skipped when corpus is FULL."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            max_executions=2,
            dedupe_strategy=EnumDedupeStrategy.NONE,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        # Fill up the corpus
        for i in range(2):
            service.capture(create_manifest_with_handler(f"handler-{i}"))

        assert service.state == EnumCaptureState.FULL

        # Try to capture one more
        result = service.capture(create_manifest_with_handler("handler-overflow"))

        assert result.was_skipped
        assert result.outcome == EnumCaptureOutcome.SKIPPED_CORPUS_FULL


@pytest.mark.skipif(CorpusCaptureEngine is None, reason="Service not yet implemented")
class TestInvalidTransitions:
    """Tests for invalid state transitions (should raise errors)."""

    def test_start_capture_from_idle_raises(self) -> None:
        """start_capture() from IDLE should raise an error."""
        service = CorpusCaptureEngine()

        with pytest.raises(OnexError):
            service.start_capture()

    def test_pause_capture_from_idle_raises(self) -> None:
        """pause_capture() from IDLE should raise an error."""
        service = CorpusCaptureEngine()

        with pytest.raises(OnexError):
            service.pause_capture()

    def test_resume_capture_without_pause_raises(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """resume_capture() without prior pause should raise an error."""
        service = CorpusCaptureEngine()
        service.create_corpus(basic_config)
        service.start_capture()

        with pytest.raises(OnexError):
            service.resume_capture()

    def test_create_corpus_when_active_raises(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """create_corpus() when already active should raise an error."""
        service = CorpusCaptureEngine()
        service.create_corpus(basic_config)
        service.start_capture()

        with pytest.raises(OnexError):
            service.create_corpus(basic_config)


# === Max Executions and Eviction Tests ===


@pytest.mark.skipif(CorpusCaptureEngine is None, reason="Service not yet implemented")
class TestMaxExecutionsEnforcement:
    """Tests for max_executions hard cap and eviction."""

    def test_corpus_count_respects_max_executions(self) -> None:
        """Corpus should never exceed max_executions."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            max_executions=5,
            dedupe_strategy=EnumDedupeStrategy.NONE,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        # Try to capture more than max
        for i in range(10):
            manifest = create_manifest_with_handler(f"handler-{i}")
            service.capture(manifest)

        corpus = service.close_corpus()
        assert corpus.execution_count <= 5

    def test_get_active_corpus_returns_current_corpus(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """get_active_corpus() should return the active corpus."""
        service = CorpusCaptureEngine()
        service.create_corpus(basic_config)

        corpus = service.get_active_corpus()

        assert corpus is not None
        assert corpus.name == "test-corpus"


# === Async Capture Tests ===


@pytest.mark.skipif(CorpusCaptureEngine is None, reason="Service not yet implemented")
class TestAsyncCapture:
    """Tests for async capture with timeout."""

    @pytest.mark.asyncio
    async def test_capture_async_succeeds(
        self, sample_manifest: ModelExecutionManifest, basic_config: ModelCaptureConfig
    ) -> None:
        """capture_async() should successfully capture manifests."""
        service = CorpusCaptureEngine()
        service.create_corpus(basic_config)
        service.start_capture()

        result = await service.capture_async(sample_manifest)

        assert result.was_captured
        assert service.get_metrics()["capture_count"] == 1

    @pytest.mark.asyncio
    async def test_capture_async_uses_config_timeout(
        self, sample_manifest: ModelExecutionManifest
    ) -> None:
        """capture_async() should use timeout from config."""
        config = ModelCaptureConfig(
            corpus_display_name="test-corpus",
            capture_timeout_ms=100.0,
        )
        service = CorpusCaptureEngine()
        service.create_corpus(config)
        service.start_capture()

        result = await service.capture_async(sample_manifest)

        assert result.was_captured

    @pytest.mark.asyncio
    async def test_capture_async_respects_explicit_timeout(
        self, sample_manifest: ModelExecutionManifest, basic_config: ModelCaptureConfig
    ) -> None:
        """capture_async() should respect explicit timeout parameter."""
        service = CorpusCaptureEngine()
        service.create_corpus(basic_config)
        service.start_capture()

        result = await service.capture_async(sample_manifest, timeout_ms=200.0)

        assert result.was_captured

    @pytest.mark.asyncio
    async def test_capture_async_tracks_metrics(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """capture_async() should track capture metrics."""
        service = CorpusCaptureEngine()
        service.create_corpus(basic_config)
        service.start_capture()

        for i in range(5):
            manifest = create_manifest_with_handler(f"handler-{i}")
            await service.capture_async(manifest)

        assert service.get_metrics()["capture_count"] == 5
        assert service.get_metrics()["capture_missed_count"] == 0
        assert service.get_metrics()["capture_timeout_count"] == 0

    @pytest.mark.asyncio
    async def test_get_metrics_returns_all_counters(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """get_metrics() should return all counter values."""
        service = CorpusCaptureEngine()
        service.create_corpus(basic_config)
        service.start_capture()

        manifest = create_manifest_with_handler("handler-a")
        await service.capture_async(manifest)

        metrics = service.get_metrics()

        assert "capture_count" in metrics
        assert "capture_missed_count" in metrics
        assert "capture_timeout_count" in metrics
        assert "corpus_size" in metrics
        assert metrics["capture_count"] == 1
        assert metrics["corpus_size"] == 1


# === Persistence Tests ===


@pytest.mark.skipif(CorpusCaptureEngine is None, reason="Service not yet implemented")
class TestPersistence:
    """Tests for persistence flush functionality."""

    @pytest.mark.asyncio
    async def test_flush_without_persistence_raises(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """flush_to_persistence() should raise if no handler configured."""
        service = CorpusCaptureEngine()  # No persistence handler
        service.create_corpus(basic_config)

        with pytest.raises(OnexError, match="No persistence handler configured"):
            await service.flush_to_persistence()

    @pytest.mark.asyncio
    async def test_flush_with_persistence_stores_manifests(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """flush_to_persistence() should call handler for each manifest."""

        # Use async function for proper async handling
        async def mock_execute(envelope: dict[str, object]) -> object:
            return {"status": "success"}

        mock_persistence = MagicMock()
        mock_persistence.execute = mock_execute

        service = CorpusCaptureEngine(persistence=mock_persistence)
        service.create_corpus(basic_config)
        service.start_capture()

        # Capture some manifests
        for i in range(3):
            manifest = create_manifest_with_handler(f"handler-{i}")
            service.capture(manifest)

        # Flush to persistence
        persisted_count = await service.flush_to_persistence()

        assert persisted_count == 3

    @pytest.mark.asyncio
    async def test_close_corpus_async_flushes_by_default(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """close_corpus_async() should flush when persistence configured."""

        async def mock_execute(envelope: dict[str, object]) -> object:
            return {"status": "success"}

        mock_persistence = MagicMock()
        mock_persistence.execute = mock_execute

        service = CorpusCaptureEngine(persistence=mock_persistence)
        service.create_corpus(basic_config)
        service.start_capture()

        manifest = create_manifest_with_handler("handler-a")
        service.capture(manifest)

        corpus, persisted_count = await service.close_corpus_async()

        assert corpus is not None
        assert persisted_count == 1
        assert service.state == EnumCaptureState.CLOSED

    @pytest.mark.asyncio
    async def test_close_corpus_async_can_skip_flush(
        self, basic_config: ModelCaptureConfig
    ) -> None:
        """close_corpus_async(flush=False) should not flush."""

        async def mock_execute(envelope: dict[str, object]) -> object:
            raise AssertionError("Should not be called")

        mock_persistence = MagicMock()
        mock_persistence.execute = mock_execute

        service = CorpusCaptureEngine(persistence=mock_persistence)
        service.create_corpus(basic_config)
        service.start_capture()

        manifest = create_manifest_with_handler("handler-a")
        service.capture(manifest)

        corpus, persisted_count = await service.close_corpus_async(flush=False)

        assert corpus is not None
        assert persisted_count == 0
