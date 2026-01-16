# SPDX-FileCopyrightText: 2025 OmniNode Team <info@omninode.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for ServiceCorpusCapture with ManifestGenerator.

Tests the full pipeline integration: ManifestGenerator callback -> ServiceCorpusCapture.

.. versionadded:: 0.5.0
    Added for OMN-1203
"""

import inspect

import pytest
from omnibase_core.enums.enum_execution_status import EnumExecutionStatus
from omnibase_core.enums.enum_handler_execution_phase import EnumHandlerExecutionPhase
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.manifest.model_contract_identity import ModelContractIdentity
from omnibase_core.models.manifest.model_execution_manifest import (
    ModelExecutionManifest,
)
from omnibase_core.models.manifest.model_node_identity import ModelNodeIdentity
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.pipeline.manifest_generator import ManifestGenerator

from omnibase_infra.enums.enum_capture_outcome import EnumCaptureOutcome
from omnibase_infra.enums.enum_capture_state import EnumCaptureState
from omnibase_infra.enums.enum_dedupe_strategy import EnumDedupeStrategy
from omnibase_infra.models.corpus.model_capture_config import ModelCaptureConfig
from omnibase_infra.services.service_corpus_capture import ServiceCorpusCapture

# Check if ManifestGenerator supports callbacks (feature not yet released to PyPI)
_sig = inspect.signature(ManifestGenerator.__init__)
_has_callback_support = "on_manifest_built" in _sig.parameters

pytestmark = pytest.mark.skipif(
    not _has_callback_support,
    reason="Requires omnibase_core with on_manifest_built callback support (not yet released)",
)


class TestManifestGeneratorIntegration:
    """Tests for ManifestGenerator callback integration."""

    def test_callback_on_init(self) -> None:
        """ServiceCorpusCapture can be registered as callback in ManifestGenerator __init__."""
        # Setup capture service
        config = ModelCaptureConfig(
            corpus_display_name="callback-test",
            max_executions=10,
            dedupe_strategy=EnumDedupeStrategy.NONE,
        )
        capture_service = ServiceCorpusCapture()
        capture_service.create_corpus(config)
        capture_service.start_capture()

        # Create callback that captures manifests
        captured_manifests: list[ModelExecutionManifest] = []

        def capture_callback(manifest: ModelExecutionManifest) -> None:
            result = capture_service.capture(manifest)
            if result.was_captured:
                captured_manifests.append(manifest)

        # Create ManifestGenerator with callback registered at init
        generator = ManifestGenerator(
            node_identity=ModelNodeIdentity(
                node_id="test-node",
                node_kind=EnumNodeKind.COMPUTE,
                node_version=ModelSemVer(major=1, minor=0, patch=0),
            ),
            contract_identity=ModelContractIdentity(contract_id="test-contract"),
            on_manifest_built=[capture_callback],
        )

        # Simulate execution
        generator.start_hook("hook-1", "handler-1", EnumHandlerExecutionPhase.EXECUTE)
        generator.complete_hook("hook-1", EnumExecutionStatus.SUCCESS)

        # Build manifest - should trigger callback
        manifest = generator.build()

        # Verify capture happened
        assert len(captured_manifests) == 1
        assert captured_manifests[0].manifest_id == manifest.manifest_id
        assert capture_service.get_active_corpus() is not None
        assert capture_service.get_active_corpus().execution_count == 1

    def test_callback_registered_dynamically(self) -> None:
        """ServiceCorpusCapture can be registered dynamically via register_on_build_callback."""
        # Setup capture service
        config = ModelCaptureConfig(
            corpus_display_name="dynamic-callback-test",
            max_executions=10,
        )
        capture_service = ServiceCorpusCapture()
        capture_service.create_corpus(config)
        capture_service.start_capture()

        # Create ManifestGenerator without callback initially
        generator = ManifestGenerator(
            node_identity=ModelNodeIdentity(
                node_id="dynamic-node",
                node_kind=EnumNodeKind.EFFECT,
                node_version=ModelSemVer(major=1, minor=0, patch=0),
            ),
            contract_identity=ModelContractIdentity(contract_id="dynamic-contract"),
        )

        # Register callback dynamically
        generator.register_on_build_callback(lambda m: capture_service.capture(m))

        # Simulate execution
        generator.start_hook("hook-1", "handler-1", EnumHandlerExecutionPhase.EXECUTE)
        generator.complete_hook("hook-1", EnumExecutionStatus.SUCCESS)

        # Build manifest
        manifest = generator.build()

        # Verify capture happened
        corpus = capture_service.get_active_corpus()
        assert corpus is not None
        assert corpus.execution_count == 1

    def test_multiple_callbacks_all_invoked(self) -> None:
        """Multiple callbacks should all be invoked in order."""
        call_order: list[str] = []

        def callback_a(manifest: ModelExecutionManifest) -> None:
            call_order.append("a")

        def callback_b(manifest: ModelExecutionManifest) -> None:
            call_order.append("b")

        def callback_c(manifest: ModelExecutionManifest) -> None:
            call_order.append("c")

        generator = ManifestGenerator(
            node_identity=ModelNodeIdentity(
                node_id="multi-callback-node",
                node_kind=EnumNodeKind.COMPUTE,
                node_version=ModelSemVer(major=1, minor=0, patch=0),
            ),
            contract_identity=ModelContractIdentity(contract_id="test-contract"),
            on_manifest_built=[callback_a, callback_b],
        )

        # Register another callback dynamically
        generator.register_on_build_callback(callback_c)

        # Build manifest
        generator.build()

        # Verify all callbacks were invoked in order
        assert call_order == ["a", "b", "c"]

    def test_callback_exception_does_not_prevent_build(self) -> None:
        """Exception in callback should not prevent manifest build or other callbacks."""
        call_order: list[str] = []

        def failing_callback(manifest: ModelExecutionManifest) -> None:
            call_order.append("failing")
            raise RuntimeError("Intentional test failure")

        def success_callback(manifest: ModelExecutionManifest) -> None:
            call_order.append("success")

        generator = ManifestGenerator(
            node_identity=ModelNodeIdentity(
                node_id="exception-test-node",
                node_kind=EnumNodeKind.COMPUTE,
                node_version=ModelSemVer(major=1, minor=0, patch=0),
            ),
            contract_identity=ModelContractIdentity(contract_id="test-contract"),
            on_manifest_built=[failing_callback, success_callback],
        )

        # Build should complete and return manifest despite callback failure
        with pytest.warns(UserWarning, match="on_manifest_built callback failed"):
            manifest = generator.build()

        # Verify manifest was built
        assert manifest is not None
        assert manifest.node_identity.node_id == "exception-test-node"

        # Verify both callbacks were attempted
        assert call_order == ["failing", "success"]


class TestEndToEndCaptureWorkflow:
    """Tests for complete capture workflow."""

    def test_capture_multiple_executions(self) -> None:
        """Multiple pipeline executions should all be captured."""
        config = ModelCaptureConfig(
            corpus_display_name="multi-execution-test",
            max_executions=50,
            dedupe_strategy=EnumDedupeStrategy.NONE,
        )
        capture_service = ServiceCorpusCapture()
        capture_service.create_corpus(config)
        capture_service.start_capture()

        # Simulate 5 pipeline executions
        for i in range(5):
            generator = ManifestGenerator(
                node_identity=ModelNodeIdentity(
                    node_id=f"node-{i}",
                    node_kind=EnumNodeKind.COMPUTE,
                    node_version=ModelSemVer(major=1, minor=0, patch=0),
                ),
                contract_identity=ModelContractIdentity(contract_id=f"contract-{i}"),
                on_manifest_built=[lambda m: capture_service.capture(m)],
            )

            generator.start_hook(
                f"hook-{i}", f"handler-{i}", EnumHandlerExecutionPhase.EXECUTE
            )
            generator.complete_hook(f"hook-{i}", EnumExecutionStatus.SUCCESS)
            generator.build()

        # Verify all executions captured
        corpus = capture_service.close_corpus()
        assert corpus.execution_count == 5
        assert capture_service.state == EnumCaptureState.CLOSED

    def test_filtering_in_callback_context(self) -> None:
        """Handler filtering should work correctly in callback context."""
        config = ModelCaptureConfig(
            corpus_display_name="filter-test",
            max_executions=10,
            handler_filter=["allowed-node"],
            dedupe_strategy=EnumDedupeStrategy.NONE,
        )
        capture_service = ServiceCorpusCapture()
        capture_service.create_corpus(config)
        capture_service.start_capture()

        results: list[EnumCaptureOutcome] = []

        def capture_and_track(manifest: ModelExecutionManifest) -> None:
            result = capture_service.capture(manifest)
            results.append(result.outcome)

        # Execute with allowed node
        allowed_generator = ManifestGenerator(
            node_identity=ModelNodeIdentity(
                node_id="allowed-node",
                node_kind=EnumNodeKind.COMPUTE,
                node_version=ModelSemVer(major=1, minor=0, patch=0),
            ),
            contract_identity=ModelContractIdentity(contract_id="contract-a"),
            on_manifest_built=[capture_and_track],
        )
        allowed_generator.build()

        # Execute with blocked node
        blocked_generator = ManifestGenerator(
            node_identity=ModelNodeIdentity(
                node_id="blocked-node",
                node_kind=EnumNodeKind.COMPUTE,
                node_version=ModelSemVer(major=1, minor=0, patch=0),
            ),
            contract_identity=ModelContractIdentity(contract_id="contract-b"),
            on_manifest_built=[capture_and_track],
        )
        blocked_generator.build()

        # Verify filtering
        assert results[0] == EnumCaptureOutcome.CAPTURED
        assert results[1] == EnumCaptureOutcome.SKIPPED_HANDLER_FILTER

        corpus = capture_service.close_corpus()
        assert corpus.execution_count == 1

    def test_max_executions_stops_capture(self) -> None:
        """Capture should stop after max_executions is reached."""
        config = ModelCaptureConfig(
            corpus_display_name="max-test",
            max_executions=3,
            dedupe_strategy=EnumDedupeStrategy.NONE,
        )
        capture_service = ServiceCorpusCapture()
        capture_service.create_corpus(config)
        capture_service.start_capture()

        outcomes: list[EnumCaptureOutcome] = []

        def capture_and_track(manifest: ModelExecutionManifest) -> None:
            result = capture_service.capture(manifest)
            outcomes.append(result.outcome)

        # Execute 5 times (but max is 3)
        for i in range(5):
            generator = ManifestGenerator(
                node_identity=ModelNodeIdentity(
                    node_id=f"node-{i}",
                    node_kind=EnumNodeKind.COMPUTE,
                    node_version=ModelSemVer(major=1, minor=0, patch=0),
                ),
                contract_identity=ModelContractIdentity(contract_id=f"contract-{i}"),
                on_manifest_built=[capture_and_track],
            )
            generator.build()

        # Verify max enforcement
        assert outcomes[:3] == [EnumCaptureOutcome.CAPTURED] * 3
        assert outcomes[3:] == [EnumCaptureOutcome.SKIPPED_CORPUS_FULL] * 2
        assert capture_service.state == EnumCaptureState.FULL

        corpus = capture_service.close_corpus()
        assert corpus.execution_count == 3
