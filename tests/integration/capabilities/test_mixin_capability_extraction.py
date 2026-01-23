# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for MixinNodeIntrospection capability extraction.

Tests that the mixin correctly extracts contract_capabilities when
a contract is provided in the config.

OMN-1136: Contract capability extraction integration tests.

Test Categories:
    1. Basic Extraction: Verifies automatic extraction when contract is provided
    2. Graceful Degradation: Verifies None is returned when contract causes errors
    3. Inference Integration: Verifies inference rules are applied correctly
    4. Independence: Verifies declared_capabilities and contract_capabilities are separate
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from omnibase_core.enums import EnumNodeKind, EnumNodeType
from omnibase_core.models.contracts import (
    ModelContractEffect,
    ModelContractOrchestrator,
    ModelContractReducer,
    ModelEventSubscription,
    ModelIOOperationConfig,
    ModelPerformanceRequirements,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.mixins import MixinNodeIntrospection
from omnibase_infra.models.discovery import ModelIntrospectionConfig

if TYPE_CHECKING:
    from omnibase_core.models.capabilities import ModelContractCapabilities

# Module-level markers
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
]

# Test UUIDs - use deterministic values for reproducible tests
TEST_NODE_UUID_1 = UUID("00000000-0000-0000-0000-000000000001")
TEST_NODE_UUID_2 = UUID("00000000-0000-0000-0000-000000000002")
TEST_NODE_UUID_3 = UUID("00000000-0000-0000-0000-000000000003")


# =============================================================================
# Contract Factory Functions
# =============================================================================


def create_effect_contract(
    name: str = "test-effect",
    version: ModelSemVer | None = None,
    tags: list[str] | None = None,
    protocol_interfaces: list[str] | None = None,
) -> ModelContractEffect:
    """Create a valid ModelContractEffect for testing.

    Args:
        name: Contract name
        version: Contract version (defaults to 1.0.0)
        tags: Explicit capability tags
        protocol_interfaces: Protocol interface names

    Returns:
        Valid ModelContractEffect instance
    """
    return ModelContractEffect(
        name=name,
        version=version or ModelSemVer(major=1, minor=0, patch=0),
        description="Test effect contract",
        node_type=EnumNodeType.EFFECT_GENERIC,
        input_model="object",
        output_model="object",
        tags=tags or [],
        protocol_interfaces=protocol_interfaces or [],
        io_operations=[
            ModelIOOperationConfig(
                name="test_operation",
                operation_type="read",
                target="database",
            )
        ],
    )


def create_reducer_contract(
    name: str = "test-reducer",
    version: ModelSemVer | None = None,
    tags: list[str] | None = None,
    protocol_interfaces: list[str] | None = None,
) -> ModelContractReducer:
    """Create a valid ModelContractReducer for testing.

    Args:
        name: Contract name
        version: Contract version (defaults to 1.0.0)
        tags: Explicit capability tags
        protocol_interfaces: Protocol interface names

    Returns:
        Valid ModelContractReducer instance
    """
    return ModelContractReducer(
        name=name,
        version=version or ModelSemVer(major=1, minor=0, patch=0),
        description="Test reducer contract",
        node_type=EnumNodeType.REDUCER_GENERIC,
        input_model="object",
        output_model="object",
        tags=tags or [],
        protocol_interfaces=protocol_interfaces or [],
    )


def create_orchestrator_contract(
    name: str = "test-orchestrator",
    version: ModelSemVer | None = None,
    tags: list[str] | None = None,
    protocol_interfaces: list[str] | None = None,
    consumed_events: list[ModelEventSubscription] | None = None,
) -> ModelContractOrchestrator:
    """Create a valid ModelContractOrchestrator for testing.

    Args:
        name: Contract name
        version: Contract version (defaults to 1.0.0)
        tags: Explicit capability tags
        protocol_interfaces: Protocol interface names
        consumed_events: List of event subscriptions

    Returns:
        Valid ModelContractOrchestrator instance
    """
    return ModelContractOrchestrator(
        name=name,
        version=version or ModelSemVer(major=1, minor=0, patch=0),
        description="Test orchestrator contract",
        node_type=EnumNodeType.ORCHESTRATOR_GENERIC,
        input_model="object",
        output_model="object",
        tags=tags or [],
        protocol_interfaces=protocol_interfaces or [],
        consumed_events=consumed_events or [],
        # Orchestrators require performance requirements
        performance=ModelPerformanceRequirements(
            single_operation_max_ms=5000,
        ),
    )


# =============================================================================
# Test Node Classes
# =============================================================================


class EffectNodeWithContract(MixinNodeIntrospection):
    """Test effect node that initializes with a contract for capability extraction."""

    def __init__(self, node_id: UUID, contract: object | None = None) -> None:
        """Initialize test node.

        Args:
            node_id: Unique identifier for this node.
            contract: Optional contract for capability extraction.
        """
        config = ModelIntrospectionConfig(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            version="1.0.0",
            contract=contract,
        )
        self.initialize_introspection(config)

    async def execute_operation(self, data: dict[str, str]) -> dict[str, str]:
        """Example operation method for introspection discovery."""
        return data


class LegacyNodeWithoutContract(MixinNodeIntrospection):
    """Test node that initializes without a contract (legacy mode)."""

    def __init__(self, node_id: UUID) -> None:
        """Initialize test node without contract.

        Args:
            node_id: Unique identifier for this node.
        """
        config = ModelIntrospectionConfig(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            version="1.0.0",
            contract=None,  # No contract - legacy mode
        )
        self.initialize_introspection(config)

    async def execute_operation(self, data: dict[str, str]) -> dict[str, str]:
        """Example operation method for introspection discovery."""
        return data


# =============================================================================
# Test Class: Basic Capability Extraction
# =============================================================================


class TestMixinCapabilityExtraction:
    """Tests for automatic capability extraction in the mixin."""

    async def test_contract_capabilities_populated_when_contract_provided(self) -> None:
        """When config has contract, introspection should have contract_capabilities."""
        # Create a real contract with tags and protocols
        contract = create_effect_contract(
            tags=["test.capability"],
            protocol_interfaces=["ProtocolDatabaseAdapter"],
        )

        # Create test node with contract
        node = EffectNodeWithContract(node_id=TEST_NODE_UUID_1, contract=contract)

        # Get introspection data
        event = await node.get_introspection_data()

        # Verify contract_capabilities is populated
        assert event.contract_capabilities is not None, (
            "contract_capabilities should be populated when contract is provided"
        )
        assert event.contract_capabilities.contract_type == "effect", (
            f"Expected contract_type 'effect', got '{event.contract_capabilities.contract_type}'"
        )
        assert "test.capability" in event.contract_capabilities.capability_tags, (
            f"Expected 'test.capability' in capability_tags, got {event.contract_capabilities.capability_tags}"
        )
        assert "ProtocolDatabaseAdapter" in event.contract_capabilities.protocols, (
            f"Expected 'ProtocolDatabaseAdapter' in protocols, got {event.contract_capabilities.protocols}"
        )

    async def test_contract_capabilities_none_when_no_contract(self) -> None:
        """When config has no contract, contract_capabilities should be None."""
        # Create test node without contract
        node = LegacyNodeWithoutContract(node_id=TEST_NODE_UUID_2)

        # Get introspection data
        event = await node.get_introspection_data()

        # contract_capabilities should be None
        assert event.contract_capabilities is None, (
            "contract_capabilities should be None when no contract is provided"
        )

    async def test_declared_capabilities_unchanged(self) -> None:
        """declared_capabilities should still work independently."""
        contract = create_effect_contract()

        node = EffectNodeWithContract(node_id=TEST_NODE_UUID_3, contract=contract)

        event = await node.get_introspection_data()

        # declared_capabilities should still be present (even if empty)
        assert event.declared_capabilities is not None, (
            "declared_capabilities should always be present"
        )
        # contract_capabilities is separate and populated
        assert event.contract_capabilities is not None, (
            "contract_capabilities should be populated when contract provided"
        )

    async def test_extraction_is_automatic(self) -> None:
        """Extraction should happen automatically - cannot be skipped."""
        # This test verifies that providing a contract automatically
        # triggers extraction - there's no way to provide a contract
        # and NOT get capabilities extracted
        contract = create_reducer_contract(
            version=ModelSemVer(major=2, minor=0, patch=0),
            tags=["auto.extracted"],
        )

        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type=EnumNodeKind.REDUCER,
            version="2.0.0",
            contract=contract,
        )

        class ReducerNode(MixinNodeIntrospection):
            def __init__(self) -> None:
                self.initialize_introspection(config)

        node = ReducerNode()
        event = await node.get_introspection_data()

        # Extraction happened automatically
        assert event.contract_capabilities is not None, (
            "contract_capabilities should be automatically extracted"
        )
        # Explicit tags extracted
        assert "auto.extracted" in event.contract_capabilities.capability_tags, (
            f"Expected 'auto.extracted' in capability_tags, got {event.contract_capabilities.capability_tags}"
        )
        # Node type tag applied (from inference rules)
        assert "node.reducer" in event.contract_capabilities.capability_tags, (
            f"Expected 'node.reducer' in capability_tags, got {event.contract_capabilities.capability_tags}"
        )


# =============================================================================
# Test Class: Graceful Degradation
# =============================================================================


class TestGracefulDegradation:
    """Tests for graceful degradation when contract extraction fails."""

    async def test_extraction_with_valid_contract_succeeds(self) -> None:
        """Valid contract should extract successfully."""
        contract = create_effect_contract(tags=["valid.tag"])

        node = EffectNodeWithContract(node_id=TEST_NODE_UUID_1, contract=contract)
        event = await node.get_introspection_data()

        # Should succeed
        assert event.contract_capabilities is not None
        assert "valid.tag" in event.contract_capabilities.capability_tags

    async def test_no_contract_returns_none_gracefully(self) -> None:
        """No contract should gracefully return None."""
        node = LegacyNodeWithoutContract(node_id=TEST_NODE_UUID_1)
        event = await node.get_introspection_data()

        # contract_capabilities should be None (graceful degradation)
        assert event.contract_capabilities is None, (
            "contract_capabilities should be None when no contract provided"
        )


# =============================================================================
# Test Class: Capability Inference Integration
# =============================================================================


class TestCapabilityInferenceIntegration:
    """Tests for inference rules integration."""

    async def test_intent_types_trigger_inference_postgres(self) -> None:
        """Intent types with postgres.* prefix should trigger tag inference."""
        # Create orchestrator with consumed_events having postgres pattern
        consumed_event = ModelEventSubscription(
            event_pattern="postgres.upsert",
            handler_function="handle_postgres_event",
        )

        contract = create_orchestrator_contract(
            consumed_events=[consumed_event],
        )

        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type=EnumNodeKind.ORCHESTRATOR,
            version="1.0.0",
            contract=contract,
        )

        class OrchestratorNode(MixinNodeIntrospection):
            def __init__(self) -> None:
                self.initialize_introspection(config)

        node = OrchestratorNode()
        event = await node.get_introspection_data()

        assert event.contract_capabilities is not None, (
            "contract_capabilities should be populated"
        )
        # Inferred tag from postgres.* pattern
        assert "postgres.storage" in event.contract_capabilities.capability_tags, (
            f"Expected 'postgres.storage' from postgres.upsert intent, got {event.contract_capabilities.capability_tags}"
        )

    async def test_protocols_trigger_inference(self) -> None:
        """Protocols in contract should trigger tag inference."""
        contract = create_effect_contract(
            protocol_interfaces=["ProtocolReducer"],
        )

        node = EffectNodeWithContract(node_id=TEST_NODE_UUID_1, contract=contract)
        event = await node.get_introspection_data()

        assert event.contract_capabilities is not None, (
            "contract_capabilities should be populated"
        )
        # Inferred tag from ProtocolReducer
        assert "state.reducer" in event.contract_capabilities.capability_tags, (
            f"Expected 'state.reducer' from ProtocolReducer, got {event.contract_capabilities.capability_tags}"
        )

    async def test_node_type_triggers_inference(self) -> None:
        """Node type should trigger base capability tag inference."""
        contract = ModelContractReducer(
            name="compute-test",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test compute contract",
            node_type=EnumNodeType.COMPUTE_GENERIC,
            input_model="object",
            output_model="object",
        )

        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type=EnumNodeKind.COMPUTE,
            version="1.0.0",
            contract=contract,
        )

        class ComputeNode(MixinNodeIntrospection):
            def __init__(self) -> None:
                self.initialize_introspection(config)

        node = ComputeNode()
        event = await node.get_introspection_data()

        assert event.contract_capabilities is not None
        # Node type tag inferred from contract node_type
        assert "node.compute" in event.contract_capabilities.capability_tags, (
            f"Expected 'node.compute' from COMPUTE_GENERIC, got {event.contract_capabilities.capability_tags}"
        )

    async def test_multiple_inference_sources_combined(self) -> None:
        """All inference sources should be combined into capability_tags."""
        # Set up contract with multiple inference triggers
        consumed_event = ModelEventSubscription(
            event_pattern="kafka.publish",
            handler_function="handle_kafka_event",
        )

        contract = create_orchestrator_contract(
            tags=["explicit.tag"],
            protocol_interfaces=["ProtocolEventBus"],
            consumed_events=[consumed_event],
        )

        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID_1,
            node_type=EnumNodeKind.ORCHESTRATOR,
            version="1.0.0",
            contract=contract,
        )

        class MultiSourceNode(MixinNodeIntrospection):
            def __init__(self) -> None:
                self.initialize_introspection(config)

        node = MultiSourceNode()
        event = await node.get_introspection_data()

        assert event.contract_capabilities is not None
        tags = event.contract_capabilities.capability_tags

        # Explicit tag
        assert "explicit.tag" in tags, f"Expected explicit.tag, got {tags}"
        # Node type inference
        assert "node.orchestrator" in tags, f"Expected node.orchestrator, got {tags}"
        # Protocol inference
        assert "event.bus" in tags, (
            f"Expected event.bus from ProtocolEventBus, got {tags}"
        )
        # Intent inference
        assert "kafka.messaging" in tags, (
            f"Expected kafka.messaging from kafka.publish, got {tags}"
        )


# =============================================================================
# Test Class: Caching Behavior
# =============================================================================


class TestCachingWithContractCapabilities:
    """Tests that caching doesn't affect contract_capabilities."""

    async def test_cache_hit_still_includes_contract_capabilities(self) -> None:
        """Verify contract_capabilities are preserved on cache hit."""
        contract = create_effect_contract(
            tags=["cached.capability"],
            protocol_interfaces=["ProtocolDatabaseAdapter"],
        )

        node = EffectNodeWithContract(node_id=TEST_NODE_UUID_1, contract=contract)

        # First call - cache miss
        data1 = await node.get_introspection_data()

        # Verify contract_capabilities present
        assert data1.contract_capabilities is not None
        assert "cached.capability" in data1.contract_capabilities.capability_tags

        # Second call - cache hit
        data2 = await node.get_introspection_data()

        # Verify contract_capabilities still present
        assert data2.contract_capabilities is not None, (
            "contract_capabilities should be present on cache hit"
        )
        assert "cached.capability" in data2.contract_capabilities.capability_tags, (
            "capability_tags should be preserved on cache hit"
        )

    async def test_cache_invalidation_reextracts_contract_capabilities(self) -> None:
        """Verify contract_capabilities are re-extracted after cache invalidation."""
        contract = create_effect_contract(
            tags=["will.be.reextracted"],
        )

        node = EffectNodeWithContract(node_id=TEST_NODE_UUID_1, contract=contract)

        # Populate cache
        await node.get_introspection_data()

        # Invalidate cache
        node.invalidate_introspection_cache()

        # Fresh call after invalidation
        data = await node.get_introspection_data()

        # Verify contract_capabilities still present
        assert data.contract_capabilities is not None, (
            "contract_capabilities should be re-extracted after cache invalidation"
        )
        assert "will.be.reextracted" in data.contract_capabilities.capability_tags


# =============================================================================
# Test Class: Version Extraction
# =============================================================================


class TestVersionExtraction:
    """Tests for contract version extraction."""

    async def test_version_extracted_from_contract(self) -> None:
        """Contract version should be extracted into contract_capabilities."""
        expected_version = ModelSemVer(major=2, minor=3, patch=4)
        contract = create_effect_contract(version=expected_version)

        node = EffectNodeWithContract(node_id=TEST_NODE_UUID_1, contract=contract)
        event = await node.get_introspection_data()

        assert event.contract_capabilities is not None
        assert event.contract_capabilities.contract_version.major == 2
        assert event.contract_capabilities.contract_version.minor == 3
        assert event.contract_capabilities.contract_version.patch == 4

    async def test_different_versions_extracted_correctly(self) -> None:
        """Different contract versions should be extracted correctly."""
        versions = [
            ModelSemVer(major=0, minor=1, patch=0),
            ModelSemVer(major=1, minor=0, patch=0),
            ModelSemVer(major=10, minor=20, patch=30),
        ]

        for version in versions:
            contract = create_effect_contract(version=version)
            node = EffectNodeWithContract(node_id=TEST_NODE_UUID_1, contract=contract)
            event = await node.get_introspection_data()

            assert event.contract_capabilities is not None
            assert event.contract_capabilities.contract_version.major == version.major
            assert event.contract_capabilities.contract_version.minor == version.minor
            assert event.contract_capabilities.contract_version.patch == version.patch
