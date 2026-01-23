# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ContractCapabilityExtractor.

Tests the contract-based capability extraction with fixtures
for each node type. Validates deterministic output, graceful degradation,
and correct extraction from various contract structures.

OMN-1136: ContractCapabilityExtractor unit test coverage.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omnibase_core.models.capabilities import ModelContractCapabilities
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.capabilities import ContractCapabilityExtractor

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor() -> ContractCapabilityExtractor:
    """Provide a fresh extractor instance for each test."""
    return ContractCapabilityExtractor()


@pytest.fixture
def minimal_effect_contract() -> MagicMock:
    """Create minimal EFFECT_GENERIC contract mock."""
    contract = MagicMock()
    contract.node_type = MagicMock(value="EFFECT_GENERIC")
    contract.version = ModelSemVer(major=1, minor=0, patch=0)
    contract.dependencies = []
    contract.protocol_interfaces = []
    contract.tags = []
    return contract


@pytest.fixture
def minimal_compute_contract() -> MagicMock:
    """Create minimal COMPUTE_GENERIC contract mock."""
    contract = MagicMock()
    contract.node_type = MagicMock(value="COMPUTE_GENERIC")
    contract.version = ModelSemVer(major=1, minor=0, patch=0)
    contract.dependencies = []
    contract.protocol_interfaces = []
    contract.tags = []
    return contract


@pytest.fixture
def minimal_reducer_contract() -> MagicMock:
    """Create minimal REDUCER_GENERIC contract mock."""
    contract = MagicMock()
    contract.node_type = MagicMock(value="REDUCER_GENERIC")
    contract.version = ModelSemVer(major=1, minor=0, patch=0)
    contract.dependencies = []
    contract.protocol_interfaces = []
    contract.tags = []
    return contract


@pytest.fixture
def minimal_orchestrator_contract() -> MagicMock:
    """Create minimal ORCHESTRATOR_GENERIC contract mock."""
    contract = MagicMock()
    contract.node_type = MagicMock(value="ORCHESTRATOR_GENERIC")
    contract.version = ModelSemVer(major=1, minor=0, patch=0)
    contract.dependencies = []
    contract.protocol_interfaces = []
    contract.tags = []
    return contract


# =============================================================================
# TestExtractBasics - Core extraction functionality
# =============================================================================


class TestExtractBasics:
    """Basic extraction tests."""

    def test_extract_returns_model_contract_capabilities(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """extract() should return ModelContractCapabilities instance."""
        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert isinstance(result, ModelContractCapabilities)

    def test_extract_none_contract(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """extract(None) should return None."""
        result = extractor.extract(None)  # type: ignore[arg-type]
        assert result is None


# =============================================================================
# TestContractTypeExtraction - Node type parsing
# =============================================================================


class TestContractTypeExtraction:
    """Tests for contract type extraction."""

    def test_effect_generic_type(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """EFFECT_GENERIC should extract as 'effect'."""
        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert result.contract_type == "effect"

    def test_compute_generic_type(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_compute_contract: MagicMock,
    ) -> None:
        """COMPUTE_GENERIC should extract as 'compute'."""
        result = extractor.extract(minimal_compute_contract)

        assert result is not None
        assert result.contract_type == "compute"

    def test_reducer_generic_type(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_reducer_contract: MagicMock,
    ) -> None:
        """REDUCER_GENERIC should extract as 'reducer'."""
        result = extractor.extract(minimal_reducer_contract)

        assert result is not None
        assert result.contract_type == "reducer"

    def test_orchestrator_generic_type(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_orchestrator_contract: MagicMock,
    ) -> None:
        """ORCHESTRATOR_GENERIC should extract as 'orchestrator'."""
        result = extractor.extract(minimal_orchestrator_contract)

        assert result is not None
        assert result.contract_type == "orchestrator"

    def test_node_type_string_value(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Should handle node_type as plain string (no .value attr)."""
        contract = MagicMock()
        # String without .value attribute
        contract.node_type = "EFFECT_GENERIC"
        contract.version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = []
        contract.tags = []

        result = extractor.extract(contract)

        assert result is not None
        assert result.contract_type == "effect"

    def test_node_type_lowercase_handling(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Should normalize case to lowercase."""
        contract = MagicMock()
        contract.node_type = MagicMock(value="REDUCER_GENERIC")
        contract.version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = []
        contract.tags = []

        result = extractor.extract(contract)

        assert result is not None
        assert result.contract_type == "reducer"
        assert result.contract_type.islower()

    def test_missing_node_type_returns_unknown(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Missing node_type should default to 'unknown'."""
        contract = MagicMock(spec=[])
        # Add minimal required attributes without node_type
        contract.version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = []
        contract.tags = []

        result = extractor.extract(contract)

        assert result is not None
        assert result.contract_type == "unknown"


# =============================================================================
# TestVersionExtraction - Semantic version extraction
# =============================================================================


class TestVersionExtraction:
    """Tests for version extraction."""

    def test_extracts_semver(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should extract ModelSemVer from contract."""
        expected_version = ModelSemVer(major=2, minor=3, patch=4)
        minimal_effect_contract.version = expected_version

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert result.contract_version == expected_version
        assert result.contract_version.major == 2
        assert result.contract_version.minor == 3
        assert result.contract_version.patch == 4

    def test_missing_version_defaults_to_0_0_0(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Missing version should default to 0.0.0."""
        contract = MagicMock(spec=[])
        contract.node_type = MagicMock(value="EFFECT_GENERIC")
        contract.dependencies = []
        contract.protocol_interfaces = []
        contract.tags = []
        # No version attribute

        result = extractor.extract(contract)

        assert result is not None
        assert result.contract_version.major == 0
        assert result.contract_version.minor == 0
        assert result.contract_version.patch == 0

    def test_non_semver_version_defaults_to_0_0_0(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Non-ModelSemVer version should default to 0.0.0."""
        minimal_effect_contract.version = "1.2.3"  # String, not ModelSemVer

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert result.contract_version.major == 0
        assert result.contract_version.minor == 0
        assert result.contract_version.patch == 0

    def test_version_none_defaults_to_0_0_0(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """None version should default to 0.0.0."""
        minimal_effect_contract.version = None

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert result.contract_version == ModelSemVer(major=0, minor=0, patch=0)

    def test_version_prerelease_preserved(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Prerelease version info should be preserved."""
        # prerelease is a tuple of identifiers per SemVer 2.0.0 spec
        version_with_prerelease = ModelSemVer(
            major=1, minor=0, patch=0, prerelease=("alpha", 1)
        )
        minimal_effect_contract.version = version_with_prerelease

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert result.contract_version.prerelease == ("alpha", 1)


# =============================================================================
# TestProtocolExtraction - Protocol interface extraction
# =============================================================================


class TestProtocolExtraction:
    """Tests for protocol extraction from dependencies."""

    def test_extracts_protocol_interfaces(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should extract from protocol_interfaces field."""
        minimal_effect_contract.protocol_interfaces = [
            "ProtocolDatabaseAdapter",
            "ProtocolEventBus",
        ]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "ProtocolDatabaseAdapter" in result.protocols
        assert "ProtocolEventBus" in result.protocols

    def test_extracts_from_protocol_dependencies_via_is_protocol(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_reducer_contract: MagicMock,
    ) -> None:
        """Should extract protocol names from dependencies using is_protocol()."""
        # Mock dependency that is a protocol (via is_protocol method)
        protocol_dep = MagicMock()
        protocol_dep.name = "ProtocolReducer"
        protocol_dep.is_protocol = MagicMock(return_value=True)

        minimal_reducer_contract.dependencies = [protocol_dep]

        result = extractor.extract(minimal_reducer_contract)

        assert result is not None
        assert "ProtocolReducer" in result.protocols

    def test_extracts_from_protocol_dependencies_via_dependency_type(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should extract protocols using dependency_type field."""
        # Mock dependency with dependency_type enum
        dep = MagicMock()
        dep.name = "ProtocolCacheAdapter"
        dep.is_protocol = MagicMock(return_value=False)  # Not via is_protocol
        dep.dependency_type = MagicMock(value="PROTOCOL")  # Via dependency_type

        minimal_effect_contract.dependencies = [dep]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "ProtocolCacheAdapter" in result.protocols

    def test_extracts_from_protocol_dependencies_via_string_dependency_type(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should extract protocols when dependency_type is string 'PROTOCOL'."""
        dep = MagicMock()
        dep.name = "ProtocolServiceDiscovery"
        dep.is_protocol = MagicMock(return_value=False)
        dep.dependency_type = "PROTOCOL"  # String, not enum

        minimal_effect_contract.dependencies = [dep]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "ProtocolServiceDiscovery" in result.protocols

    def test_protocols_sorted_deduped(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Protocols should be sorted and deduplicated."""
        minimal_effect_contract.protocol_interfaces = [
            "ProtocolZ",
            "ProtocolA",
            "ProtocolM",
            "ProtocolA",  # Duplicate
            "ProtocolZ",  # Duplicate
        ]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        # Check sorted order
        assert result.protocols == ["ProtocolA", "ProtocolM", "ProtocolZ"]
        # Check no duplicates
        assert len(result.protocols) == 3

    def test_empty_protocol_interfaces(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Empty protocol_interfaces should result in empty protocols list."""
        minimal_effect_contract.protocol_interfaces = []
        minimal_effect_contract.dependencies = []

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert result.protocols == []

    def test_combines_interfaces_and_dependencies(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should combine protocols from interfaces and dependencies."""
        minimal_effect_contract.protocol_interfaces = ["ProtocolA"]

        dep = MagicMock()
        dep.name = "ProtocolB"
        dep.is_protocol = MagicMock(return_value=True)
        minimal_effect_contract.dependencies = [dep]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "ProtocolA" in result.protocols
        assert "ProtocolB" in result.protocols

    def test_skips_non_protocol_dependencies(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should skip dependencies that are not protocols."""
        non_protocol_dep = MagicMock()
        non_protocol_dep.name = "SomeService"
        non_protocol_dep.is_protocol = MagicMock(return_value=False)
        non_protocol_dep.dependency_type = MagicMock(value="SERVICE")

        minimal_effect_contract.dependencies = [non_protocol_dep]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "SomeService" not in result.protocols


# =============================================================================
# TestIntentTypeExtraction - Intent type extraction
# =============================================================================


class TestIntentTypeExtraction:
    """Tests for intent type extraction."""

    def test_extracts_from_effect_event_type_primary_events(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Effect contracts should extract from event_type.primary_events."""
        event_type = MagicMock()
        event_type.primary_events = ["NodeRegistered", "NodeUpdated"]
        minimal_effect_contract.event_type = event_type

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "NodeRegistered" in result.intent_types
        assert "NodeUpdated" in result.intent_types

    def test_extracts_from_orchestrator_consumed_events(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_orchestrator_contract: MagicMock,
    ) -> None:
        """Orchestrator contracts should extract from consumed_events."""
        event1 = MagicMock()
        event1.event_pattern = "consul.register"
        event2 = MagicMock()
        event2.event_pattern = "postgres.upsert"

        minimal_orchestrator_contract.consumed_events = [event1, event2]

        result = extractor.extract(minimal_orchestrator_contract)

        assert result is not None
        assert "consul.register" in result.intent_types
        assert "postgres.upsert" in result.intent_types

    def test_extracts_from_orchestrator_published_events(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_orchestrator_contract: MagicMock,
    ) -> None:
        """Orchestrator contracts should extract from published_events."""
        event1 = MagicMock()
        event1.event_name = "RegistrationCompleted"
        event2 = MagicMock()
        event2.event_name = "RegistrationFailed"

        minimal_orchestrator_contract.published_events = [event1, event2]

        result = extractor.extract(minimal_orchestrator_contract)

        assert result is not None
        assert "RegistrationCompleted" in result.intent_types
        assert "RegistrationFailed" in result.intent_types

    def test_extracts_from_reducer_aggregation_functions(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_reducer_contract: MagicMock,
    ) -> None:
        """Reducer contracts should extract from aggregation.aggregation_functions."""
        func1 = MagicMock()
        func1.output_field = "total_count"
        func2 = MagicMock()
        func2.output_field = "average_value"

        aggregation = MagicMock()
        aggregation.aggregation_functions = [func1, func2]
        minimal_reducer_contract.aggregation = aggregation

        result = extractor.extract(minimal_reducer_contract)

        assert result is not None
        assert "aggregate.total_count" in result.intent_types
        assert "aggregate.average_value" in result.intent_types

    def test_intent_types_sorted_deduped(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_orchestrator_contract: MagicMock,
    ) -> None:
        """Intent types should be sorted and deduplicated."""
        event1 = MagicMock()
        event1.event_pattern = "z.event"
        event2 = MagicMock()
        event2.event_pattern = "a.event"
        event3 = MagicMock()
        event3.event_pattern = "z.event"  # Duplicate

        minimal_orchestrator_contract.consumed_events = [event1, event2, event3]

        result = extractor.extract(minimal_orchestrator_contract)

        assert result is not None
        assert result.intent_types == ["a.event", "z.event"]

    def test_empty_intent_types(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Contract without intent sources should have empty intent_types."""
        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert result.intent_types == []

    def test_skips_empty_event_patterns(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_orchestrator_contract: MagicMock,
    ) -> None:
        """Should skip events with empty/None event_pattern."""
        event1 = MagicMock()
        event1.event_pattern = "valid.event"
        event2 = MagicMock()
        event2.event_pattern = ""  # Empty
        event3 = MagicMock()
        event3.event_pattern = None  # None

        minimal_orchestrator_contract.consumed_events = [event1, event2, event3]

        result = extractor.extract(minimal_orchestrator_contract)

        assert result is not None
        assert result.intent_types == ["valid.event"]


# =============================================================================
# TestExplicitTagExtraction - Tag extraction from contract
# =============================================================================


class TestExplicitTagExtraction:
    """Tests for explicit capability tag extraction."""

    def test_extracts_from_tags_field(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should extract from contract.tags field."""
        minimal_effect_contract.tags = ["custom.tag", "another.tag"]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "custom.tag" in result.capability_tags
        assert "another.tag" in result.capability_tags

    def test_empty_tags_field(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Empty tags should still produce capability_tags (from inference)."""
        minimal_effect_contract.tags = []

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        # Should have at least the node type tag from inference
        assert "node.effect" in result.capability_tags

    def test_missing_tags_field(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Missing tags field should not cause error."""
        contract = MagicMock(spec=[])
        contract.node_type = MagicMock(value="EFFECT_GENERIC")
        contract.version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = []
        # No tags attribute

        result = extractor.extract(contract)

        assert result is not None
        # Should have inferred tags at minimum
        assert "node.effect" in result.capability_tags


# =============================================================================
# TestTagUnion - Explicit + inferred tag union
# =============================================================================


class TestTagUnion:
    """Tests for explicit + inferred tag union."""

    def test_unions_explicit_and_inferred_tags(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_reducer_contract: MagicMock,
    ) -> None:
        """Should union explicit tags with inferred tags."""
        # Add explicit tag
        minimal_reducer_contract.tags = ["explicit.tag"]

        # Add consumed_events to trigger postgres inference
        event = MagicMock()
        event.event_pattern = "postgres.upsert"
        minimal_reducer_contract.consumed_events = [event]

        result = extractor.extract(minimal_reducer_contract)

        assert result is not None
        # Should have explicit tag
        assert "explicit.tag" in result.capability_tags
        # Should have inferred tag from postgres intent
        assert "postgres.storage" in result.capability_tags
        # Should have node type tag
        assert "node.reducer" in result.capability_tags

    def test_union_is_deduplicated(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Union should not have duplicates."""
        # Explicit tag matches what would be inferred
        minimal_effect_contract.tags = ["node.effect", "node.effect"]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        # Should only appear once
        assert result.capability_tags.count("node.effect") == 1

    def test_inferred_from_protocols(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should infer tags from protocol names."""
        minimal_effect_contract.protocol_interfaces = ["ProtocolDatabaseAdapter"]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "database.adapter" in result.capability_tags

    def test_inferred_from_intent_types(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_orchestrator_contract: MagicMock,
    ) -> None:
        """Should infer tags from intent type patterns."""
        event = MagicMock()
        event.event_pattern = "kafka.publish"
        minimal_orchestrator_contract.consumed_events = [event]

        result = extractor.extract(minimal_orchestrator_contract)

        assert result is not None
        assert "kafka.messaging" in result.capability_tags

    def test_all_inference_patterns(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_orchestrator_contract: MagicMock,
    ) -> None:
        """Test all known inference patterns produce correct tags."""
        # Add events for multiple patterns
        events = []
        for pattern in [
            "postgres.query",
            "consul.register",
            "kafka.send",
            "vault.read",
        ]:
            event = MagicMock()
            event.event_pattern = pattern
            events.append(event)
        minimal_orchestrator_contract.consumed_events = events

        result = extractor.extract(minimal_orchestrator_contract)

        assert result is not None
        assert "postgres.storage" in result.capability_tags
        assert "consul.registration" in result.capability_tags
        assert "kafka.messaging" in result.capability_tags
        assert "vault.secrets" in result.capability_tags


# =============================================================================
# TestDeterminism - Output consistency
# =============================================================================


class TestDeterminism:
    """Tests for deterministic output."""

    def test_same_input_same_output(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Same contract should always produce same output."""

        def make_contract() -> MagicMock:
            contract = MagicMock()
            contract.node_type = MagicMock(value="EFFECT_GENERIC")
            contract.version = ModelSemVer(major=1, minor=0, patch=0)
            contract.tags = ["z.tag", "a.tag", "m.tag"]
            contract.dependencies = []
            contract.protocol_interfaces = ["ProtocolZ", "ProtocolA"]
            return contract

        result1 = extractor.extract(make_contract())
        result2 = extractor.extract(make_contract())

        assert result1 is not None
        assert result2 is not None
        assert result1.capability_tags == result2.capability_tags
        assert result1.protocols == result2.protocols
        assert result1.intent_types == result2.intent_types
        assert result1.contract_type == result2.contract_type

    def test_output_always_sorted_capability_tags(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """capability_tags must be sorted."""
        contract = MagicMock()
        contract.node_type = MagicMock(value="EFFECT_GENERIC")
        contract.version = ModelSemVer(major=1, minor=0, patch=0)
        contract.tags = ["zebra", "apple", "mango"]
        contract.protocol_interfaces = []
        contract.dependencies = []

        result = extractor.extract(contract)

        assert result is not None
        assert result.capability_tags == sorted(result.capability_tags)

    def test_output_always_sorted_protocols(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """protocols must be sorted."""
        contract = MagicMock()
        contract.node_type = MagicMock(value="EFFECT_GENERIC")
        contract.version = ModelSemVer(major=1, minor=0, patch=0)
        contract.tags = []
        contract.protocol_interfaces = ["ZProtocol", "AProtocol", "MProtocol"]
        contract.dependencies = []

        result = extractor.extract(contract)

        assert result is not None
        assert result.protocols == sorted(result.protocols)

    def test_output_always_sorted_intent_types(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_orchestrator_contract: MagicMock,
    ) -> None:
        """intent_types must be sorted."""
        events = []
        for pattern in ["z.event", "a.event", "m.event"]:
            event = MagicMock()
            event.event_pattern = pattern
            events.append(event)
        minimal_orchestrator_contract.consumed_events = events

        result = extractor.extract(minimal_orchestrator_contract)

        assert result is not None
        assert result.intent_types == sorted(result.intent_types)

    def test_multiple_extractions_independent(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Multiple extractions should not affect each other."""
        contract1 = MagicMock()
        contract1.node_type = MagicMock(value="EFFECT_GENERIC")
        contract1.version = ModelSemVer(major=1, minor=0, patch=0)
        contract1.tags = ["tag1"]
        contract1.protocol_interfaces = ["Proto1"]
        contract1.dependencies = []

        contract2 = MagicMock()
        contract2.node_type = MagicMock(value="REDUCER_GENERIC")
        contract2.version = ModelSemVer(major=2, minor=0, patch=0)
        contract2.tags = ["tag2"]
        contract2.protocol_interfaces = ["Proto2"]
        contract2.dependencies = []

        result1 = extractor.extract(contract1)
        result2 = extractor.extract(contract2)
        result1_again = extractor.extract(contract1)

        assert result1 is not None
        assert result2 is not None
        assert result1_again is not None

        # Results should be independent
        assert result1.contract_type != result2.contract_type
        assert result1.capability_tags == result1_again.capability_tags
        assert result1.protocols == result1_again.protocols


# =============================================================================
# TestEdgeCases - Boundary conditions and unusual inputs
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_contract(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Contract with all empty fields should still work."""
        contract = MagicMock()
        contract.node_type = MagicMock(value="COMPUTE_GENERIC")
        contract.version = ModelSemVer(major=0, minor=0, patch=0)
        contract.tags = []
        contract.protocol_interfaces = []
        contract.dependencies = []

        result = extractor.extract(contract)

        assert result is not None
        assert result.contract_type == "compute"
        assert result.capability_tags == ["node.compute"]

    def test_none_values_in_lists(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should handle None values in lists gracefully."""
        minimal_effect_contract.tags = ["valid", None, "also_valid"]  # type: ignore[list-item]
        minimal_effect_contract.protocol_interfaces = [None, "ProtocolA"]  # type: ignore[list-item]

        # May fail or handle gracefully - depends on implementation
        result = extractor.extract(minimal_effect_contract)

        # If it succeeds, verify valid items are present
        if result is not None:
            assert (
                "valid" in result.capability_tags
                or "also_valid" in result.capability_tags
            )

    def test_special_characters_in_tags(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should handle special characters in tags."""
        minimal_effect_contract.tags = [
            "tag-with-dashes",
            "tag_with_underscores",
            "tag.with.dots",
        ]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "tag-with-dashes" in result.capability_tags
        assert "tag_with_underscores" in result.capability_tags
        assert "tag.with.dots" in result.capability_tags

    def test_very_long_tag_name(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should handle very long tag names."""
        long_tag = "a" * 1000
        minimal_effect_contract.tags = [long_tag]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert long_tag in result.capability_tags

    def test_unicode_in_tags(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should handle unicode characters in tags."""
        minimal_effect_contract.tags = ["tag_with_unicode_\u2603"]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "tag_with_unicode_\u2603" in result.capability_tags

    def test_unknown_node_type_value(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Should handle unknown node type values."""
        contract = MagicMock()
        contract.node_type = MagicMock(value="UNKNOWN_TYPE")
        contract.version = ModelSemVer(major=1, minor=0, patch=0)
        contract.tags = []
        contract.protocol_interfaces = []
        contract.dependencies = []

        result = extractor.extract(contract)

        assert result is not None
        assert result.contract_type == "unknown_type"
        # Should not have node type tag since type is unknown
        assert "node.unknown_type" not in result.capability_tags


# =============================================================================
# TestModelContractCapabilitiesOutput - Output model validation
# =============================================================================


class TestModelContractCapabilitiesOutput:
    """Tests for the output ModelContractCapabilities structure."""

    def test_output_is_frozen(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Output model should be frozen (immutable)."""
        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        # ModelContractCapabilities is frozen, so this should raise
        with pytest.raises(Exception):  # ValidationError or AttributeError
            result.contract_type = "modified"  # type: ignore[misc]

    def test_output_has_all_required_fields(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Output should have all required fields."""
        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert hasattr(result, "contract_type")
        assert hasattr(result, "contract_version")
        assert hasattr(result, "intent_types")
        assert hasattr(result, "protocols")
        assert hasattr(result, "capability_tags")

    def test_output_lists_are_not_none(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """List fields should never be None, always lists."""
        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert result.intent_types is not None
        assert isinstance(result.intent_types, list)
        assert result.protocols is not None
        assert isinstance(result.protocols, list)
        assert result.capability_tags is not None
        assert isinstance(result.capability_tags, list)


# =============================================================================
# TestExtractorInstantiation - Constructor tests
# =============================================================================


class TestExtractorInstantiation:
    """Tests for extractor instantiation."""

    def test_can_create_extractor(self) -> None:
        """Should be able to create an extractor instance."""
        extractor = ContractCapabilityExtractor()
        assert extractor is not None

    def test_extractor_has_rules(self) -> None:
        """Extractor should have rules engine."""
        extractor = ContractCapabilityExtractor()
        assert hasattr(extractor, "_rules")
        assert extractor._rules is not None

    def test_multiple_extractors_independent(self) -> None:
        """Multiple extractor instances should be independent."""
        extractor1 = ContractCapabilityExtractor()
        extractor2 = ContractCapabilityExtractor()

        assert extractor1 is not extractor2
        assert extractor1._rules is not extractor2._rules
