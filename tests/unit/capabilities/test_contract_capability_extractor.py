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

from omnibase_core.enums import EnumNodeType
from omnibase_core.models.capabilities import ModelContractCapabilities
from omnibase_core.models.contracts import ModelContractEffect, ModelIOOperationConfig
from omnibase_core.models.contracts.subcontracts.model_event_type_subcontract import (
    ModelEventTypeSubcontract,
)
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
    contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
    contract.dependencies = []
    contract.protocol_interfaces = []
    contract.tags = []
    return contract


@pytest.fixture
def minimal_compute_contract() -> MagicMock:
    """Create minimal COMPUTE_GENERIC contract mock."""
    contract = MagicMock()
    contract.node_type = MagicMock(value="COMPUTE_GENERIC")
    contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
    contract.dependencies = []
    contract.protocol_interfaces = []
    contract.tags = []
    return contract


@pytest.fixture
def minimal_reducer_contract() -> MagicMock:
    """Create minimal REDUCER_GENERIC contract mock."""
    contract = MagicMock()
    contract.node_type = MagicMock(value="REDUCER_GENERIC")
    contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
    contract.dependencies = []
    contract.protocol_interfaces = []
    contract.tags = []
    return contract


@pytest.fixture
def minimal_orchestrator_contract() -> MagicMock:
    """Create minimal ORCHESTRATOR_GENERIC contract mock."""
    contract = MagicMock()
    contract.node_type = MagicMock(value="ORCHESTRATOR_GENERIC")
    contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
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
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
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
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = []
        contract.tags = []

        result = extractor.extract(contract)

        assert result is not None
        assert result.contract_type == "reducer"
        assert result.contract_type.islower()

    def test_missing_node_type_raises_value_error(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Missing node_type should raise ValueError (fail-fast)."""
        contract = MagicMock(spec=[])
        # Add minimal required attributes without node_type
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = []
        contract.tags = []

        with pytest.raises(ValueError, match="Contract must have node_type field"):
            extractor.extract(contract)


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
        minimal_effect_contract.contract_version = expected_version

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert result.contract_version == expected_version
        assert result.contract_version.major == 2
        assert result.contract_version.minor == 3
        assert result.contract_version.patch == 4

    def test_missing_version_raises_value_error(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Missing version should raise ValueError (fail-fast)."""
        contract = MagicMock(spec=[])
        contract.node_type = MagicMock(value="EFFECT_GENERIC")
        contract.dependencies = []
        contract.protocol_interfaces = []
        contract.tags = []
        # No version attribute

        with pytest.raises(ValueError, match="Contract must have contract_version"):
            extractor.extract(contract)

    def test_non_semver_version_raises_value_error(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Non-ModelSemVer version should raise ValueError (fail-fast)."""
        minimal_effect_contract.contract_version = "1.2.3"  # String, not ModelSemVer

        with pytest.raises(ValueError, match="Contract must have contract_version"):
            extractor.extract(minimal_effect_contract)

    def test_version_none_raises_value_error(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """None version should raise ValueError (fail-fast)."""
        minimal_effect_contract.contract_version = None

        with pytest.raises(ValueError, match="Contract must have contract_version"):
            extractor.extract(minimal_effect_contract)

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
        minimal_effect_contract.contract_version = version_with_prerelease

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
# TestTagValidation - Tag format validation
# =============================================================================


class TestTagValidation:
    """Tests for tag format validation."""

    def test_empty_string_tags_filtered_out(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Empty string tags should be filtered out."""
        minimal_effect_contract.tags = ["valid.tag", "", "another.valid"]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "valid.tag" in result.capability_tags
        assert "another.valid" in result.capability_tags
        assert "" not in result.capability_tags

    def test_whitespace_only_tags_filtered_out(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Whitespace-only tags should be filtered out."""
        minimal_effect_contract.tags = ["valid.tag", "   ", "\t", "\n", "  \t\n  "]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "valid.tag" in result.capability_tags
        # Whitespace-only tags should not appear
        assert "   " not in result.capability_tags
        assert "\t" not in result.capability_tags
        assert "\n" not in result.capability_tags
        assert "  \t\n  " not in result.capability_tags

    def test_none_tags_filtered_out(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """None tags should be filtered out."""
        minimal_effect_contract.tags = ["valid.tag", None, "another.valid"]  # type: ignore[list-item]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "valid.tag" in result.capability_tags
        assert "another.valid" in result.capability_tags

    def test_valid_tags_pass_through(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Valid tags should pass through validation."""
        minimal_effect_contract.tags = [
            "simple",
            "with.dots",
            "with-dashes",
            "with_underscores",
            "MixedCase",
            "123numeric",
        ]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        assert "simple" in result.capability_tags
        assert "with.dots" in result.capability_tags
        assert "with-dashes" in result.capability_tags
        assert "with_underscores" in result.capability_tags
        assert "MixedCase" in result.capability_tags
        assert "123numeric" in result.capability_tags

    def test_mixed_valid_and_invalid_tags(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """Should filter invalid tags while keeping valid ones."""
        minimal_effect_contract.tags = [
            "valid.first",
            "",
            "valid.second",
            "   ",
            None,  # type: ignore[list-item]
            "valid.third",
            "\t\n",
        ]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        # Valid tags present
        assert "valid.first" in result.capability_tags
        assert "valid.second" in result.capability_tags
        assert "valid.third" in result.capability_tags
        # Invalid tags absent
        assert "" not in result.capability_tags
        assert "   " not in result.capability_tags
        assert "\t\n" not in result.capability_tags

    def test_all_invalid_tags_results_in_only_inferred_tags(
        self,
        extractor: ContractCapabilityExtractor,
        minimal_effect_contract: MagicMock,
    ) -> None:
        """If all explicit tags are invalid, result should have only inferred tags."""
        minimal_effect_contract.tags = ["", "   ", None]  # type: ignore[list-item]

        result = extractor.extract(minimal_effect_contract)

        assert result is not None
        # Should have inferred tag from node type
        assert "node.effect" in result.capability_tags
        # Invalid tags should not be present
        assert "" not in result.capability_tags
        assert "   " not in result.capability_tags


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
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
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
            contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
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
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
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
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
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
        contract1.contract_version = ModelSemVer(major=1, minor=0, patch=0)
        contract1.tags = ["tag1"]
        contract1.protocol_interfaces = ["Proto1"]
        contract1.dependencies = []

        contract2 = MagicMock()
        contract2.node_type = MagicMock(value="REDUCER_GENERIC")
        contract2.contract_version = ModelSemVer(major=2, minor=0, patch=0)
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
        contract.contract_version = ModelSemVer(major=0, minor=0, patch=0)
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
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
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


# =============================================================================
# TestExtractorWithCustomRules - Dependency injection tests
# =============================================================================


class TestExtractorWithCustomRules:
    """Tests for extractor with custom inference rules."""

    def test_extractor_accepts_custom_rules(self) -> None:
        """Extractor should accept custom CapabilityInferenceRules instance."""
        from omnibase_infra.capabilities import CapabilityInferenceRules

        custom_rules = CapabilityInferenceRules(
            intent_patterns={"redis.": "redis.caching"}
        )
        extractor = ContractCapabilityExtractor(rules=custom_rules)

        assert extractor._rules is custom_rules

    def test_extractor_uses_custom_rules_for_inference(self) -> None:
        """Extractor should use custom rules for tag inference."""
        from omnibase_infra.capabilities import CapabilityInferenceRules

        custom_rules = CapabilityInferenceRules(
            intent_patterns={"redis.": "redis.caching"}
        )
        extractor = ContractCapabilityExtractor(rules=custom_rules)

        # Create contract with redis intent
        contract = MagicMock()
        contract.node_type = MagicMock(value="EFFECT_GENERIC")
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = []
        contract.tags = []

        # Add consumed_events with redis pattern
        event = MagicMock()
        event.event_pattern = "redis.get"
        contract.consumed_events = [event]

        result = extractor.extract(contract)

        assert result is not None
        assert "redis.caching" in result.capability_tags

    def test_extractor_custom_rules_override_defaults(self) -> None:
        """Extractor should use overridden rules from custom instance."""
        from omnibase_infra.capabilities import CapabilityInferenceRules

        # Override postgres to custom tag
        custom_rules = CapabilityInferenceRules(
            intent_patterns={"postgres.": "custom.database"}
        )
        extractor = ContractCapabilityExtractor(rules=custom_rules)

        # Create contract with postgres intent
        contract = MagicMock()
        contract.node_type = MagicMock(value="EFFECT_GENERIC")
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = []
        contract.tags = []

        event = MagicMock()
        event.event_pattern = "postgres.upsert"
        contract.consumed_events = [event]

        result = extractor.extract(contract)

        assert result is not None
        assert "custom.database" in result.capability_tags
        assert "postgres.storage" not in result.capability_tags

    def test_extractor_none_rules_uses_defaults(self) -> None:
        """Extractor with rules=None should use default rules."""
        extractor = ContractCapabilityExtractor(rules=None)

        # Create contract with postgres intent
        contract = MagicMock()
        contract.node_type = MagicMock(value="EFFECT_GENERIC")
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = []
        contract.tags = []

        event = MagicMock()
        event.event_pattern = "postgres.upsert"
        contract.consumed_events = [event]

        result = extractor.extract(contract)

        assert result is not None
        # Should use default mapping
        assert "postgres.storage" in result.capability_tags

    def test_extractor_custom_protocol_inference(self) -> None:
        """Extractor should use custom protocol rules."""
        from omnibase_infra.capabilities import CapabilityInferenceRules

        custom_rules = CapabilityInferenceRules(
            protocol_tags={"ProtocolCustom": "custom.protocol"}
        )
        extractor = ContractCapabilityExtractor(rules=custom_rules)

        contract = MagicMock()
        contract.node_type = MagicMock(value="EFFECT_GENERIC")
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = ["ProtocolCustom"]
        contract.tags = []

        result = extractor.extract(contract)

        assert result is not None
        assert "custom.protocol" in result.capability_tags

    def test_extractor_custom_node_type_inference(self) -> None:
        """Extractor should use custom node type rules."""
        from omnibase_infra.capabilities import CapabilityInferenceRules

        custom_rules = CapabilityInferenceRules(
            node_type_tags={"effect": "custom.effect"}
        )
        extractor = ContractCapabilityExtractor(rules=custom_rules)

        contract = MagicMock()
        contract.node_type = MagicMock(value="EFFECT_GENERIC")
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = []
        contract.tags = []

        result = extractor.extract(contract)

        assert result is not None
        assert "custom.effect" in result.capability_tags
        assert "node.effect" not in result.capability_tags

    def test_extractor_custom_rules_combined(self) -> None:
        """Extractor should work with multiple custom rule types combined."""
        from omnibase_infra.capabilities import CapabilityInferenceRules

        custom_rules = CapabilityInferenceRules(
            intent_patterns={"redis.": "redis.caching"},
            protocol_tags={"ProtocolCustom": "custom.protocol"},
            node_type_tags={"gateway": "node.gateway"},
        )
        extractor = ContractCapabilityExtractor(rules=custom_rules)

        contract = MagicMock()
        contract.node_type = "gateway"  # Custom node type
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = ["ProtocolCustom"]
        contract.tags = []

        event = MagicMock()
        event.event_pattern = "redis.get"
        contract.consumed_events = [event]

        result = extractor.extract(contract)

        assert result is not None
        assert "redis.caching" in result.capability_tags
        assert "custom.protocol" in result.capability_tags
        assert "node.gateway" in result.capability_tags

    def test_extractor_isolates_rules_between_instances(self) -> None:
        """Different extractor instances with different rules should be isolated."""
        from omnibase_infra.capabilities import CapabilityInferenceRules

        custom_rules = CapabilityInferenceRules(
            intent_patterns={"redis.": "redis.caching"}
        )
        extractor1 = ContractCapabilityExtractor(rules=custom_rules)
        extractor2 = ContractCapabilityExtractor()  # Default rules

        # Create contract with redis intent
        contract = MagicMock()
        contract.node_type = MagicMock(value="EFFECT_GENERIC")
        contract.contract_version = ModelSemVer(major=1, minor=0, patch=0)
        contract.dependencies = []
        contract.protocol_interfaces = []
        contract.tags = []

        event = MagicMock()
        event.event_pattern = "redis.get"
        contract.consumed_events = [event]

        result1 = extractor1.extract(contract)
        result2 = extractor2.extract(contract)

        assert result1 is not None
        assert result2 is not None

        # extractor1 should have custom redis tag
        assert "redis.caching" in result1.capability_tags

        # extractor2 should NOT have redis tag (unrecognized pattern)
        assert "redis.caching" not in result2.capability_tags


# =============================================================================
# TestRealContractModels - Tests with actual contract models (not mocks)
# =============================================================================


class TestRealContractModels:
    """Tests using real contract models (not mocks) to verify extraction paths.

    These tests validate that the extraction logic works correctly with actual
    Pydantic models from omnibase_core, not just MagicMock objects. This ensures
    the extractor handles real model field access patterns correctly.
    """

    def test_effect_contract_event_type_extraction(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Verify event_type.primary_events extraction works with real ModelContractEffect.

        This test validates the extraction path that retrieves intent types from
        the event_type.primary_events field of an EFFECT contract. The mock-based
        test in TestIntentTypeExtraction verifies the logic works with mocks, but
        this test ensures the actual ModelContractEffect model structure is compatible.
        """
        # Create real event_type subcontract with primary_events
        event_type_config = ModelEventTypeSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            primary_events=["NodeRegistered", "NodeUpdated", "NodeDeleted"],
            event_categories=["node_lifecycle"],
            event_routing="direct",
        )

        # Create real ModelContractEffect with event_type populated
        contract = ModelContractEffect(
            name="test-effect-with-events",
            contract_version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test effect contract with event_type",
            node_type=EnumNodeType.EFFECT_GENERIC,
            input_model="object",
            output_model="object",
            event_type=event_type_config,
            io_operations=[
                ModelIOOperationConfig(
                    name="test_operation",
                    operation_type="read",
                    target="database",
                )
            ],
        )

        # Extract capabilities
        result = extractor.extract(contract)

        # Verify extraction succeeded
        assert result is not None, (
            "Extraction should succeed for real ModelContractEffect"
        )

        # Verify intent_types were extracted from event_type.primary_events
        assert "NodeRegistered" in result.intent_types, (
            f"Expected 'NodeRegistered' in intent_types, got {result.intent_types}"
        )
        assert "NodeUpdated" in result.intent_types, (
            f"Expected 'NodeUpdated' in intent_types, got {result.intent_types}"
        )
        assert "NodeDeleted" in result.intent_types, (
            f"Expected 'NodeDeleted' in intent_types, got {result.intent_types}"
        )

        # Verify intent_types are sorted (determinism requirement)
        assert result.intent_types == sorted(result.intent_types), (
            "intent_types should be sorted for deterministic output"
        )

        # Verify contract_type is correctly extracted
        assert result.contract_type == "effect", (
            f"Expected contract_type 'effect', got '{result.contract_type}'"
        )

    def test_effect_contract_without_event_type(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Verify extraction works when event_type is None (optional field).

        ModelContractEffect.event_type is optional. This test ensures the
        extractor handles the None case gracefully without errors.
        """
        # Create real ModelContractEffect without event_type
        contract = ModelContractEffect(
            name="test-effect-no-events",
            contract_version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test effect contract without event_type",
            node_type=EnumNodeType.EFFECT_GENERIC,
            input_model="object",
            output_model="object",
            event_type=None,  # Explicitly None
            io_operations=[
                ModelIOOperationConfig(
                    name="test_operation",
                    operation_type="read",
                    target="database",
                )
            ],
        )

        # Extract capabilities
        result = extractor.extract(contract)

        # Verify extraction succeeded
        assert result is not None, "Extraction should succeed when event_type is None"

        # Verify intent_types is empty (no events to extract)
        assert result.intent_types == [], (
            f"Expected empty intent_types when event_type is None, got {result.intent_types}"
        )

        # Verify contract_type is still correctly extracted
        assert result.contract_type == "effect"

    def test_effect_contract_with_single_primary_event(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Verify extraction handles single-element primary_events list.

        The primary_events field requires at least one event. This test
        ensures the extractor correctly handles the minimum valid case.
        """
        # Create event_type with single primary_event (minimum valid)
        event_type_config = ModelEventTypeSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            primary_events=["SingleEvent"],  # Minimum required
            event_categories=["single_category"],
            event_routing="direct",
        )

        contract = ModelContractEffect(
            name="test-effect-single-event",
            contract_version=ModelSemVer(major=1, minor=0, patch=0),
            description="Test effect contract with single primary_event",
            node_type=EnumNodeType.EFFECT_GENERIC,
            input_model="object",
            output_model="object",
            event_type=event_type_config,
            io_operations=[
                ModelIOOperationConfig(
                    name="test_operation",
                    operation_type="read",
                    target="database",
                )
            ],
        )

        # Extract capabilities
        result = extractor.extract(contract)

        # Verify extraction succeeded
        assert result is not None

        # Verify intent_types contains the single event
        assert result.intent_types == ["SingleEvent"], (
            f"Expected ['SingleEvent'], got {result.intent_types}"
        )

    def test_effect_contract_full_extraction(
        self,
        extractor: ContractCapabilityExtractor,
    ) -> None:
        """Verify full capability extraction from a complete ModelContractEffect.

        This test validates that all extraction paths work together:
        - contract_type from node_type
        - contract_version from contract_version
        - intent_types from event_type.primary_events
        - protocols from protocol_interfaces
        - capability_tags from explicit tags and inference
        """
        event_type_config = ModelEventTypeSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            primary_events=["DatabaseConnected", "DatabaseDisconnected"],
            event_categories=["database_lifecycle"],
            event_routing="direct",
        )

        contract = ModelContractEffect(
            name="test-complete-effect",
            contract_version=ModelSemVer(major=2, minor=3, patch=4),
            description="Complete test effect contract",
            node_type=EnumNodeType.EFFECT_GENERIC,
            input_model="object",
            output_model="object",
            event_type=event_type_config,
            tags=["custom.capability", "database.integration"],
            protocol_interfaces=["ProtocolDatabaseAdapter"],
            io_operations=[
                ModelIOOperationConfig(
                    name="query_operation",
                    operation_type="read",
                    target="postgres",
                )
            ],
        )

        result = extractor.extract(contract)

        # Verify all extraction paths
        assert result is not None

        # contract_type
        assert result.contract_type == "effect"

        # contract_version
        assert result.contract_version.major == 2
        assert result.contract_version.minor == 3
        assert result.contract_version.patch == 4

        # intent_types from event_type.primary_events
        assert "DatabaseConnected" in result.intent_types
        assert "DatabaseDisconnected" in result.intent_types

        # protocols from protocol_interfaces
        assert "ProtocolDatabaseAdapter" in result.protocols

        # capability_tags include explicit tags
        assert "custom.capability" in result.capability_tags
        assert "database.integration" in result.capability_tags

        # capability_tags include inferred tags
        assert "node.effect" in result.capability_tags  # from node_type
        assert (
            "database.adapter" in result.capability_tags
        )  # from ProtocolDatabaseAdapter
