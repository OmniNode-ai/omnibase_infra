# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for ModelAutoWiringManifest (OMN-8854).

Verifies that:
1. ModelAutoWiringManifest can aggregate topics from discovered contracts
2. all_subscribe_topics() alias works correctly (ProtocolAutoWiringManifestLike)
3. get_all_publish_topics() collects all publish topics correctly
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovery_error import (
    ModelDiscoveryError,
)

pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture
def sample_contracts():
    """Create sample discovered contracts for testing."""
    from pathlib import Path

    from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
        ModelContractVersion,
    )
    from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
        ModelEventBusWiring,
    )

    # Contract 1: subscribes to topic-a, publishes to topic-x
    contract1 = ModelDiscoveredContract(
        name="test_node_1",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/path1/contract.yaml"),
        entry_point_name="test_node_1",
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("topic-a",), publish_topics=("topic-x",)
        ),
    )

    # Contract 2: subscribes to topic-b and topic-c, publishes to topic-y
    contract2 = ModelDiscoveredContract(
        name="test_node_2",
        node_type="test_node_2",  # Different node_type for filtering test
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/path2/contract.yaml"),
        entry_point_name="test_node_2",
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("topic-b", "topic-c"), publish_topics=("topic-y",)
        ),
    )

    # Contract 3: no event bus
    contract3 = ModelDiscoveredContract(
        name="test_node_3",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/path3/contract.yaml"),
        entry_point_name="test_node_3",
        package_name="test-pkg",
        event_bus=None,
    )

    return (contract1, contract2, contract3)


@pytest.fixture
def sample_errors():
    """Create sample discovery errors for testing."""
    error1 = ModelDiscoveryError(
        entry_point_name="bad_node",
        package_name="test-bad-pkg",
        error="ImportError: Module not found",
    )
    return (error1,)


def test_manifest_instantiation(sample_contracts, sample_errors):
    """Test that ModelAutoWiringManifest can be instantiated."""
    manifest = ModelAutoWiringManifest(contracts=sample_contracts, errors=sample_errors)
    assert manifest is not None
    assert len(manifest.contracts) == 3
    assert len(manifest.errors) == 1


def test_manifest_total_discovered(sample_contracts):
    """Test that total_discovered property works."""
    manifest = ModelAutoWiringManifest(contracts=sample_contracts)
    assert manifest.total_discovered == 3


def test_manifest_total_errors(sample_errors):
    """Test that total_errors property works."""
    manifest = ModelAutoWiringManifest(errors=sample_errors)
    assert manifest.total_errors == 1


def test_manifest_get_by_node_type(sample_contracts):
    """Test that get_by_node_type filters correctly."""
    manifest = ModelAutoWiringManifest(contracts=sample_contracts)
    results = manifest.get_by_node_type("test_node_2")
    assert len(results) == 1
    assert results[0].node_type == "test_node_2"


def test_manifest_get_all_subscribe_topics(sample_contracts):
    """Test that get_all_subscribe_topics aggregates correctly."""
    manifest = ModelAutoWiringManifest(contracts=sample_contracts)
    topics = manifest.get_all_subscribe_topics()
    assert isinstance(topics, frozenset)
    assert topics == frozenset({"topic-a", "topic-b", "topic-c"})


def test_manifest_all_subscribe_topics_alias(sample_contracts):
    """Test that all_subscribe_topics() alias works (OMN-8854)."""
    manifest = ModelAutoWiringManifest(contracts=sample_contracts)
    # Both methods should return the same result
    topics_via_get = manifest.get_all_subscribe_topics()
    topics_via_alias = manifest.all_subscribe_topics()
    assert topics_via_get == topics_via_alias
    assert isinstance(topics_via_alias, frozenset)
    assert topics_via_alias == frozenset({"topic-a", "topic-b", "topic-c"})


def test_manifest_get_all_publish_topics(sample_contracts):
    """Test that get_all_publish_topics aggregates correctly."""
    manifest = ModelAutoWiringManifest(contracts=sample_contracts)
    topics = manifest.get_all_publish_topics()
    assert isinstance(topics, frozenset)
    assert topics == frozenset({"topic-x", "topic-y"})


def test_manifest_empty_contracts():
    """Test that manifest works with no contracts."""
    manifest = ModelAutoWiringManifest()
    assert manifest.total_discovered == 0
    assert manifest.total_errors == 0
    assert manifest.get_all_subscribe_topics() == frozenset()
    assert manifest.all_subscribe_topics() == frozenset()
    assert manifest.get_all_publish_topics() == frozenset()


def test_manifest_frozen():
    """Test that ModelAutoWiringManifest is immutable (frozen=True)."""
    manifest = ModelAutoWiringManifest()
    with pytest.raises(Exception):  # Pydantic raises ValidationError or similar
        manifest.contracts = ()  # type: ignore[misc]


def test_manifest_no_extra_fields():
    """Test that extra fields are forbidden."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        ModelAutoWiringManifest(
            contracts=(),
            errors=(),
            extra_field="not_allowed",  # type: ignore[call-arg]
        )
