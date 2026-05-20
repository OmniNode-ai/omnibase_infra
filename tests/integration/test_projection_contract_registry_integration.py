# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for projection contract registry (OMN-11199).

Validates that declared projection contracts reference valid source topics
and that all contracts are internally consistent when loaded together.
"""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_degraded_behavior import EnumDegradedBehavior
from omnibase_infra.enums.generated.enum_platform_topic import EnumPlatformTopic
from omnibase_infra.models.projection.projection_contract_registry import (
    PROJECTION_CONTRACTS,
    get_projection_contract,
)


@pytest.mark.integration
class TestProjectionContractRegistryIntegration:
    """Verify projection contracts are consistent with the platform topic enum."""

    def test_all_source_topics_are_valid_platform_topics(self) -> None:
        """Every source_topic in a projection contract must exist in EnumPlatformTopic."""
        valid_topics = {e.value for e in EnumPlatformTopic}
        for contract in PROJECTION_CONTRACTS:
            for topic in contract.source_topics:
                assert topic in valid_topics, (
                    f"Projection '{contract.projection_name}' references unknown topic '{topic}'"
                )

    def test_no_duplicate_projection_names(self) -> None:
        """Each projection_name must be unique across the registry."""
        names = [c.projection_name for c in PROJECTION_CONTRACTS]
        assert len(names) == len(set(names)), (
            f"Duplicate projection names: {[n for n in names if names.count(n) > 1]}"
        )

    def test_get_projection_contract_round_trips_all_entries(self) -> None:
        """get_projection_contract must resolve every registered contract."""
        for contract in PROJECTION_CONTRACTS:
            resolved = get_projection_contract(contract.projection_name)
            assert resolved is contract, (
                f"get_projection_contract('{contract.projection_name}') returned wrong contract"
            )

    def test_degraded_semantics_are_valid_enum_values(self) -> None:
        """All contracts must use a valid EnumDegradedBehavior variant."""
        for contract in PROJECTION_CONTRACTS:
            assert isinstance(contract.degraded_semantics, EnumDegradedBehavior), (
                f"Projection '{contract.projection_name}' has invalid degraded_semantics"
            )

    def test_freshness_fields_are_non_empty(self) -> None:
        """Freshness metadata must be fully specified."""
        for contract in PROJECTION_CONTRACTS:
            assert contract.freshness_field, (
                f"Projection '{contract.projection_name}' missing freshness_field"
            )
            assert contract.freshness_source_table, (
                f"Projection '{contract.projection_name}' missing freshness_source_table"
            )
