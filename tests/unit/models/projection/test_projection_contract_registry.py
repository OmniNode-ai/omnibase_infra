# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for projection_contract_registry (OMN-11199)."""

from __future__ import annotations

import pytest

from omnibase_core.enums.enum_degraded_behavior import EnumDegradedBehavior
from omnibase_core.models.projection.model_projection_contract import (
    ModelProjectionContract,
)
from omnibase_infra.models.projection.projection_contract_registry import (
    CONTRACT_REGISTRY_PROJECTION,
    PROJECTION_CONTRACTS,
    REGISTRATION_PROJECTION,
    TOPIC_REGISTRY_PROJECTION,
    get_projection_contract,
)


@pytest.mark.unit
class TestProjectionContractRegistry:
    def test_all_contracts_present(self) -> None:
        names = {c.projection_name for c in PROJECTION_CONTRACTS}
        assert "contract_registry" in names
        assert "topic_registry" in names
        assert "registration" in names

    def test_all_contracts_are_model_projection_contract(self) -> None:
        for contract in PROJECTION_CONTRACTS:
            assert isinstance(contract, ModelProjectionContract)

    def test_contract_registry_fields(self) -> None:
        c = CONTRACT_REGISTRY_PROJECTION
        assert c.projection_name == "contract_registry"
        assert c.freshness_field == "last_seen_at"
        assert c.freshness_source_table == "contracts"
        assert c.freshness_sla_seconds == 30
        assert c.degraded_semantics == EnumDegradedBehavior.SERVE_STALE
        assert c.cursor.cursor_type == "kafka_offset"
        assert c.cursor.supports_replay is True
        assert len(c.source_topics) > 0

    def test_topic_registry_fields(self) -> None:
        c = TOPIC_REGISTRY_PROJECTION
        assert c.projection_name == "topic_registry"
        assert c.freshness_field == "last_seen_at"
        assert c.freshness_source_table == "topics"
        assert c.freshness_sla_seconds == 30
        assert c.degraded_semantics == EnumDegradedBehavior.SERVE_STALE
        assert c.cursor.cursor_type == "kafka_offset"
        assert c.cursor.supports_replay is True

    def test_registration_fields(self) -> None:
        c = REGISTRATION_PROJECTION
        assert c.projection_name == "registration"
        assert c.freshness_field == "updated_at"
        assert c.freshness_source_table == "registration_projections"
        assert c.freshness_sla_seconds == 60
        assert c.degraded_semantics == EnumDegradedBehavior.SERVE_STALE
        assert c.cursor.cursor_type == "kafka_offset"
        assert c.cursor.supports_replay is True

    def test_get_projection_contract_found(self) -> None:
        assert (
            get_projection_contract("contract_registry") is CONTRACT_REGISTRY_PROJECTION
        )
        assert get_projection_contract("topic_registry") is TOPIC_REGISTRY_PROJECTION
        assert get_projection_contract("registration") is REGISTRATION_PROJECTION

    def test_get_projection_contract_not_found(self) -> None:
        assert get_projection_contract("nonexistent") is None

    def test_all_contracts_have_source_topics(self) -> None:
        for contract in PROJECTION_CONTRACTS:
            assert len(contract.source_topics) > 0, (
                f"{contract.projection_name} has no source_topics"
            )

    def test_all_contracts_have_schema_model(self) -> None:
        for contract in PROJECTION_CONTRACTS:
            assert contract.schema_model, (
                f"{contract.projection_name} missing schema_model"
            )

    def test_all_contracts_have_positive_sla(self) -> None:
        for contract in PROJECTION_CONTRACTS:
            assert contract.freshness_sla_seconds > 0, (
                f"{contract.projection_name} has non-positive SLA"
            )
