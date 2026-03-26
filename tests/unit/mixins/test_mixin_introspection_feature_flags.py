# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for feature flag population in introspection events.

OMN-5577: Wire introspection to populate feature flags from contract.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.contracts.model_contract_base import ModelContractBase
from omnibase_core.models.contracts.model_contract_feature_flag import (
    ModelContractFeatureFlag,
)
from omnibase_core.models.core.model_feature_flags import ModelFeatureFlags
from omnibase_infra.mixins.mixin_node_introspection import MixinNodeIntrospection
from omnibase_infra.models.discovery import ModelIntrospectionConfig

pytestmark = [pytest.mark.unit]

TEST_NODE_UUID = UUID("00000000-0000-0000-0000-000000000099")


class _TestContract(ModelContractBase):
    """Concrete contract subclass for testing."""

    def validate_node_specific_config(self) -> None:
        pass


def _make_contract_with_flags(
    flags: list[ModelContractFeatureFlag],
) -> _TestContract:
    """Build a minimal contract with feature flags."""
    from omnibase_core.models.primitives.model_semver import ModelSemVer

    return _TestContract(
        name="test_node",
        contract_version=ModelSemVer(major=1, minor=0, patch=0),
        description="Test contract",
        node_type="EFFECT_GENERIC",
        input_model="dict",
        output_model="dict",
        feature_flags=flags,
    )


class _IntrospectableNode(MixinNodeIntrospection):
    """Minimal node class for testing introspection mixin."""

    def __init__(self, config: ModelIntrospectionConfig) -> None:
        self.initialize_introspection(config)


class TestIntrospectionPopulatesDeclaredFeatureFlags:
    """Verify that declared feature flags appear on the event."""

    @pytest.mark.asyncio
    async def test_introspection_populates_declared_feature_flags(self) -> None:
        """Node with contract flags -> event has declared defaults."""
        contract = _make_contract_with_flags(
            [
                ModelContractFeatureFlag(
                    name="enable_caching",
                    default_value=True,
                    category="infrastructure",
                ),
                ModelContractFeatureFlag(
                    name="debug_mode",
                    default_value=False,
                    category="observability",
                ),
            ]
        )

        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID,
            node_type=EnumNodeKind.EFFECT,
            node_name="test_node",
            contract=contract,
        )
        node = _IntrospectableNode(config)
        event = await node.get_introspection_data()

        declared_ff = event.declared_capabilities.feature_flags
        assert isinstance(declared_ff, ModelFeatureFlags)
        assert declared_ff.is_enabled("enable_caching") is True
        assert declared_ff.is_enabled("debug_mode") is False
        assert declared_ff.get_flag_count() == 2


class TestIntrospectionPopulatesResolvedFeatureFlags:
    """Verify that resolved flags apply env overrides."""

    @pytest.mark.asyncio
    async def test_introspection_populates_resolved_feature_flags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With env var set, resolved differs from defaults."""
        monkeypatch.setenv("ENABLE_CACHING", "false")

        contract = _make_contract_with_flags(
            [
                ModelContractFeatureFlag(
                    name="enable_caching",
                    default_value=True,
                    env_var="ENABLE_CACHING",
                    category="infrastructure",
                ),
                ModelContractFeatureFlag(
                    name="debug_mode",
                    default_value=False,
                    category="observability",
                ),
            ]
        )

        config = ModelIntrospectionConfig(
            node_id=TEST_NODE_UUID,
            node_type=EnumNodeKind.EFFECT,
            node_name="test_node",
            contract=contract,
        )
        node = _IntrospectableNode(config)
        event = await node.get_introspection_data()

        # Declared should have contract defaults
        declared_ff = event.declared_capabilities.feature_flags
        assert declared_ff.is_enabled("enable_caching") is True

        # Resolved should have env override
        resolved_ff = event.resolved_feature_flags
        assert isinstance(resolved_ff, ModelFeatureFlags)
        assert resolved_ff.is_enabled("enable_caching") is False
        assert resolved_ff.is_enabled("debug_mode") is False
