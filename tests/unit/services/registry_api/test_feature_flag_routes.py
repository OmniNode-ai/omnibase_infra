# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for feature flag registry API endpoints (OMN-5579).

Tests aggregation logic for GET /registry/feature-flags including:
- Basic aggregation across projections
- Conflict detection (env_var, category disagreements)
- Mixed process values when nodes disagree
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.enums.enum_feature_flag_category import EnumFeatureFlagCategory
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection.model_projected_flag_meta import (
    ModelProjectedFlagMeta,
)
from omnibase_infra.models.projection.model_registration_projection import (
    ModelRegistrationProjection,
)
from omnibase_infra.services.registry_api.service import ServiceRegistryDiscovery


def _make_projection(
    *,
    feature_flags: dict[str, bool] | None = None,
    feature_flag_defaults: dict[str, bool] | None = None,
    feature_flag_metadata: dict[str, ModelProjectedFlagMeta] | None = None,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
) -> ModelRegistrationProjection:
    """Create a minimal projection for testing."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=uuid4(),
        current_state=state,
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer(major=1, minor=0, patch=0),
        last_applied_event_id=uuid4(),
        last_applied_offset=1,
        registered_at=now,
        updated_at=now,
        feature_flags=feature_flags or {},
        feature_flag_defaults=feature_flag_defaults or {},
        feature_flag_metadata=feature_flag_metadata or {},
    )


def _make_service(
    projections_by_state: dict[EnumRegistrationState, list[ModelRegistrationProjection]]
    | None = None,
) -> ServiceRegistryDiscovery:
    """Create a ServiceRegistryDiscovery with mocked projection reader."""
    container = MagicMock()
    reader = AsyncMock()

    async def mock_get_by_state(
        state: EnumRegistrationState,
        limit: int = 10000,
        correlation_id=None,
    ) -> list[ModelRegistrationProjection]:
        if projections_by_state is None:
            return []
        return projections_by_state.get(state, [])

    reader.get_by_state = mock_get_by_state

    return ServiceRegistryDiscovery(
        container=container,
        projection_reader=reader,
    )


@pytest.mark.unit
class TestGetAggregatedFlags:
    """Tests for get_aggregated_feature_flags service method."""

    @pytest.mark.asyncio
    async def test_get_aggregated_flags(self) -> None:
        """Aggregation returns correct structure for projections with flags."""
        proj = _make_projection(
            feature_flags={"enable_caching": True, "enable_debug": False},
            feature_flag_defaults={"enable_caching": True, "enable_debug": False},
            feature_flag_metadata={
                "enable_caching": ModelProjectedFlagMeta(
                    description="Enable response caching",
                    category=EnumFeatureFlagCategory.RUNTIME,
                    env_var="ENABLE_CACHING",
                    owner="platform-team",
                ),
                "enable_debug": ModelProjectedFlagMeta(
                    description="Enable debug mode",
                    category=EnumFeatureFlagCategory.GENERAL,
                    env_var="ENABLE_DEBUG",
                ),
            },
        )

        service = _make_service(
            projections_by_state={EnumRegistrationState.ACTIVE: [proj]}
        )

        with patch.dict("os.environ", {}, clear=False):
            flags, degraded = await service.get_aggregated_feature_flags()

        assert not degraded
        assert len(flags) == 2

        # Check structure of first flag
        caching_flag = next(f for f in flags if f.name == "enable_caching")
        assert caching_flag.default_value is True
        assert caching_flag.process_value is True
        assert caching_flag.env_var == "ENABLE_CACHING"
        assert caching_flag.category == EnumFeatureFlagCategory.RUNTIME
        assert caching_flag.conflict_status == "clean"
        assert caching_flag.declaring_nodes_count == 1
        assert caching_flag.state_alignment == "aligned"
        assert caching_flag.effective_value is None
        assert caching_flag.effective_value_status == "deferred"

    @pytest.mark.asyncio
    async def test_conflicting_declarations_surfaced(self) -> None:
        """Two projections with same flag but different env_var -> conflicted."""
        proj1 = _make_projection(
            feature_flags={"enable_feature_x": True},
            feature_flag_defaults={"enable_feature_x": True},
            feature_flag_metadata={
                "enable_feature_x": ModelProjectedFlagMeta(
                    env_var="FEATURE_X_V1",
                    category=EnumFeatureFlagCategory.GENERAL,
                ),
            },
        )
        proj2 = _make_projection(
            feature_flags={"enable_feature_x": True},
            feature_flag_defaults={"enable_feature_x": True},
            feature_flag_metadata={
                "enable_feature_x": ModelProjectedFlagMeta(
                    env_var="FEATURE_X_V2",
                    category=EnumFeatureFlagCategory.GENERAL,
                ),
            },
        )

        service = _make_service(
            projections_by_state={EnumRegistrationState.ACTIVE: [proj1, proj2]}
        )

        with patch.dict("os.environ", {}, clear=False):
            flags, degraded = await service.get_aggregated_feature_flags()

        assert not degraded
        assert len(flags) == 1
        flag = flags[0]
        assert flag.name == "enable_feature_x"
        assert flag.conflict_status == "conflicted"
        assert flag.conflict_details is not None
        assert any("env_var" in d for d in flag.conflict_details)
        assert flag.declaring_nodes_count == 2

    @pytest.mark.asyncio
    async def test_mixed_process_values(self) -> None:
        """Two projections disagree on flag value -> process_value is None."""
        proj1 = _make_projection(
            feature_flags={"enable_feature_y": True},
            feature_flag_defaults={"enable_feature_y": True},
            feature_flag_metadata={
                "enable_feature_y": ModelProjectedFlagMeta(
                    env_var="FEATURE_Y",
                ),
            },
        )
        proj2 = _make_projection(
            feature_flags={"enable_feature_y": False},
            feature_flag_defaults={"enable_feature_y": True},
            feature_flag_metadata={
                "enable_feature_y": ModelProjectedFlagMeta(
                    env_var="FEATURE_Y",
                ),
            },
        )

        service = _make_service(
            projections_by_state={EnumRegistrationState.ACTIVE: [proj1, proj2]}
        )

        with patch.dict("os.environ", {}, clear=False):
            flags, degraded = await service.get_aggregated_feature_flags()

        assert not degraded
        assert len(flags) == 1
        flag = flags[0]
        assert flag.name == "enable_feature_y"
        assert flag.process_value is None
        assert flag.conflict_status == "clean"  # value disagreement != env_var conflict

    @pytest.mark.asyncio
    async def test_no_projection_reader_returns_degraded(self) -> None:
        """Service without projection reader returns empty + degraded."""
        container = MagicMock()
        service = ServiceRegistryDiscovery(container=container)

        flags, degraded = await service.get_aggregated_feature_flags()

        assert flags == []
        assert degraded is True

    @pytest.mark.asyncio
    async def test_writable_when_infisical_configured(self) -> None:
        """Flags show writable=True when INFISICAL_ADDR is set."""
        proj = _make_projection(
            feature_flags={"my_flag": True},
            feature_flag_defaults={"my_flag": True},
            feature_flag_metadata={
                "my_flag": ModelProjectedFlagMeta(),
            },
        )
        service = _make_service(
            projections_by_state={EnumRegistrationState.ACTIVE: [proj]}
        )

        with patch.dict(
            "os.environ", {"INFISICAL_ADDR": "http://localhost:8880"}, clear=False
        ):
            flags, degraded = await service.get_aggregated_feature_flags()

        assert not degraded
        assert len(flags) == 1
        assert flags[0].writable is True
