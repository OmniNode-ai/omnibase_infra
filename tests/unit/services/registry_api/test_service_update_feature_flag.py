# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for feature flag toggle service method (OMN-5580).

Tests update_feature_flag() outcomes:
- persisted_and_emitted: Infisical + Kafka both succeed
- persist_failed: Infisical write fails
- persisted_emit_failed: Infisical succeeds but Kafka fails
- emit_succeeded_persist_skipped: No INFISICAL_ADDR
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
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
) -> ModelRegistrationProjection:
    """Create a minimal projection for testing."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=uuid4(),
        current_state=EnumRegistrationState.ACTIVE,
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


def _make_service_with_flag(
    flag_name: str = "my_flag",
    flag_value: bool = True,
    env_var: str | None = "MY_FLAG",
) -> ServiceRegistryDiscovery:
    """Create a service with a single flag in projection."""
    proj = _make_projection(
        feature_flags={flag_name: flag_value},
        feature_flag_defaults={flag_name: flag_value},
        feature_flag_metadata={
            flag_name: ModelProjectedFlagMeta(env_var=env_var),
        },
    )

    container = MagicMock()
    reader = AsyncMock()

    async def mock_get_by_state(
        state: EnumRegistrationState,
        limit: int = 10000,
        correlation_id=None,
    ) -> list[ModelRegistrationProjection]:
        if state == EnumRegistrationState.ACTIVE:
            return [proj]
        return []

    reader.get_by_state = mock_get_by_state

    return ServiceRegistryDiscovery(
        container=container,
        projection_reader=reader,
    )


@pytest.mark.unit
class TestUpdateFeatureFlag:
    """Tests for update_feature_flag service method."""

    @pytest.mark.asyncio
    async def test_toggle_returns_persisted_and_emitted(self) -> None:
        """Mock Infisical success + Kafka success -> persisted_and_emitted."""
        service = _make_service_with_flag()
        kafka_producer = AsyncMock()
        kafka_producer.send = AsyncMock()

        with patch.dict(
            "os.environ",
            {"INFISICAL_ADDR": "http://localhost:8880"},
            clear=False,
        ):
            result = await service.update_feature_flag(
                flag_name="my_flag",
                enabled=False,
                kafka_producer=kafka_producer,
            )

        assert result.outcome == "persisted_and_emitted"
        assert result.flag_name == "my_flag"
        assert result.requested_value is False
        assert result.persistence_key == "/shared/feature_flags/my_flag"
        assert result.event_id is not None
        kafka_producer.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_toggle_infisical_failure_returns_persist_failed(self) -> None:
        """Flag not found -> persist_failed."""
        # Service with no flags
        container = MagicMock()
        reader = AsyncMock()
        reader.get_by_state = AsyncMock(return_value=[])

        service = ServiceRegistryDiscovery(
            container=container,
            projection_reader=reader,
        )

        with patch.dict("os.environ", {}, clear=False):
            result = await service.update_feature_flag(
                flag_name="nonexistent_flag",
                enabled=True,
            )

        assert result.outcome == "persist_failed"
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_toggle_kafka_failure_returns_persisted_emit_failed(self) -> None:
        """Infisical configured + Kafka send fails -> persisted_emit_failed."""
        service = _make_service_with_flag()
        kafka_producer = AsyncMock()
        kafka_producer.send = AsyncMock(side_effect=RuntimeError("Kafka down"))

        with patch.dict(
            "os.environ",
            {"INFISICAL_ADDR": "http://localhost:8880"},
            clear=False,
        ):
            result = await service.update_feature_flag(
                flag_name="my_flag",
                enabled=True,
                kafka_producer=kafka_producer,
            )

        assert result.outcome == "persisted_emit_failed"
        assert result.persistence_key == "/shared/feature_flags/my_flag"
        assert result.event_id is None

    @pytest.mark.asyncio
    async def test_toggle_no_infisical_returns_emit_succeeded_persist_skipped(
        self,
    ) -> None:
        """No INFISICAL_ADDR -> emit_succeeded_persist_skipped."""
        service = _make_service_with_flag()

        env_without_infisical = {
            k: v for k, v in __import__("os").environ.items() if k != "INFISICAL_ADDR"
        }

        with patch.dict("os.environ", env_without_infisical, clear=True):
            result = await service.update_feature_flag(
                flag_name="my_flag",
                enabled=True,
            )

        assert result.outcome == "emit_succeeded_persist_skipped"
        assert result.persistence_key is None

    @pytest.mark.asyncio
    async def test_toggle_conflicted_flag_rejected(self) -> None:
        """Conflicted flag (different env_vars) -> persist_failed."""
        proj1 = _make_projection(
            feature_flags={"conflict_flag": True},
            feature_flag_defaults={"conflict_flag": True},
            feature_flag_metadata={
                "conflict_flag": ModelProjectedFlagMeta(env_var="VAR_A"),
            },
        )
        proj2 = _make_projection(
            feature_flags={"conflict_flag": True},
            feature_flag_defaults={"conflict_flag": True},
            feature_flag_metadata={
                "conflict_flag": ModelProjectedFlagMeta(env_var="VAR_B"),
            },
        )

        container = MagicMock()
        reader = AsyncMock()

        async def mock_get_by_state(
            state: EnumRegistrationState,
            limit: int = 10000,
            correlation_id=None,
        ) -> list[ModelRegistrationProjection]:
            if state == EnumRegistrationState.ACTIVE:
                return [proj1, proj2]
            return []

        reader.get_by_state = mock_get_by_state
        service = ServiceRegistryDiscovery(
            container=container,
            projection_reader=reader,
        )

        with patch.dict("os.environ", {}, clear=False):
            result = await service.update_feature_flag(
                flag_name="conflict_flag",
                enabled=False,
            )

        assert result.outcome == "persist_failed"
        assert "conflicted" in result.message.lower()
