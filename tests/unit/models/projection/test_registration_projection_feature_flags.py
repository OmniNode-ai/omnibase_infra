# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for feature flag fields on ModelRegistrationProjection.

OMN-5578: Add feature flag fields to registration projection.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection.model_projected_flag_meta import (
    ModelProjectedFlagMeta,
)
from omnibase_infra.models.projection.model_registration_projection import (
    ModelRegistrationProjection,
)

pytestmark = [pytest.mark.unit]


def _make_projection(**kwargs: object) -> ModelRegistrationProjection:
    """Build a minimal valid projection with defaults."""
    now = datetime.now(UTC)
    defaults: dict[str, object] = {
        "entity_id": uuid4(),
        "current_state": EnumRegistrationState.ACTIVE,
        "node_type": EnumNodeKind.EFFECT,
        "last_applied_event_id": uuid4(),
        "registered_at": now,
        "updated_at": now,
    }
    defaults.update(kwargs)
    return ModelRegistrationProjection(**defaults)  # type: ignore[arg-type]


class TestRegistrationProjectionFeatureFlags:
    """Tests for feature flag fields on the projection model."""

    def test_projection_stores_feature_flags(self) -> None:
        """Construct projection with all 3 flag fields."""
        from omnibase_core.enums.enum_feature_flag_category import (
            EnumFeatureFlagCategory,
        )

        meta = ModelProjectedFlagMeta(
            description="Enable caching",
            category=EnumFeatureFlagCategory.INFRASTRUCTURE,
            env_var="ENABLE_CACHING",
            owner="infra-team",
            value_source="env",
        )

        proj = _make_projection(
            feature_flags={"enable_caching": True, "debug_mode": False},
            feature_flag_defaults={"enable_caching": False, "debug_mode": False},
            feature_flag_metadata={"enable_caching": meta},
        )

        assert proj.feature_flags == {"enable_caching": True, "debug_mode": False}
        assert proj.feature_flag_defaults == {
            "enable_caching": False,
            "debug_mode": False,
        }
        assert "enable_caching" in proj.feature_flag_metadata
        assert proj.feature_flag_metadata["enable_caching"].value_source == "env"

    def test_projection_default_empty(self) -> None:
        """Default construction has empty dicts for all flag fields."""
        proj = _make_projection()

        assert proj.feature_flags == {}
        assert proj.feature_flag_defaults == {}
        assert proj.feature_flag_metadata == {}
