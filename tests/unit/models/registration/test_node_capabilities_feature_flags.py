# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for feature_flags field on ModelNodeCapabilities.

OMN-5575: Add contract-declared feature flags to node capabilities.
"""

from __future__ import annotations

import pytest

from omnibase_core.models.core.model_feature_flags import ModelFeatureFlags
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)

pytestmark = [pytest.mark.unit]


class TestNodeCapabilitiesFeatureFlags:
    """Tests for the feature_flags field on ModelNodeCapabilities."""

    def test_capabilities_with_feature_flags(self) -> None:
        """Construct with explicit feature flags and verify type."""
        flags = ModelFeatureFlags()
        flags.enable("enable_caching")
        flags.disable("debug_mode")

        caps = ModelNodeCapabilities(
            postgres=True,
            feature_flags=flags,
        )

        assert isinstance(caps.feature_flags, ModelFeatureFlags)
        assert caps.feature_flags.is_enabled("enable_caching") is True
        assert caps.feature_flags.is_enabled("debug_mode") is False
        assert caps.feature_flags.get_flag_count() == 2

    def test_capabilities_default_empty_flags(self) -> None:
        """Default construction has empty ModelFeatureFlags."""
        caps = ModelNodeCapabilities()

        assert isinstance(caps.feature_flags, ModelFeatureFlags)
        assert caps.feature_flags.get_flag_count() == 0
