# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelEventBusConfig.

Tests validate:
- Default type is EnumEventBusType.KAFKA (not inmemory)
- Default profile is LANE (fail-closed) — OMN-17304 profile axis
- INMEMORY is rejected under the lane profile and ACCEPTED under the local
  profile ('inmemory' is a first-class configured value for local runtimes,
  and what the shipped tier-0 default declares)
- KAFKA and CLOUD types are accepted under both profiles
- Model is frozen and forbids extra fields
"""

import pytest
from pydantic import ValidationError

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_infra.runtime.models.enum_event_bus_profile import EnumEventBusProfile
from omnibase_infra.runtime.models.model_event_bus_config import ModelEventBusConfig


@pytest.mark.unit
class TestModelEventBusConfig:
    """Tests for ModelEventBusConfig default and validation."""

    def test_default_is_kafka(self) -> None:
        """ModelEventBusConfig() should default to KAFKA, not inmemory."""
        config = ModelEventBusConfig()
        assert config.type == EnumEventBusType.KAFKA

    def test_default_profile_is_lane(self) -> None:
        """The profile axis defaults to LANE — every pre-axis config validates
        exactly as strictly as before (fail-closed, OMN-17304)."""
        config = ModelEventBusConfig()
        assert config.profile == EnumEventBusProfile.LANE

    def test_inmemory_raises_under_the_default_lane_profile(self) -> None:
        """ModelEventBusConfig(type='inmemory') must raise ValidationError.

        The default profile is 'lane': an in-memory bus in a deployed lane
        silently strands evidence outside the shared projections, so the
        pre-OMN-17304 rejection is preserved for every config that does not
        explicitly declare the local profile.
        """
        with pytest.raises(ValidationError, match="not production-safe"):
            ModelEventBusConfig(type="inmemory")

    def test_inmemory_rejection_names_the_local_profile_remedy(self) -> None:
        """The lane rejection tells the operator HOW a local runtime says it."""
        with pytest.raises(ValidationError, match="profile"):
            ModelEventBusConfig(type="inmemory")

    def test_inmemory_accepted_under_the_local_profile(self) -> None:
        """OMN-17304: 'inmemory' is a first-class configured value for local
        runtimes — the shipped tier-0 default declares exactly this pair."""
        config = ModelEventBusConfig(type="inmemory", profile=EnumEventBusProfile.LOCAL)
        assert config.type == EnumEventBusType.INMEMORY
        assert config.profile == EnumEventBusProfile.LOCAL

    def test_kafka_accepted_under_both_profiles(self) -> None:
        """Broker-backed transports are legal everywhere."""
        assert ModelEventBusConfig(type="kafka").type == EnumEventBusType.KAFKA
        assert (
            ModelEventBusConfig(type="kafka", profile=EnumEventBusProfile.LOCAL).type
            == EnumEventBusType.KAFKA
        )

    def test_cloud_accepted(self) -> None:
        """ModelEventBusConfig(type='cloud') should succeed."""
        config = ModelEventBusConfig(type="cloud")
        assert config.type == EnumEventBusType.CLOUD

    def test_invalid_type_raises(self) -> None:
        """ModelEventBusConfig(type='redis') should raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelEventBusConfig(type="redis")

    def test_invalid_profile_raises(self) -> None:
        """ModelEventBusConfig(profile='prod') should raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelEventBusConfig(profile="prod")  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        """ModelEventBusConfig instances should be immutable."""
        config = ModelEventBusConfig()
        with pytest.raises(ValidationError):
            config.type = EnumEventBusType.CLOUD  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """ModelEventBusConfig should reject unknown fields."""
        with pytest.raises(ValidationError):
            ModelEventBusConfig(unknown_field="value")  # type: ignore[call-arg]
