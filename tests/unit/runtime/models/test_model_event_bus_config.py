# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelEventBusConfig.

Tests validate:
- Default type is EnumEventBusType.KAFKA (not inmemory)
- INMEMORY type is rejected via is_production_safe validator
- KAFKA and CLOUD types are accepted
- Model is frozen and forbids extra fields
"""

import pytest
from pydantic import ValidationError

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_infra.runtime.models.model_event_bus_config import ModelEventBusConfig


@pytest.mark.unit
class TestModelEventBusConfig:
    """Tests for ModelEventBusConfig default and validation."""

    def test_default_is_kafka(self) -> None:
        """ModelEventBusConfig() should default to KAFKA, not inmemory."""
        config = ModelEventBusConfig()
        assert config.type == EnumEventBusType.KAFKA

    def test_inmemory_raises(self) -> None:
        """ModelEventBusConfig(type='inmemory') must raise ValidationError."""
        with pytest.raises(ValidationError, match="not production-safe"):
            ModelEventBusConfig(type="inmemory")

    def test_kafka_accepted(self) -> None:
        """ModelEventBusConfig(type='kafka') should succeed."""
        config = ModelEventBusConfig(type="kafka")
        assert config.type == EnumEventBusType.KAFKA

    def test_cloud_accepted(self) -> None:
        """ModelEventBusConfig(type='cloud') should succeed."""
        config = ModelEventBusConfig(type="cloud")
        assert config.type == EnumEventBusType.CLOUD

    def test_invalid_type_raises(self) -> None:
        """ModelEventBusConfig(type='redis') should raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelEventBusConfig(type="redis")

    def test_frozen(self) -> None:
        """ModelEventBusConfig instances should be immutable."""
        config = ModelEventBusConfig()
        with pytest.raises(ValidationError):
            config.type = EnumEventBusType.CLOUD  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """ModelEventBusConfig should reject unknown fields."""
        with pytest.raises(ValidationError):
            ModelEventBusConfig(unknown_field="value")  # type: ignore[call-arg]
