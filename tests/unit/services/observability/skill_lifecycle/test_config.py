# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ConfigSkillLifecycleConsumer (OMN-2934).

Tests:
    - Default values for all configuration fields
    - Pydantic validation (bounds, empty topics)
    - Model validators (topic configuration)
    - Environment variable loading with OMNIBASE_INFRA_SKILL_LIFECYCLE_ prefix

All tests mock environment state via monkeypatch - no real infrastructure required.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.services.observability.skill_lifecycle.config import (
    ConfigSkillLifecycleConsumer,
)

_REQUIRED_DSN = "postgresql://postgres:secret@localhost:5432/testdb"
_ENV_PREFIX = "OMNIBASE_INFRA_SKILL_LIFECYCLE_"


def _make_config(**overrides: object) -> ConfigSkillLifecycleConsumer:
    """Create a config with required fields filled in."""
    defaults: dict[str, object] = {
        "postgres_dsn": _REQUIRED_DSN,
        "_env_file": None,
    }
    defaults.update(overrides)
    return ConfigSkillLifecycleConsumer(**defaults)  # type: ignore[arg-type]


# =============================================================================
# Tests: Default Values
# =============================================================================


class TestConfigDefaults:
    """Verify all default field values match the contract specification."""

    @pytest.mark.unit
    def test_default_scalar_values(self) -> None:
        """All scalar defaults match expected values."""
        cfg = _make_config()

        assert cfg.kafka_bootstrap_servers == "localhost:9092"
        assert cfg.kafka_group_id == "skill-lifecycle-postgres"
        assert cfg.auto_offset_reset == "earliest"
        assert cfg.enable_auto_commit is False
        assert cfg.batch_size == 100
        assert cfg.batch_timeout_ms == 1000
        assert cfg.poll_timeout_buffer_seconds == 5.0
        assert cfg.circuit_breaker_threshold == 5
        assert cfg.circuit_breaker_reset_timeout == 60.0
        assert cfg.circuit_breaker_half_open_successes == 1
        assert cfg.dlq_topic == "onex.evt.omniclaude.skill-lifecycle-dlq.v1"
        assert cfg.dlq_enabled is True
        assert cfg.max_retry_count == 3
        assert cfg.health_check_port == 8092
        assert cfg.health_check_host == "127.0.0.1"
        assert cfg.health_check_staleness_seconds == 300
        assert cfg.health_check_poll_staleness_seconds == 60

    @pytest.mark.unit
    def test_default_topics(self) -> None:
        """Default topics contain both skill lifecycle topics."""
        cfg = _make_config()

        assert "onex.evt.omniclaude.skill-started.v1" in cfg.topics
        assert "onex.evt.omniclaude.skill-completed.v1" in cfg.topics
        assert len(cfg.topics) == 2

    @pytest.mark.unit
    def test_postgres_dsn_stored(self) -> None:
        """The required postgres_dsn is stored as provided."""
        dsn = "postgresql://user:pass@db.example.com:5436/mydb"
        cfg = _make_config(postgres_dsn=dsn)

        assert cfg.postgres_dsn == dsn


# =============================================================================
# Tests: Validation
# =============================================================================


class TestConfigValidation:
    """Test Pydantic field constraints and model validators."""

    @pytest.mark.unit
    def test_empty_topics_raises(self) -> None:
        """Setting topics to an empty list raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="No topics configured"):
            _make_config(topics=[])

    @pytest.mark.unit
    def test_batch_size_lower_bound(self) -> None:
        """batch_size below 1 is rejected."""
        with pytest.raises(ValidationError, match="batch_size"):
            _make_config(batch_size=0)

    @pytest.mark.unit
    def test_batch_size_upper_bound(self) -> None:
        """batch_size above 1000 is rejected."""
        with pytest.raises(ValidationError, match="batch_size"):
            _make_config(batch_size=1001)

    @pytest.mark.unit
    def test_batch_timeout_ms_lower_bound(self) -> None:
        """batch_timeout_ms below 100 is rejected."""
        with pytest.raises(ValidationError, match="batch_timeout_ms"):
            _make_config(batch_timeout_ms=99)

    @pytest.mark.unit
    def test_batch_timeout_ms_upper_bound(self) -> None:
        """batch_timeout_ms above 60000 is rejected."""
        with pytest.raises(ValidationError, match="batch_timeout_ms"):
            _make_config(batch_timeout_ms=60001)

    @pytest.mark.unit
    def test_health_check_port_lower_bound(self) -> None:
        """Port below 1024 is rejected."""
        with pytest.raises(ValidationError, match="health_check_port"):
            _make_config(health_check_port=1023)

    @pytest.mark.unit
    def test_health_check_port_upper_bound(self) -> None:
        """Port above 65535 is rejected."""
        with pytest.raises(ValidationError, match="health_check_port"):
            _make_config(health_check_port=65536)

    @pytest.mark.unit
    def test_health_check_port_valid(self) -> None:
        """Ports within [1024, 65535] are accepted."""
        cfg_low = _make_config(health_check_port=1024)
        cfg_high = _make_config(health_check_port=65535)

        assert cfg_low.health_check_port == 1024
        assert cfg_high.health_check_port == 65535

    @pytest.mark.unit
    def test_circuit_breaker_threshold_lower_bound(self) -> None:
        """Threshold below 1 is rejected."""
        with pytest.raises(ValidationError, match="circuit_breaker_threshold"):
            _make_config(circuit_breaker_threshold=0)

    @pytest.mark.unit
    def test_circuit_breaker_reset_timeout_lower_bound(self) -> None:
        """Reset timeout below 1.0 is rejected."""
        with pytest.raises(ValidationError, match="circuit_breaker_reset_timeout"):
            _make_config(circuit_breaker_reset_timeout=0.5)

    @pytest.mark.unit
    def test_max_retry_count_lower_bound(self) -> None:
        """max_retry_count below 0 is rejected."""
        with pytest.raises(ValidationError, match="max_retry_count"):
            _make_config(max_retry_count=-1)

    @pytest.mark.unit
    def test_max_retry_count_upper_bound(self) -> None:
        """max_retry_count above 10 is rejected."""
        with pytest.raises(ValidationError, match="max_retry_count"):
            _make_config(max_retry_count=11)

    @pytest.mark.unit
    def test_extra_fields_ignored(self) -> None:
        """Extra fields are silently ignored."""
        cfg = _make_config(unknown_field="ignored")

        assert cfg.postgres_dsn == _REQUIRED_DSN


# =============================================================================
# Tests: Environment Variable Loading
# =============================================================================


class TestConfigEnvironment:
    """Test environment variable integration with pydantic-settings."""

    @pytest.mark.unit
    def test_env_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables with OMNIBASE_INFRA_SKILL_LIFECYCLE_ prefix are loaded."""
        monkeypatch.setenv(
            f"{_ENV_PREFIX}POSTGRES_DSN",
            "postgresql://env:pass@envhost:5432/envdb",
        )
        monkeypatch.setenv(f"{_ENV_PREFIX}KAFKA_BOOTSTRAP_SERVERS", "kafka.env:9092")

        cfg = ConfigSkillLifecycleConsumer(_env_file=None)  # type: ignore[call-arg]

        assert cfg.postgres_dsn == "postgresql://env:pass@envhost:5432/envdb"
        assert cfg.kafka_bootstrap_servers == "kafka.env:9092"

    @pytest.mark.unit
    def test_missing_required_dsn_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing postgres_dsn raises ValidationError."""
        monkeypatch.delenv(f"{_ENV_PREFIX}POSTGRES_DSN", raising=False)

        with pytest.raises(ValidationError, match="postgres_dsn"):
            ConfigSkillLifecycleConsumer(_env_file=None)  # type: ignore[call-arg]

    @pytest.mark.unit
    def test_env_batch_size_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """batch_size can be overridden via environment variable."""
        monkeypatch.setenv(f"{_ENV_PREFIX}POSTGRES_DSN", _REQUIRED_DSN)
        monkeypatch.setenv(f"{_ENV_PREFIX}BATCH_SIZE", "42")

        cfg = ConfigSkillLifecycleConsumer(_env_file=None)  # type: ignore[call-arg]

        assert cfg.batch_size == 42

    @pytest.mark.unit
    def test_env_health_port_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """health_check_port can be overridden via environment variable."""
        monkeypatch.setenv(f"{_ENV_PREFIX}POSTGRES_DSN", _REQUIRED_DSN)
        monkeypatch.setenv(f"{_ENV_PREFIX}HEALTH_CHECK_PORT", "9999")

        cfg = ConfigSkillLifecycleConsumer(_env_file=None)  # type: ignore[call-arg]

        assert cfg.health_check_port == 9999
