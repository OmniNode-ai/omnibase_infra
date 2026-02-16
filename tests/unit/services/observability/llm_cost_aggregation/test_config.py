# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ConfigLlmCostAggregation.

Tests:
    - Default values for all configuration fields
    - Pydantic validation (bounds, empty topics, pool size ordering)
    - Model validators (timing warnings, topic configuration)
    - Environment variable loading with OMNIBASE_INFRA_LLM_COST_ prefix

All tests mock environment state via monkeypatch - no real infrastructure required.

Related Tickets:
    - OMN-2240: E1-T4 LLM cost aggregation service
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.services.observability.llm_cost_aggregation.config import (
    ConfigLlmCostAggregation,
)

# Required field that has no default - all tests need to provide it
_REQUIRED_DSN = "postgresql://postgres:secret@localhost:5432/testdb"

# Prefix used by pydantic-settings for env var loading
_ENV_PREFIX = "OMNIBASE_INFRA_LLM_COST_"


def _make_config(**overrides: object) -> ConfigLlmCostAggregation:
    """Create a config with required fields filled in.

    Provides the required ``postgres_dsn`` and merges any overrides.
    Bypasses .env file loading via ``_env_file=None`` to isolate tests.
    """
    defaults: dict[str, object] = {
        "postgres_dsn": _REQUIRED_DSN,
        "_env_file": None,
    }
    defaults.update(overrides)
    return ConfigLlmCostAggregation(**defaults)  # type: ignore[arg-type]


# =============================================================================
# Tests: Default Values
# =============================================================================


class TestConfigDefaults:
    """Verify all default field values match the contract specification."""

    @pytest.mark.unit
    def test_default_values(self) -> None:
        """All scalar defaults match expected values from config.py."""
        cfg = _make_config()

        assert cfg.kafka_bootstrap_servers == "localhost:9092"
        assert cfg.kafka_group_id == "llm-cost-aggregation-postgres"
        assert cfg.auto_offset_reset == "earliest"
        assert cfg.batch_size == 100
        assert cfg.batch_timeout_ms == 1000
        assert cfg.poll_timeout_buffer_seconds == 5.0
        assert cfg.pool_min_size == 2
        assert cfg.pool_max_size == 10
        assert cfg.circuit_breaker_threshold == 5
        assert cfg.circuit_breaker_reset_timeout == 60.0
        assert cfg.circuit_breaker_half_open_successes == 1
        assert cfg.health_check_port == 8089
        assert cfg.health_check_host == "0.0.0.0"  # noqa: S104
        assert cfg.health_check_staleness_seconds == 300
        assert cfg.health_check_poll_staleness_seconds == 60
        assert cfg.startup_grace_period_seconds == 60.0

    @pytest.mark.unit
    def test_kafka_topics_default(self) -> None:
        """Default topics list contains the LLM call completed topic."""
        cfg = _make_config()

        assert cfg.topics == ["onex.evt.omniintelligence.llm-call-completed.v1"]
        assert len(cfg.topics) == 1

    @pytest.mark.unit
    def test_postgres_dsn_stored(self) -> None:
        """The required postgres_dsn is stored as provided."""
        dsn = "postgresql://user:pass@db.example.com:5436/mydb"
        cfg = _make_config(postgres_dsn=dsn)

        assert cfg.postgres_dsn == dsn

    @pytest.mark.unit
    def test_batch_defaults(self) -> None:
        """Batch processing defaults are within documented ranges."""
        cfg = _make_config()

        assert 1 <= cfg.batch_size <= 1000
        assert 100 <= cfg.batch_timeout_ms <= 60000
        assert 1.0 <= cfg.poll_timeout_buffer_seconds <= 30.0

    @pytest.mark.unit
    def test_circuit_breaker_defaults(self) -> None:
        """Circuit breaker defaults are within documented ranges."""
        cfg = _make_config()

        assert 1 <= cfg.circuit_breaker_threshold <= 100
        assert 1.0 <= cfg.circuit_breaker_reset_timeout <= 3600.0
        assert 1 <= cfg.circuit_breaker_half_open_successes <= 10


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
        """batch_size below 1 is rejected by Pydantic."""
        with pytest.raises(ValidationError, match="batch_size"):
            _make_config(batch_size=0)

    @pytest.mark.unit
    def test_batch_size_upper_bound(self) -> None:
        """batch_size above 1000 is rejected by Pydantic."""
        with pytest.raises(ValidationError, match="batch_size"):
            _make_config(batch_size=1001)

    @pytest.mark.unit
    def test_batch_size_within_range(self) -> None:
        """batch_size within [1, 1000] is accepted."""
        cfg_low = _make_config(batch_size=1)
        cfg_high = _make_config(batch_size=1000)

        assert cfg_low.batch_size == 1
        assert cfg_high.batch_size == 1000

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
    def test_pool_min_size_lower_bound(self) -> None:
        """pool_min_size below 1 is rejected."""
        with pytest.raises(ValidationError, match="pool_min_size"):
            _make_config(pool_min_size=0)

    @pytest.mark.unit
    def test_pool_max_size_upper_bound(self) -> None:
        """pool_max_size above 50 is rejected."""
        with pytest.raises(ValidationError, match="pool_max_size"):
            _make_config(pool_max_size=51)

    @pytest.mark.unit
    def test_pool_size_valid_range(self) -> None:
        """pool_min_size and pool_max_size within bounds are accepted."""
        cfg = _make_config(pool_min_size=5, pool_max_size=20)

        assert cfg.pool_min_size == 5
        assert cfg.pool_max_size == 20

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
    def test_timing_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning is logged when circuit breaker timeout < 2x batch timeout."""
        # batch_timeout_ms=10000 means batch_timeout_seconds=10
        # min_recommended_circuit_timeout = 20
        # circuit_breaker_reset_timeout=5.0 < 20 -> triggers warning
        with caplog.at_level(logging.WARNING):
            _make_config(
                batch_timeout_ms=10000,
                circuit_breaker_reset_timeout=5.0,
            )

        assert any(
            "Circuit breaker timeout" in record.message
            and "less than 2x batch timeout" in record.message
            for record in caplog.records
        ), (
            f"Expected timing warning not found. Records: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.unit
    def test_no_timing_warning_when_sufficient(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when circuit breaker timeout >= 2x batch timeout."""
        # batch_timeout_ms=1000 -> batch_timeout_seconds=1
        # min_recommended = 2.0
        # circuit_breaker_reset_timeout=60.0 >= 2.0 -> no warning
        with caplog.at_level(logging.WARNING):
            _make_config(
                batch_timeout_ms=1000,
                circuit_breaker_reset_timeout=60.0,
            )

        timing_warnings = [
            r for r in caplog.records if "Circuit breaker timeout" in r.message
        ]
        assert len(timing_warnings) == 0

    @pytest.mark.unit
    def test_poll_timeout_buffer_lower_bound(self) -> None:
        """poll_timeout_buffer_seconds below 1.0 is rejected."""
        with pytest.raises(ValidationError, match="poll_timeout_buffer_seconds"):
            _make_config(poll_timeout_buffer_seconds=0.5)

    @pytest.mark.unit
    def test_poll_timeout_buffer_upper_bound(self) -> None:
        """poll_timeout_buffer_seconds above 30.0 is rejected."""
        with pytest.raises(ValidationError, match="poll_timeout_buffer_seconds"):
            _make_config(poll_timeout_buffer_seconds=31.0)

    @pytest.mark.unit
    def test_startup_grace_period_bounds(self) -> None:
        """startup_grace_period_seconds bounds are enforced."""
        # Lower bound: 0.0 is valid
        cfg = _make_config(startup_grace_period_seconds=0.0)
        assert cfg.startup_grace_period_seconds == 0.0

        # Upper bound: 600.0 is valid
        cfg = _make_config(startup_grace_period_seconds=600.0)
        assert cfg.startup_grace_period_seconds == 600.0

        # Above upper bound: rejected
        with pytest.raises(ValidationError, match="startup_grace_period_seconds"):
            _make_config(startup_grace_period_seconds=601.0)

    @pytest.mark.unit
    def test_pool_min_exceeds_max_raises(self) -> None:
        """pool_min_size > pool_max_size raises ProtocolConfigurationError."""
        with pytest.raises(
            ProtocolConfigurationError, match=r"pool_min_size.*must not exceed"
        ):
            _make_config(pool_min_size=20, pool_max_size=5)

    @pytest.mark.unit
    def test_auto_offset_reset_invalid_raises(self) -> None:
        """auto_offset_reset outside Literal values raises ValidationError."""
        with pytest.raises(ValidationError, match="auto_offset_reset"):
            _make_config(auto_offset_reset="invalid")

    @pytest.mark.unit
    def test_health_check_staleness_bounds(self) -> None:
        """health_check_staleness_seconds bounds are enforced."""
        cfg = _make_config(health_check_staleness_seconds=60)
        assert cfg.health_check_staleness_seconds == 60

        with pytest.raises(ValidationError, match="health_check_staleness_seconds"):
            _make_config(health_check_staleness_seconds=59)

        with pytest.raises(ValidationError, match="health_check_staleness_seconds"):
            _make_config(health_check_staleness_seconds=3601)

    @pytest.mark.unit
    def test_health_check_poll_staleness_bounds(self) -> None:
        """health_check_poll_staleness_seconds bounds are enforced."""
        cfg = _make_config(health_check_poll_staleness_seconds=10)
        assert cfg.health_check_poll_staleness_seconds == 10

        with pytest.raises(
            ValidationError, match="health_check_poll_staleness_seconds"
        ):
            _make_config(health_check_poll_staleness_seconds=9)

        with pytest.raises(
            ValidationError, match="health_check_poll_staleness_seconds"
        ):
            _make_config(health_check_poll_staleness_seconds=301)


# =============================================================================
# Tests: Environment Variable Loading
# =============================================================================


class TestConfigEnvironment:
    """Test environment variable integration with pydantic-settings."""

    @pytest.mark.unit
    def test_env_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables with OMNIBASE_INFRA_LLM_COST_ prefix are loaded."""
        monkeypatch.setenv(
            f"{_ENV_PREFIX}POSTGRES_DSN",
            "postgresql://env:pass@envhost:5432/envdb",
        )
        monkeypatch.setenv(f"{_ENV_PREFIX}KAFKA_BOOTSTRAP_SERVERS", "kafka.env:9092")

        cfg = ConfigLlmCostAggregation(_env_file=None)  # type: ignore[call-arg]

        assert cfg.postgres_dsn == "postgresql://env:pass@envhost:5432/envdb"
        assert cfg.kafka_bootstrap_servers == "kafka.env:9092"

    @pytest.mark.unit
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables override default values."""
        monkeypatch.setenv(f"{_ENV_PREFIX}POSTGRES_DSN", _REQUIRED_DSN)
        monkeypatch.setenv(f"{_ENV_PREFIX}BATCH_SIZE", "42")
        monkeypatch.setenv(f"{_ENV_PREFIX}HEALTH_CHECK_PORT", "9999")
        monkeypatch.setenv(f"{_ENV_PREFIX}CIRCUIT_BREAKER_THRESHOLD", "10")

        cfg = ConfigLlmCostAggregation(_env_file=None)  # type: ignore[call-arg]

        assert cfg.batch_size == 42
        assert cfg.health_check_port == 9999
        assert cfg.circuit_breaker_threshold == 10

    @pytest.mark.unit
    def test_env_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables are matched case-insensitively."""
        # pydantic-settings with case_sensitive=False matches env vars
        # regardless of case. The env var name itself is always uppercase
        # on most OSes, but the field name mapping is case-insensitive.
        monkeypatch.setenv(f"{_ENV_PREFIX}POSTGRES_DSN", _REQUIRED_DSN)
        monkeypatch.setenv(f"{_ENV_PREFIX}KAFKA_GROUP_ID", "custom-group")

        cfg = ConfigLlmCostAggregation(_env_file=None)  # type: ignore[call-arg]

        assert cfg.kafka_group_id == "custom-group"

    @pytest.mark.unit
    def test_env_topics_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Topics can be overridden via environment variable (JSON list)."""
        monkeypatch.setenv(f"{_ENV_PREFIX}POSTGRES_DSN", _REQUIRED_DSN)
        monkeypatch.setenv(
            f"{_ENV_PREFIX}TOPICS",
            '["topic-a", "topic-b"]',
        )

        cfg = ConfigLlmCostAggregation(_env_file=None)  # type: ignore[call-arg]

        assert cfg.topics == ["topic-a", "topic-b"]

    @pytest.mark.unit
    def test_missing_required_dsn_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing postgres_dsn without env var raises ValidationError."""
        # Clear any potential env var that might provide the DSN
        monkeypatch.delenv(f"{_ENV_PREFIX}POSTGRES_DSN", raising=False)

        with pytest.raises(ValidationError, match="postgres_dsn"):
            ConfigLlmCostAggregation(_env_file=None)  # type: ignore[call-arg]

    @pytest.mark.unit
    def test_extra_fields_ignored(self) -> None:
        """Extra fields are silently ignored (extra='ignore' in model_config)."""
        # Should not raise - extra fields are ignored
        cfg = _make_config(some_unknown_field="ignored_value")

        assert cfg.postgres_dsn == _REQUIRED_DSN
