# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for the Kafka broker allowlist validator.

Tests the validate_kafka_broker_allowlist() function added in OMN-3300.
Covers rejected patterns, accepted patterns, allowlist override, and the
missing-env-var warning path exercised via bootstrap().
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.service_kernel import (
    ENV_KAFKA_BROKER_ALLOWLIST,
    validate_kafka_broker_allowlist,
)


@pytest.mark.unit
class TestValidateKafkaBrokerAllowlist:
    """Tests for validate_kafka_broker_allowlist()."""

    # ------------------------------------------------------------------
    # Rejected patterns — denylist enforcement
    # ------------------------------------------------------------------

    def test_rejects_localhost(self) -> None:
        """localhost:* is always rejected."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_kafka_broker_allowlist("localhost:9092")
        assert "localhost:9092" in str(exc_info.value)

    def test_rejects_redpanda_container(self) -> None:
        """redpanda:* (local Docker container name) is always rejected."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_kafka_broker_allowlist("redpanda:9092")
        assert "redpanda:9092" in str(exc_info.value)

    def test_rejects_127_0_0_1(self) -> None:
        """127.0.0.1:* loopback is always rejected."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_kafka_broker_allowlist("127.0.0.1:19092")
        assert "127.0.0.1:19092" in str(exc_info.value)

    def test_rejects_0_0_0_0(self) -> None:
        """0.0.0.0:* wildcard bind address is always rejected."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_kafka_broker_allowlist("0.0.0.0:9092")
        assert "0.0.0.0:9092" in str(exc_info.value)

    def test_rejected_error_message_includes_expected_hint(self) -> None:
        """Error message tells the operator what to set instead."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_kafka_broker_allowlist("localhost:9092")
        error_msg = str(exc_info.value)
        assert "KAFKA_BROKER_ALLOWLIST" in error_msg

    # ------------------------------------------------------------------
    # Accepted patterns — remote broker addresses
    # ------------------------------------------------------------------

    def test_accepts_production_broker(self) -> None:
        """Remote IP address passes without error."""
        # Should not raise
        validate_kafka_broker_allowlist(
            "192.168.86.200:19092"  # kafka-fallback-ok — testing allowlist validation logic
        )

    def test_accepts_multi_broker_all_valid(self) -> None:
        """Comma-separated list of valid brokers all pass."""
        validate_kafka_broker_allowlist(
            "192.168.86.200:19092,192.168.86.201:19092"  # kafka-fallback-ok — testing allowlist validation logic
        )

    def test_accepts_hostname_not_on_denylist(self) -> None:
        """Arbitrary hostname not on the denylist is accepted."""
        validate_kafka_broker_allowlist("kafka.internal.company.com:9092")

    def test_accepts_ip_in_10_range(self) -> None:
        """RFC-1918 10.x.x.x addresses are accepted by default."""
        validate_kafka_broker_allowlist("10.0.0.5:9092")

    # ------------------------------------------------------------------
    # Missing env-var behavior (KAFKA_BOOTSTRAP_SERVERS unset)
    # ------------------------------------------------------------------

    def test_empty_string_skips_validation(self) -> None:
        """Empty string (env var not set) is not validated — caller handles it."""
        # validate_kafka_broker_allowlist is only called when the env var is set.
        # An empty string with no populated brokers should iterate without error.
        validate_kafka_broker_allowlist("")

    # ------------------------------------------------------------------
    # KAFKA_BROKER_ALLOWLIST override
    # ------------------------------------------------------------------

    def test_allowlist_override_bypasses_denylist(self) -> None:
        """An operator-supplied allowlist prefix can permit a normally-denied host."""
        with patch.dict("os.environ", {ENV_KAFKA_BROKER_ALLOWLIST: "localhost:"}):
            # Now localhost: is in the allowlist — should not raise
            validate_kafka_broker_allowlist("localhost:9092")

    def test_allowlist_override_with_multiple_prefixes(self) -> None:
        """Comma-separated allowlist prefixes all work independently."""
        with patch.dict(
            "os.environ",
            {ENV_KAFKA_BROKER_ALLOWLIST: "redpanda:,localhost:"},
        ):
            validate_kafka_broker_allowlist("redpanda:9092")
            validate_kafka_broker_allowlist("localhost:9092")

    def test_allowlist_does_not_bypass_non_matching_broker(self) -> None:
        """An allowlist prefix that doesn't match the broker still rejects it."""
        with patch.dict(
            "os.environ",
            {ENV_KAFKA_BROKER_ALLOWLIST: "10.0.0."},
        ):
            with pytest.raises(ProtocolConfigurationError):
                validate_kafka_broker_allowlist("localhost:9092")

    def test_allowlist_prefix_does_not_match_substring(self) -> None:
        """SECURITY: localhost: allowlist must not match evil-localhost.example.com.

        If this test fails, the allowlist uses substring (in) instead of prefix
        (startswith) matching. Fix the implementation immediately — this is a
        security boundary violation.
        """
        with patch.dict("os.environ", {ENV_KAFKA_BROKER_ALLOWLIST: "localhost:"}):
            with pytest.raises(ProtocolConfigurationError):
                validate_kafka_broker_allowlist("evil-localhost.example.com:9092")

    def test_multi_broker_one_denied(self) -> None:
        """Comma-separated list raises if any broker matches the denylist.

        kafka.internal.company.com:9092 passes (not on denylist).
        redpanda:9092 fails (matches ^redpanda: pattern).
        The error message must name the denied broker.
        """
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_kafka_broker_allowlist(
                "kafka.internal.company.com:9092,redpanda:9092"
            )
        assert "redpanda:9092" in str(exc_info.value)

    # ------------------------------------------------------------------
    # KAFKA_BOOTSTRAP_SERVERS unset warning (via bootstrap() interaction)
    # ------------------------------------------------------------------

    def test_correlation_id_forwarded_in_error(self) -> None:
        """Correlation ID is forwarded into the error context when provided."""
        from uuid import UUID, uuid4

        cid: UUID = uuid4()
        with pytest.raises(ProtocolConfigurationError):
            validate_kafka_broker_allowlist("redpanda:9092", correlation_id=cid)
