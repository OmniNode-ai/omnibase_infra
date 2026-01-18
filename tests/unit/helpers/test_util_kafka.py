# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for tests.helpers.util_kafka module.

Tests the Kafka configuration validation utilities, including:
- parse_bootstrap_servers: Parsing various KAFKA_BOOTSTRAP_SERVERS formats
- validate_bootstrap_servers: Comprehensive validation with skip reasons
"""

from __future__ import annotations

import pytest

from tests.helpers.util_kafka import (
    KafkaConfigValidationResult,
    parse_bootstrap_servers,
    validate_bootstrap_servers,
)


class TestParseBootstrapServers:
    """Tests for parse_bootstrap_servers function."""

    def test_standard_host_port(self) -> None:
        """Test parsing standard host:port format."""
        host, port = parse_bootstrap_servers("localhost:9092")
        assert host == "localhost"
        assert port == "9092"

    def test_ip_with_port(self) -> None:
        """Test parsing IP address with port."""
        host, port = parse_bootstrap_servers("localhost:29092")
        assert host == "localhost"
        assert port == "29092"

    def test_hostname_without_port(self) -> None:
        """Test parsing hostname without port returns default."""
        host, port = parse_bootstrap_servers("kafka-broker")
        assert host == "kafka-broker"
        assert port == "29092"  # Default port

    def test_empty_string(self) -> None:
        """Test parsing empty string returns not-set marker."""
        host, port = parse_bootstrap_servers("")
        assert host == "<not set>"
        assert port == "29092"

    def test_whitespace_only(self) -> None:
        """Test parsing whitespace-only string returns not-set marker."""
        host, port = parse_bootstrap_servers("   ")
        assert host == "<not set>"
        assert port == "29092"

    def test_ipv6_with_brackets(self) -> None:
        """Test parsing IPv6 address with brackets."""
        host, port = parse_bootstrap_servers("[::1]:9092")
        assert host == "[::1]"
        assert port == "9092"

    def test_ipv6_without_port(self) -> None:
        """Test parsing IPv6 address without port after bracket."""
        host, port = parse_bootstrap_servers("[::1]")
        assert host == "[::1]"
        assert port == "29092"  # Default port

    def test_host_with_empty_port(self) -> None:
        """Test parsing host with trailing colon but no port."""
        host, port = parse_bootstrap_servers("localhost:")
        assert host == "localhost"
        assert port == "29092"  # Default port for empty port section

    # =========================================================================
    # Bare IPv6 Address Tests
    # =========================================================================

    def test_bare_ipv6_localhost(self) -> None:
        """Test parsing bare IPv6 localhost without brackets."""
        host, port = parse_bootstrap_servers("::1")
        assert host == "::1"
        assert port == "29092"  # Default port for bare IPv6

    def test_bare_ipv6_full_address(self) -> None:
        """Test parsing bare IPv6 full address without brackets."""
        host, port = parse_bootstrap_servers("2001:db8::1")
        assert host == "2001:db8::1"
        assert port == "29092"  # Default port for bare IPv6

    def test_bare_ipv6_with_all_segments(self) -> None:
        """Test parsing bare IPv6 with all segments specified."""
        host, port = parse_bootstrap_servers("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert host == "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
        assert port == "29092"  # Default port for bare IPv6

    def test_bare_ipv6_mapped_ipv4(self) -> None:
        """Test parsing IPv4-mapped IPv6 address."""
        host, port = parse_bootstrap_servers("::ffff:192.168.1.1")
        assert host == "::ffff:192.168.1.1"
        assert port == "29092"  # Default port for bare IPv6

    def test_bare_ipv6_ambiguous_with_port_like_suffix(self) -> None:
        """Test bare IPv6 with port-like suffix is treated as full address.

        The string "::1:9092" is ambiguous - it could mean:
        - IPv6 "::1" with port 9092, OR
        - IPv6 "::1:9092" without a port

        We treat it as bare IPv6 (full address) with default port.
        For unambiguous IPv6 with port, use bracketed format: [::1]:9092
        """
        host, port = parse_bootstrap_servers("::1:9092")
        assert host == "::1:9092"  # Entire string is the host
        assert port == "29092"  # Default port since it's bare IPv6


class TestValidateBootstrapServers:
    """Tests for validate_bootstrap_servers function."""

    def test_valid_standard_format(self) -> None:
        """Test validation passes for standard host:port format."""
        result = validate_bootstrap_servers("localhost:9092")
        assert result.is_valid is True
        assert result.host == "localhost"
        assert result.port == "9092"
        assert result.error_message is None
        assert result.skip_reason is None

    def test_valid_ip_format(self) -> None:
        """Test validation passes for IP:port format."""
        result = validate_bootstrap_servers("localhost:29092")
        assert result.is_valid is True
        assert result.host == "localhost"
        assert result.port == "29092"

    def test_valid_hostname_without_port(self) -> None:
        """Test validation passes for hostname without port (uses default)."""
        result = validate_bootstrap_servers("kafka-broker")
        assert result.is_valid is True
        assert result.host == "kafka-broker"
        assert result.port == "29092"  # Default port

    def test_none_input(self) -> None:
        """Test validation fails for None input."""
        result = validate_bootstrap_servers(None)
        assert result.is_valid is False
        assert result.host == "<not set>"
        assert result.error_message is not None
        assert "not set" in result.error_message.lower()
        assert result.skip_reason is not None
        assert "KAFKA_BOOTSTRAP_SERVERS" in result.skip_reason

    def test_empty_string(self) -> None:
        """Test validation fails for empty string."""
        result = validate_bootstrap_servers("")
        assert result.is_valid is False
        assert result.host == "<not set>"
        assert result.error_message is not None
        assert "empty" in result.error_message.lower()
        assert result.skip_reason is not None
        assert "KAFKA_BOOTSTRAP_SERVERS" in result.skip_reason

    def test_whitespace_only(self) -> None:
        """Test validation fails for whitespace-only string."""
        result = validate_bootstrap_servers("   \t\n  ")
        assert result.is_valid is False
        assert result.host == "<not set>"
        assert result.error_message is not None
        assert result.skip_reason is not None

    def test_non_numeric_port(self) -> None:
        """Test validation fails for non-numeric port."""
        result = validate_bootstrap_servers("localhost:abc")
        assert result.is_valid is False
        # Host is preserved from the first entry for better error messages
        assert result.host == "localhost"
        assert result.port == "abc"
        assert result.error_message is not None
        assert "numeric" in result.error_message.lower()
        assert result.skip_reason is not None

    def test_port_zero(self) -> None:
        """Test validation fails for port 0."""
        result = validate_bootstrap_servers("localhost:0")
        assert result.is_valid is False
        assert result.error_message is not None
        assert "1-65535" in result.error_message

    def test_port_too_high(self) -> None:
        """Test validation fails for port > 65535."""
        result = validate_bootstrap_servers("localhost:65536")
        assert result.is_valid is False
        assert result.error_message is not None
        assert "1-65535" in result.error_message

    def test_port_boundary_low(self) -> None:
        """Test validation passes for port 1 (minimum valid)."""
        result = validate_bootstrap_servers("localhost:1")
        assert result.is_valid is True
        assert result.port == "1"

    def test_port_boundary_high(self) -> None:
        """Test validation passes for port 65535 (maximum valid)."""
        result = validate_bootstrap_servers("localhost:65535")
        assert result.is_valid is True
        assert result.port == "65535"

    def test_ipv6_with_valid_port(self) -> None:
        """Test validation passes for IPv6 address with valid port."""
        result = validate_bootstrap_servers("[::1]:9092")
        assert result.is_valid is True
        assert result.host == "[::1]"
        assert result.port == "9092"

    def test_ipv6_without_port(self) -> None:
        """Test validation passes for IPv6 without explicit port."""
        result = validate_bootstrap_servers("[2001:db8::1]")
        assert result.is_valid is True
        assert result.host == "[2001:db8::1]"
        assert result.port == "29092"  # Default

    # =========================================================================
    # Bare IPv6 Address Validation Tests
    # =========================================================================

    def test_bare_ipv6_localhost_valid(self) -> None:
        """Test validation passes for bare IPv6 localhost."""
        result = validate_bootstrap_servers("::1")
        assert result.is_valid is True
        assert result.host == "::1"
        assert result.port == "29092"  # Default port for bare IPv6
        assert result.error_message is None

    def test_bare_ipv6_full_address_valid(self) -> None:
        """Test validation passes for bare IPv6 full address."""
        result = validate_bootstrap_servers("2001:db8::1")
        assert result.is_valid is True
        assert result.host == "2001:db8::1"
        assert result.port == "29092"  # Default port for bare IPv6

    def test_bare_ipv6_mapped_ipv4_valid(self) -> None:
        """Test validation passes for IPv4-mapped IPv6 address."""
        result = validate_bootstrap_servers("::ffff:192.168.1.1")
        assert result.is_valid is True
        assert result.host == "::ffff:192.168.1.1"
        assert result.port == "29092"  # Default port for bare IPv6

    def test_bare_ipv6_ambiguous_port_like_suffix_valid(self) -> None:
        """Test validation passes for bare IPv6 with port-like suffix.

        The string "::1:9092" is ambiguous and is treated as a bare IPv6 address
        (the entire string is the host, not "::1" with port "9092").
        For unambiguous IPv6 with port, use bracketed format: [::1]:9092
        """
        result = validate_bootstrap_servers("::1:9092")
        assert result.is_valid is True
        assert result.host == "::1:9092"  # Entire string is the host
        assert result.port == "29092"  # Default port since it's bare IPv6
        assert result.error_message is None

    def test_skip_reason_contains_example(self) -> None:
        """Test skip reason contains helpful example."""
        result = validate_bootstrap_servers("")
        assert result.skip_reason is not None
        assert "export KAFKA_BOOTSTRAP_SERVERS" in result.skip_reason
        assert "29092" in result.skip_reason

    # =========================================================================
    # Comma-Separated Server List Tests
    # =========================================================================

    def test_multiple_servers_valid(self) -> None:
        """Test validation passes for comma-separated server list."""
        result = validate_bootstrap_servers("server1:9092,server2:9093")
        assert result.is_valid is True
        # First server's host/port is returned
        assert result.host == "server1"
        assert result.port == "9092"
        assert result.error_message is None

    def test_multiple_servers_with_whitespace(self) -> None:
        """Test validation passes with whitespace around commas."""
        result = validate_bootstrap_servers("server1:9092 , server2:9093")
        assert result.is_valid is True
        assert result.host == "server1"
        assert result.port == "9092"

    def test_trailing_comma_handled(self) -> None:
        """Test validation passes with trailing comma (empty entry filtered)."""
        result = validate_bootstrap_servers("server1:9092,")
        assert result.is_valid is True
        assert result.host == "server1"
        assert result.port == "9092"

    def test_leading_comma_handled(self) -> None:
        """Test validation passes with leading comma (empty entry filtered)."""
        result = validate_bootstrap_servers(",server1:9092")
        assert result.is_valid is True
        assert result.host == "server1"
        assert result.port == "9092"

    def test_double_comma_handled(self) -> None:
        """Test validation passes with double comma (empty entry filtered)."""
        result = validate_bootstrap_servers("server1:9092,,server2:9093")
        assert result.is_valid is True
        assert result.host == "server1"
        assert result.port == "9092"

    def test_only_commas_invalid(self) -> None:
        """Test validation fails for only commas."""
        result = validate_bootstrap_servers(",,,")
        assert result.is_valid is False
        assert result.host == "<not set>"
        assert "no valid server entries" in result.skip_reason.lower()

    def test_commas_and_whitespace_invalid(self) -> None:
        """Test validation fails for commas and whitespace only."""
        result = validate_bootstrap_servers(", , , ")
        assert result.is_valid is False
        assert result.host == "<not set>"

    def test_one_valid_one_invalid_server(self) -> None:
        """Test validation fails when one server has invalid port."""
        result = validate_bootstrap_servers("server1:9092,server2:abc")
        assert result.is_valid is False
        # First valid server's host is preserved
        assert result.host == "server1"
        assert result.port == "9092"
        assert "invalid entries" in result.error_message.lower()
        assert "server2:abc" in result.error_message

    def test_all_servers_invalid(self) -> None:
        """Test validation fails when all servers have invalid ports."""
        result = validate_bootstrap_servers("server1:abc,server2:xyz")
        assert result.is_valid is False
        # First entry's host is preserved for error messages
        assert result.host == "server1"
        assert result.port == "abc"
        assert "invalid entries" in result.error_message.lower()

    def test_three_servers_valid(self) -> None:
        """Test validation passes for three comma-separated servers."""
        result = validate_bootstrap_servers("s1:9092,s2:9093,s3:9094")
        assert result.is_valid is True
        assert result.host == "s1"
        assert result.port == "9092"

    def test_mixed_ipv4_and_ipv6_servers(self) -> None:
        """Test validation passes for mixed IPv4 and IPv6 servers."""
        result = validate_bootstrap_servers("192.168.1.1:9092,[::1]:9093")
        assert result.is_valid is True
        assert result.host == "192.168.1.1"
        assert result.port == "9092"


class TestKafkaConfigValidationResultBool:
    """Tests for KafkaConfigValidationResult __bool__ method."""

    def test_bool_true_when_valid(self) -> None:
        """Test __bool__ returns True when is_valid is True."""
        result = KafkaConfigValidationResult(
            is_valid=True,
            host="localhost",
            port="9092",
        )
        assert bool(result) is True
        assert result  # Direct truthiness check

    def test_bool_false_when_invalid(self) -> None:
        """Test __bool__ returns False when is_valid is False."""
        result = KafkaConfigValidationResult(
            is_valid=False,
            host="<not set>",
            port="29092",
            error_message="Not configured",
            skip_reason="Skip reason",
        )
        assert bool(result) is False
        assert not result  # Direct falsiness check

    def test_conditional_usage_pattern(self) -> None:
        """Test typical usage pattern with if statement."""
        valid_result = validate_bootstrap_servers("localhost:9092")
        invalid_result = validate_bootstrap_servers("")

        # This is the expected usage pattern in fixtures
        if valid_result:
            # Should enter this branch
            assert valid_result.is_valid
        else:
            pytest.fail("Valid result should be truthy")

        if not invalid_result:
            # Should enter this branch
            assert not invalid_result.is_valid
        else:
            pytest.fail("Invalid result should be falsy")
