# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for catalog warning code constants.

Verifies that all 7 warning codes are defined as string constants with the
exact values the DoD specifies, and that the constants are exported from both
the module and the package __init__.

Related Tickets:
    - OMN-2312: Topic Catalog: response warnings channel
"""

from __future__ import annotations

import pytest

import omnibase_infra.models.catalog as catalog_pkg
from omnibase_infra.models.catalog.catalog_warning_codes import (
    CONSUL_KV_MAX_KEYS_REACHED,
    CONSUL_SCAN_TIMEOUT,
    CONSUL_UNAVAILABLE,
    INTERNAL_ERROR,
    INVALID_QUERY_PAYLOAD,
    PARTIAL_NODE_DATA,
    VERSION_UNKNOWN,
)

# ---------------------------------------------------------------------------
# Constant value tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWarningCodeValues:
    """Each warning code has the exact string value specified in OMN-2312."""

    def test_consul_unavailable_value(self) -> None:
        """CONSUL_UNAVAILABLE must equal 'consul_unavailable'."""
        assert CONSUL_UNAVAILABLE == "consul_unavailable"

    def test_consul_scan_timeout_value(self) -> None:
        """CONSUL_SCAN_TIMEOUT must equal 'consul_scan_timeout'."""
        assert CONSUL_SCAN_TIMEOUT == "consul_scan_timeout"

    def test_invalid_query_payload_value(self) -> None:
        """INVALID_QUERY_PAYLOAD must equal 'invalid_query_payload'."""
        assert INVALID_QUERY_PAYLOAD == "invalid_query_payload"

    def test_partial_node_data_value(self) -> None:
        """PARTIAL_NODE_DATA must equal 'partial_node_data'."""
        assert PARTIAL_NODE_DATA == "partial_node_data"

    def test_version_unknown_value(self) -> None:
        """VERSION_UNKNOWN must equal 'version_unknown'."""
        assert VERSION_UNKNOWN == "version_unknown"

    def test_consul_kv_max_keys_reached_value(self) -> None:
        """CONSUL_KV_MAX_KEYS_REACHED must equal 'consul_kv_max_keys_reached'."""
        assert CONSUL_KV_MAX_KEYS_REACHED == "consul_kv_max_keys_reached"

    def test_internal_error_value(self) -> None:
        """INTERNAL_ERROR must equal 'internal_error'."""
        assert INTERNAL_ERROR == "internal_error"

    def test_all_constants_are_strings(self) -> None:
        """All warning codes must be plain str instances."""
        for constant in (
            CONSUL_UNAVAILABLE,
            CONSUL_SCAN_TIMEOUT,
            CONSUL_KV_MAX_KEYS_REACHED,
            INTERNAL_ERROR,
            INVALID_QUERY_PAYLOAD,
            PARTIAL_NODE_DATA,
            VERSION_UNKNOWN,
        ):
            assert isinstance(constant, str), f"{constant!r} is not a str"

    def test_all_constants_are_distinct(self) -> None:
        """All warning codes must be distinct strings (no accidental duplicates)."""
        codes = [
            CONSUL_UNAVAILABLE,
            CONSUL_SCAN_TIMEOUT,
            CONSUL_KV_MAX_KEYS_REACHED,
            INTERNAL_ERROR,
            INVALID_QUERY_PAYLOAD,
            PARTIAL_NODE_DATA,
            VERSION_UNKNOWN,
        ]
        assert len(set(codes)) == len(codes), "Duplicate warning codes detected"


# ---------------------------------------------------------------------------
# Package export tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWarningCodePackageExports:
    """Warning codes are accessible from the catalog package __init__."""

    def test_consul_unavailable_exported_from_package(self) -> None:
        """CONSUL_UNAVAILABLE is in the catalog package namespace."""
        assert hasattr(catalog_pkg, "CONSUL_UNAVAILABLE")
        assert catalog_pkg.CONSUL_UNAVAILABLE == "consul_unavailable"

    def test_consul_scan_timeout_exported_from_package(self) -> None:
        """CONSUL_SCAN_TIMEOUT is in the catalog package namespace."""
        assert hasattr(catalog_pkg, "CONSUL_SCAN_TIMEOUT")
        assert catalog_pkg.CONSUL_SCAN_TIMEOUT == "consul_scan_timeout"

    def test_invalid_query_payload_exported_from_package(self) -> None:
        """INVALID_QUERY_PAYLOAD is in the catalog package namespace."""
        assert hasattr(catalog_pkg, "INVALID_QUERY_PAYLOAD")
        assert catalog_pkg.INVALID_QUERY_PAYLOAD == "invalid_query_payload"

    def test_partial_node_data_exported_from_package(self) -> None:
        """PARTIAL_NODE_DATA is in the catalog package namespace."""
        assert hasattr(catalog_pkg, "PARTIAL_NODE_DATA")
        assert catalog_pkg.PARTIAL_NODE_DATA == "partial_node_data"

    def test_version_unknown_exported_from_package(self) -> None:
        """VERSION_UNKNOWN is in the catalog package namespace."""
        assert hasattr(catalog_pkg, "VERSION_UNKNOWN")
        assert catalog_pkg.VERSION_UNKNOWN == "version_unknown"

    def test_consul_kv_max_keys_reached_exported_from_package(self) -> None:
        """CONSUL_KV_MAX_KEYS_REACHED is in the catalog package namespace."""
        assert hasattr(catalog_pkg, "CONSUL_KV_MAX_KEYS_REACHED")
        assert catalog_pkg.CONSUL_KV_MAX_KEYS_REACHED == "consul_kv_max_keys_reached"

    def test_internal_error_exported_from_package(self) -> None:
        """INTERNAL_ERROR is in the catalog package namespace."""
        assert hasattr(catalog_pkg, "INTERNAL_ERROR")
        assert catalog_pkg.INTERNAL_ERROR == "internal_error"
