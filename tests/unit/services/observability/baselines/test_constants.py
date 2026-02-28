# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for baselines constants and parse_execute_count.

Ticket: OMN-3041
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.services.observability.baselines.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_QUERY_TIMEOUT,
    TREATMENT_CONFIDENCE_THRESHOLD,
    parse_execute_count,
)


class TestConstants:
    """Tests for module-level constants."""

    def test_default_batch_size(self) -> None:
        assert DEFAULT_BATCH_SIZE == 500

    def test_default_query_timeout(self) -> None:
        assert DEFAULT_QUERY_TIMEOUT == 60.0

    def test_treatment_confidence_threshold(self) -> None:
        assert TREATMENT_CONFIDENCE_THRESHOLD == 0.8


class TestParseExecuteCount:
    """Tests for parse_execute_count covering all 5 variants."""

    def test_parse_insert_string(self) -> None:
        """INSERT 0 42 -> 42"""
        assert parse_execute_count("INSERT 0 42") == 42

    def test_parse_update_string(self) -> None:
        """UPDATE 15 -> 15"""
        assert parse_execute_count("UPDATE 15") == 15

    def test_parse_delete_string(self) -> None:
        """DELETE 0 -> 0"""
        assert parse_execute_count("DELETE 0") == 0

    def test_parse_int_variant(self) -> None:
        """int direct return"""
        assert parse_execute_count(7) == 7  # type: ignore[arg-type]

    def test_parse_none_returns_zero(self) -> None:
        """None -> 0"""
        assert parse_execute_count(None) == 0  # type: ignore[arg-type]

    def test_parse_zero_insert(self) -> None:
        """INSERT 0 0 -> 0"""
        assert parse_execute_count("INSERT 0 0") == 0

    def test_parse_empty_string(self) -> None:
        """empty string -> 0"""
        assert parse_execute_count("") == 0

    def test_parse_invalid_string(self) -> None:
        """non-parseable string -> 0"""
        assert parse_execute_count("not a valid result") == 0

    def test_parse_non_string_non_int_returns_zero(self) -> None:
        """non-string, non-int -> 0"""
        assert parse_execute_count(3.14) == 0  # type: ignore[arg-type]
