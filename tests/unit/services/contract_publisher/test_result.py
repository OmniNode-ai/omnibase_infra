# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for contract publisher result models."""

from __future__ import annotations

import pytest

from omnibase_infra.services.contract_publisher import (
    ModelContractError,
    ModelInfraError,
    ModelPublishResult,
    ModelPublishStats,
)


class TestModelContractError:
    """Tests for ModelContractError."""

    def test_yaml_parse_error(self) -> None:
        """Test yaml_parse error type."""
        error = ModelContractError(
            contract_path="/app/contracts/foo/contract.yaml",
            handler_id=None,
            error_type="yaml_parse",
            message="Invalid YAML at line 10",
        )
        assert error.error_type == "yaml_parse"
        assert error.handler_id is None

    def test_schema_validation_error(self) -> None:
        """Test schema_validation error type."""
        error = ModelContractError(
            contract_path="/app/contracts/foo/contract.yaml",
            handler_id="foo.handler",
            error_type="schema_validation",
            message="Missing required field: name",
        )
        assert error.error_type == "schema_validation"
        assert error.handler_id == "foo.handler"

    def test_duplicate_conflict_error(self) -> None:
        """Test duplicate_conflict error type."""
        error = ModelContractError(
            contract_path="/app/contracts/bar/contract.yaml",
            handler_id="foo.handler",
            error_type="duplicate_conflict",
            message="Same handler_id with different content",
        )
        assert error.error_type == "duplicate_conflict"


class TestModelInfraError:
    """Tests for ModelInfraError."""

    def test_publisher_unavailable(self) -> None:
        """Test publisher_unavailable error."""
        error = ModelInfraError(
            error_type="publisher_unavailable",
            message="Event bus publisher not available",
            retriable=False,
        )
        assert error.error_type == "publisher_unavailable"
        assert error.retriable is False

    def test_kafka_timeout(self) -> None:
        """Test kafka_timeout error."""
        error = ModelInfraError(
            error_type="kafka_timeout",
            message="Publish timed out after 30s",
            retriable=True,
        )
        assert error.error_type == "kafka_timeout"
        assert error.retriable is True

    def test_default_retriable_is_false(self) -> None:
        """Test default retriable value."""
        error = ModelInfraError(
            error_type="publish_failed",
            message="Failed to publish",
        )
        assert error.retriable is False


class TestModelPublishStats:
    """Tests for ModelPublishStats."""

    def test_all_fields(self) -> None:
        """Test stats with all fields."""
        stats = ModelPublishStats(
            discovered_count=10,
            valid_count=8,
            published_count=8,
            errored_count=2,
            dedup_count=1,
            duration_ms=1234.5,
            discover_ms=100.0,
            validate_ms=500.0,
            publish_ms=634.5,
            environment="dev",
            filesystem_count=6,
            package_count=4,
        )
        assert stats.discovered_count == 10
        assert stats.valid_count == 8
        assert stats.published_count == 8
        assert stats.errored_count == 2
        assert stats.dedup_count == 1
        assert stats.duration_ms == 1234.5
        assert stats.filesystem_count == 6
        assert stats.package_count == 4

    def test_default_per_origin_counts(self) -> None:
        """Test default per-origin counts are 0."""
        stats = ModelPublishStats(
            discovered_count=5,
            valid_count=5,
            published_count=5,
            errored_count=0,
            dedup_count=0,
            duration_ms=100.0,
            discover_ms=50.0,
            validate_ms=30.0,
            publish_ms=20.0,
            environment="dev",
        )
        assert stats.filesystem_count == 0
        assert stats.package_count == 0

    def test_non_negative_constraints(self) -> None:
        """Test stats enforce non-negative values."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelPublishStats(
                discovered_count=-1,
                valid_count=0,
                published_count=0,
                errored_count=0,
                dedup_count=0,
                duration_ms=0.0,
                discover_ms=0.0,
                validate_ms=0.0,
                publish_ms=0.0,
                environment="dev",
            )


class TestModelPublishResult:
    """Tests for ModelPublishResult."""

    def test_successful_result(self) -> None:
        """Test result with successful publishes."""
        stats = ModelPublishStats(
            discovered_count=3,
            valid_count=3,
            published_count=3,
            errored_count=0,
            dedup_count=0,
            duration_ms=100.0,
            discover_ms=30.0,
            validate_ms=30.0,
            publish_ms=40.0,
            environment="dev",
        )
        result = ModelPublishResult(
            published=["handler.a", "handler.b", "handler.c"],
            contract_errors=[],
            infra_errors=[],
            stats=stats,
        )
        assert len(result.published) == 3
        assert result.has_errors is False
        assert bool(result) is True

    def test_bool_true_when_published(self) -> None:
        """Test __bool__ returns True when contracts published."""
        stats = ModelPublishStats(
            discovered_count=1,
            valid_count=1,
            published_count=1,
            errored_count=0,
            dedup_count=0,
            duration_ms=10.0,
            discover_ms=3.0,
            validate_ms=3.0,
            publish_ms=4.0,
            environment="dev",
        )
        result = ModelPublishResult(
            published=["handler.a"],
            stats=stats,
        )
        assert bool(result) is True

    def test_bool_false_when_empty(self) -> None:
        """Test __bool__ returns False when no contracts published."""
        stats = ModelPublishStats(
            discovered_count=1,
            valid_count=0,
            published_count=0,
            errored_count=1,
            dedup_count=0,
            duration_ms=10.0,
            discover_ms=3.0,
            validate_ms=7.0,
            publish_ms=0.0,
            environment="dev",
        )
        result = ModelPublishResult(
            published=[],
            contract_errors=[
                ModelContractError(
                    contract_path="/app/foo/contract.yaml",
                    handler_id=None,
                    error_type="yaml_parse",
                    message="Invalid YAML",
                )
            ],
            stats=stats,
        )
        assert bool(result) is False

    def test_has_contract_errors(self) -> None:
        """Test has_contract_errors property."""
        stats = ModelPublishStats(
            discovered_count=2,
            valid_count=1,
            published_count=1,
            errored_count=1,
            dedup_count=0,
            duration_ms=10.0,
            discover_ms=3.0,
            validate_ms=4.0,
            publish_ms=3.0,
            environment="dev",
        )
        result = ModelPublishResult(
            published=["handler.a"],
            contract_errors=[
                ModelContractError(
                    contract_path="/app/bar/contract.yaml",
                    handler_id=None,
                    error_type="yaml_parse",
                    message="Invalid YAML",
                )
            ],
            stats=stats,
        )
        assert result.has_contract_errors is True
        assert result.has_infra_errors is False
        assert result.has_errors is True

    def test_has_infra_errors(self) -> None:
        """Test has_infra_errors property."""
        stats = ModelPublishStats(
            discovered_count=1,
            valid_count=1,
            published_count=0,
            errored_count=0,
            dedup_count=0,
            duration_ms=10.0,
            discover_ms=3.0,
            validate_ms=3.0,
            publish_ms=4.0,
            environment="dev",
        )
        result = ModelPublishResult(
            published=[],
            infra_errors=[
                ModelInfraError(
                    error_type="kafka_timeout",
                    message="Timeout",
                    retriable=True,
                )
            ],
            stats=stats,
        )
        assert result.has_contract_errors is False
        assert result.has_infra_errors is True
        assert result.has_errors is True
