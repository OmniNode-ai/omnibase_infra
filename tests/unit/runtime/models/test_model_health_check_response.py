# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelHealthCheckResponse.

Tests validate:
- Factory method construction (success/failure)
- JSON serialization with exclude_none for backward compatibility
- Literal status validation
- Immutability (frozen=True)

.. versionadded:: 1.0.0
    Initial test coverage for ModelHealthCheckResponse.

Related Tickets:
    - PR #111 nitpick: Add Pydantic model for health check response
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_health_check_response import (
    ModelHealthCheckResponse,
)


class TestModelHealthCheckResponseSuccess:
    """Tests for ModelHealthCheckResponse.success() factory method."""

    def test_success_healthy(self) -> None:
        """Test success factory with healthy status."""
        resp = ModelHealthCheckResponse.success(
            status="healthy",
            version="1.0.0",
            details={"healthy": True, "handlers": {}},
        )
        assert resp.status == "healthy"
        assert resp.version == "1.0.0"
        assert resp.details == {"healthy": True, "handlers": {}}
        assert resp.error is None
        assert resp.error_type is None
        assert resp.correlation_id is None

    def test_success_degraded(self) -> None:
        """Test success factory with degraded status."""
        resp = ModelHealthCheckResponse.success(
            status="degraded",
            version="2.0.0",
            details={"healthy": False, "degraded": True},
        )
        assert resp.status == "degraded"
        assert resp.version == "2.0.0"

    def test_success_unhealthy(self) -> None:
        """Test success factory with unhealthy status."""
        resp = ModelHealthCheckResponse.success(
            status="unhealthy",
            version="3.0.0",
            details={"healthy": False, "degraded": False},
        )
        assert resp.status == "unhealthy"


class TestModelHealthCheckResponseFailure:
    """Tests for ModelHealthCheckResponse.failure() factory method."""

    def test_failure_creates_unhealthy_response(self) -> None:
        """Test failure factory creates unhealthy response."""
        resp = ModelHealthCheckResponse.failure(
            version="1.0.0",
            error="Connection refused",
            error_type="ConnectionError",
            correlation_id="abc-123",
        )
        assert resp.status == "unhealthy"
        assert resp.version == "1.0.0"
        assert resp.error == "Connection refused"
        assert resp.error_type == "ConnectionError"
        assert resp.correlation_id == "abc-123"
        assert resp.details is None


class TestModelHealthCheckResponseSerialization:
    """Tests for ModelHealthCheckResponse JSON serialization."""

    def test_success_json_excludes_none(self) -> None:
        """Test success response excludes None fields for backward compatibility."""
        resp = ModelHealthCheckResponse.success(
            status="healthy",
            version="1.0.0",
            details={"healthy": True},
        )
        json_str = resp.model_dump_json(exclude_none=True)

        # Should include these fields
        assert '"status":"healthy"' in json_str
        assert '"version":"1.0.0"' in json_str
        assert '"details"' in json_str

        # Should NOT include these None fields
        assert '"error"' not in json_str
        assert '"error_type"' not in json_str
        assert '"correlation_id"' not in json_str

    def test_failure_json_excludes_none(self) -> None:
        """Test failure response excludes None fields for backward compatibility."""
        resp = ModelHealthCheckResponse.failure(
            version="1.0.0",
            error="Timeout",
            error_type="TimeoutError",
            correlation_id="xyz-789",
        )
        json_str = resp.model_dump_json(exclude_none=True)

        # Should include these fields
        assert '"status":"unhealthy"' in json_str
        assert '"version":"1.0.0"' in json_str
        assert '"error":"Timeout"' in json_str
        assert '"error_type":"TimeoutError"' in json_str
        assert '"correlation_id":"xyz-789"' in json_str

        # Should NOT include None details
        assert '"details":null' not in json_str

    def test_json_roundtrip(self) -> None:
        """Test JSON roundtrip serialization."""
        original = ModelHealthCheckResponse.success(
            status="healthy",
            version="1.0.0",
            details={"healthy": True, "handlers": {"http": True}},
        )
        json_str = original.model_dump_json()
        restored = ModelHealthCheckResponse.model_validate_json(json_str)
        assert original == restored


class TestModelHealthCheckResponseValidation:
    """Tests for ModelHealthCheckResponse field validation."""

    def test_invalid_status_rejected(self) -> None:
        """Test that invalid status values are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelHealthCheckResponse(
                status="invalid",  # type: ignore[arg-type]
                version="1.0.0",
            )
        assert "status" in str(exc_info.value)

    def test_status_required(self) -> None:
        """Test that status is a required field."""
        with pytest.raises(ValidationError) as exc_info:
            ModelHealthCheckResponse(version="1.0.0")  # type: ignore[call-arg]
        assert "status" in str(exc_info.value)

    def test_version_required(self) -> None:
        """Test that version is a required field."""
        with pytest.raises(ValidationError) as exc_info:
            ModelHealthCheckResponse(status="healthy")  # type: ignore[call-arg]
        assert "version" in str(exc_info.value)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ModelHealthCheckResponse(
                status="healthy",
                version="1.0.0",
                unknown_field="unexpected",  # type: ignore[call-arg]
            )
        error_str = str(exc_info.value).lower()
        assert "unknown_field" in error_str or "extra" in error_str


class TestModelHealthCheckResponseImmutability:
    """Tests for ModelHealthCheckResponse immutability (frozen=True)."""

    def test_status_is_immutable(self) -> None:
        """Test that status cannot be modified after creation."""
        resp = ModelHealthCheckResponse.success(
            status="healthy",
            version="1.0.0",
            details={},
        )
        with pytest.raises(ValidationError):
            resp.status = "unhealthy"  # type: ignore[misc]

    def test_frozen_model_with_dict_not_hashable(self) -> None:
        """Test that frozen model with dict field is not hashable.

        Note: Unlike simple frozen models, this model contains a dict field
        (details) which makes it unhashable. This is expected behavior for
        models with mutable types in their fields.
        """
        resp = ModelHealthCheckResponse.success(
            status="healthy",
            version="1.0.0",
            details={"healthy": True},
        )
        # Models with dict fields are not hashable
        with pytest.raises(TypeError, match="unhashable"):
            hash(resp)

    def test_failure_response_is_hashable(self) -> None:
        """Test that failure response without dict is hashable."""
        resp = ModelHealthCheckResponse.failure(
            version="1.0.0",
            error="Connection refused",
            error_type="ConnectionError",
            correlation_id="abc-123",
        )
        # Failure responses have details=None, so they should be hashable
        hash_value = hash(resp)
        assert isinstance(hash_value, int)


class TestModelHealthCheckResponseEquality:
    """Tests for ModelHealthCheckResponse equality comparison."""

    def test_same_values_are_equal(self) -> None:
        """Test that models with same values are equal."""
        resp1 = ModelHealthCheckResponse.success(
            status="healthy",
            version="1.0.0",
            details={"healthy": True},
        )
        resp2 = ModelHealthCheckResponse.success(
            status="healthy",
            version="1.0.0",
            details={"healthy": True},
        )
        assert resp1 == resp2

    def test_different_status_not_equal(self) -> None:
        """Test that different status makes models not equal."""
        resp1 = ModelHealthCheckResponse.success(
            status="healthy",
            version="1.0.0",
            details={},
        )
        resp2 = ModelHealthCheckResponse.success(
            status="degraded",
            version="1.0.0",
            details={},
        )
        assert resp1 != resp2
