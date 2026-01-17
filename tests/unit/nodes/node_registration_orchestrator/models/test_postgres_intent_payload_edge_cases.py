# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for edge case handling in ModelPostgresIntentPayload.

This module tests the explicit handling of edge cases in the endpoints field
validator to ensure invalid input is not silently masked.

Edge Cases Tested:
    - None input: Should raise ValidationError
    - Empty Mapping {}: Should warn and convert to empty tuple
    - Empty tuple (): Should pass through (same as default)
    - Invalid types (list, int, str): Should raise ValidationError
    - Non-empty Mapping: Should convert to tuple of (key, value) pairs

Related:
    - PR #92 review feedback: Fix silent fallback to empty tuple
    - Module: model_postgres_intent_payload.py

.. versionadded:: 0.7.0
    Created as part of PR #92 review to address silent fallback issue.
"""

from __future__ import annotations

import warnings
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.nodes.node_registration_orchestrator.models.model_postgres_intent_payload import (
    ModelPostgresIntentPayload,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def base_payload_kwargs() -> dict[str, object]:
    """Return base kwargs for ModelPostgresIntentPayload without endpoints.

    These are the minimum required fields for creating a valid payload.
    """
    return {
        "node_id": uuid4(),
        "node_type": EnumNodeKind.EFFECT,
        "correlation_id": uuid4(),
        "timestamp": "2025-01-01T00:00:00Z",
    }


# ============================================================================
# Tests for endpoints field validator edge cases
# ============================================================================


@pytest.mark.unit
class TestEndpointsValidatorEdgeCases:
    """Tests for edge case handling in the endpoints field validator.

    The validator should explicitly handle all input types rather than
    silently falling back to empty tuple, which could mask invalid input.
    """

    # ------------------------------------------------------------------------
    # None input tests
    # ------------------------------------------------------------------------

    def test_none_input_raises_validation_error(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify that None input raises ValidationError, not silently ignored.

        None is an invalid input type and should be explicitly rejected
        rather than silently converted to an empty tuple.
        """
        with pytest.raises(ValidationError) as exc_info:
            ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints=None,
            )

        # Verify the error message indicates the type problem
        error_str = str(exc_info.value)
        assert "endpoints" in error_str
        assert "NoneType" in error_str or "tuple or Mapping" in error_str

    def test_none_input_error_message_is_descriptive(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify the error message for None input is descriptive."""
        with pytest.raises(ValidationError) as exc_info:
            ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints=None,
            )

        # Error should explain what was expected
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("endpoints",)
        assert "tuple or Mapping" in errors[0]["msg"]

    # ------------------------------------------------------------------------
    # Empty Mapping tests
    # ------------------------------------------------------------------------

    def test_empty_dict_emits_warning(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify that empty dict {} emits a warning when coerced.

        Empty Mapping coercion should not be silent - it should warn
        to help detect cases where endpoints were expected but not populated.
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            payload = ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints={},
            )

            # Verify warning was emitted
            assert len(caught_warnings) >= 1
            warning_messages = [str(w.message) for w in caught_warnings]
            assert any(
                "Empty Mapping provided for endpoints" in msg
                for msg in warning_messages
            )

            # Verify the result is empty tuple
            assert payload.endpoints == ()

    def test_empty_dict_converts_to_empty_tuple(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify that empty dict {} is converted to empty tuple.

        While a warning is emitted, the conversion should still succeed
        for backwards compatibility.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warning for this test

            payload = ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints={},
            )

            assert payload.endpoints == ()
            assert isinstance(payload.endpoints, tuple)

    def test_empty_dict_warning_is_user_warning(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify the warning category is UserWarning."""
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints={},
            )

            # Find the relevant warning
            endpoint_warnings = [
                w
                for w in caught_warnings
                if "Empty Mapping provided for endpoints" in str(w.message)
            ]
            assert len(endpoint_warnings) >= 1
            assert endpoint_warnings[0].category is UserWarning

    # ------------------------------------------------------------------------
    # Empty tuple tests
    # ------------------------------------------------------------------------

    def test_empty_tuple_passes_through_without_warning(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify that empty tuple () passes through without warning.

        Empty tuple is the default value and is a valid, intentional input.
        It should not trigger any warning.
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            payload = ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints=(),
            )

            # No warnings should be emitted for empty tuple
            endpoint_warnings = [
                w for w in caught_warnings if "endpoints" in str(w.message).lower()
            ]
            assert len(endpoint_warnings) == 0

            assert payload.endpoints == ()

    def test_default_endpoints_is_empty_tuple(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify that not providing endpoints uses default empty tuple.

        When endpoints is not provided, it should default to () without
        any warning.
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            # Don't provide endpoints - should use default
            payload = ModelPostgresIntentPayload(**base_payload_kwargs)

            # No warnings for using default
            endpoint_warnings = [
                w for w in caught_warnings if "endpoints" in str(w.message).lower()
            ]
            assert len(endpoint_warnings) == 0

            assert payload.endpoints == ()

    # ------------------------------------------------------------------------
    # Invalid type tests
    # ------------------------------------------------------------------------

    def test_list_input_raises_validation_error(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify that list input raises ValidationError.

        Lists are not Mappings and should be explicitly rejected.
        """
        with pytest.raises(ValidationError) as exc_info:
            ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints=[("health", "/health")],  # List, not tuple or dict
            )

        error_str = str(exc_info.value)
        assert "endpoints" in error_str
        assert "list" in error_str.lower()

    def test_empty_list_raises_validation_error(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify that empty list [] raises ValidationError.

        Even empty list is an invalid type (not a Mapping).
        """
        with pytest.raises(ValidationError) as exc_info:
            ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints=[],
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("endpoints",)

    def test_int_input_raises_validation_error(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify that int input raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints=123,
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "int" in errors[0]["msg"]

    def test_string_input_raises_validation_error(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify that string input raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints="invalid",
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "str" in errors[0]["msg"]

    def test_set_input_raises_validation_error(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify that set input raises ValidationError.

        Sets are not Mappings.
        """
        with pytest.raises(ValidationError) as exc_info:
            ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints={"health", "api"},  # Set, not dict
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1

    # ------------------------------------------------------------------------
    # Valid Mapping tests
    # ------------------------------------------------------------------------

    def test_non_empty_dict_converts_without_warning(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify that non-empty dict converts to tuple without warning.

        Non-empty Mapping is a valid input and should be converted
        without any warning.
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            payload = ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints={"health": "/health", "api": "/api/v1"},
            )

            # No warnings for non-empty dict
            endpoint_warnings = [
                w for w in caught_warnings if "endpoints" in str(w.message).lower()
            ]
            assert len(endpoint_warnings) == 0

            # Verify conversion
            assert payload.endpoints == (("health", "/health"), ("api", "/api/v1"))

    def test_single_endpoint_dict_converts_correctly(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify single-entry dict converts correctly."""
        payload = ModelPostgresIntentPayload(
            **base_payload_kwargs,
            endpoints={"health": "/health"},
        )

        assert payload.endpoints == (("health", "/health"),)
        assert len(payload.endpoints) == 1

    def test_endpoints_dict_property_works_after_coercion(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify endpoints_dict property returns correct read-only view."""
        payload = ModelPostgresIntentPayload(
            **base_payload_kwargs,
            endpoints={"health": "/health", "api": "/api"},
        )

        endpoints_view = payload.endpoints_dict
        assert endpoints_view["health"] == "/health"
        assert endpoints_view["api"] == "/api"
        assert len(endpoints_view) == 2

    def test_non_string_keys_raises_validation_error(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify non-string dict keys raise ValidationError in strict mode.

        Per PR #92 review: validators now use strict mode that rejects non-string
        keys instead of silently coercing them to strings. This ensures explicit
        type handling rather than masking potential data issues.
        """
        with pytest.raises(ValidationError) as exc_info:
            ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints={1: "/one", 2: "/two"},  # type: ignore[dict-item]
            )

        # Verify error indicates the type problem
        error_str = str(exc_info.value)
        assert "endpoints" in error_str
        assert "key must be a string" in error_str
        assert "int" in error_str

    def test_non_string_values_raises_validation_error(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify non-string dict values raise ValidationError in strict mode.

        Per PR #92 review: validators now use strict mode that rejects non-string
        values instead of silently coercing them to strings. This ensures explicit
        type handling rather than masking potential data issues.
        """
        with pytest.raises(ValidationError) as exc_info:
            ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints={"port": 8080, "timeout": 30},  # type: ignore[dict-item]
            )

        # Verify error indicates the type problem
        error_str = str(exc_info.value)
        assert "endpoints" in error_str
        assert "value must be a string" in error_str
        assert "int" in error_str

    # ------------------------------------------------------------------------
    # Tuple passthrough tests
    # ------------------------------------------------------------------------

    def test_tuple_of_pairs_passes_through(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify tuple of pairs passes through unchanged."""
        input_tuple = (("health", "/health"), ("api", "/api"))

        payload = ModelPostgresIntentPayload(
            **base_payload_kwargs,
            endpoints=input_tuple,
        )

        assert payload.endpoints == input_tuple
        assert payload.endpoints is not input_tuple  # Pydantic creates copy

    def test_nested_tuple_with_single_pair(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify tuple with single pair works correctly."""
        payload = ModelPostgresIntentPayload(
            **base_payload_kwargs,
            endpoints=(("health", "/health"),),
        )

        assert payload.endpoints == (("health", "/health"),)
        assert len(payload.endpoints) == 1


# ============================================================================
# Tests for immutability and thread safety
# ============================================================================


@pytest.mark.unit
class TestEndpointsImmutability:
    """Tests for immutability of the endpoints field after validation."""

    def test_endpoints_is_tuple_not_dict(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify endpoints is stored as tuple, not dict."""
        payload = ModelPostgresIntentPayload(
            **base_payload_kwargs,
            endpoints={"health": "/health"},
        )

        assert isinstance(payload.endpoints, tuple)
        assert not isinstance(payload.endpoints, dict)

    def test_model_is_frozen(self, base_payload_kwargs: dict[str, object]) -> None:
        """Verify the model is frozen (immutable)."""
        payload = ModelPostgresIntentPayload(
            **base_payload_kwargs,
            endpoints={"health": "/health"},
        )

        # Attempting to modify should raise error
        with pytest.raises(ValidationError):
            payload.endpoints = ()  # type: ignore[misc]

    def test_endpoints_dict_returns_mapping_proxy(
        self, base_payload_kwargs: dict[str, object]
    ) -> None:
        """Verify endpoints_dict returns a MappingProxyType (read-only)."""
        from types import MappingProxyType

        payload = ModelPostgresIntentPayload(
            **base_payload_kwargs,
            endpoints={"health": "/health"},
        )

        endpoints_view = payload.endpoints_dict
        assert isinstance(endpoints_view, MappingProxyType)

        # Attempting to modify should raise error
        with pytest.raises(TypeError):
            endpoints_view["new_key"] = "/new"  # type: ignore[index]


# ============================================================================
# Tests for logging behavior
# ============================================================================


@pytest.mark.unit
class TestEndpointsLogging:
    """Tests for logging behavior in the endpoints validator."""

    def test_empty_dict_logs_warning(
        self,
        base_payload_kwargs: dict[str, object],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify that empty dict logs a warning message."""
        import logging

        with caplog.at_level(logging.WARNING):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress UserWarning
                ModelPostgresIntentPayload(
                    **base_payload_kwargs,
                    endpoints={},
                )

        # Check log contains the warning
        assert any(
            "Empty Mapping provided for endpoints" in record.message
            for record in caplog.records
        )

    def test_non_empty_dict_does_not_log_warning(
        self,
        base_payload_kwargs: dict[str, object],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify that non-empty dict does not log any warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints={"health": "/health"},
            )

        # Check no endpoint-related warnings
        endpoint_warnings = [
            r for r in caplog.records if "endpoints" in r.message.lower()
        ]
        assert len(endpoint_warnings) == 0

    def test_empty_tuple_does_not_log_warning(
        self,
        base_payload_kwargs: dict[str, object],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify that empty tuple does not log any warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            ModelPostgresIntentPayload(
                **base_payload_kwargs,
                endpoints=(),
            )

        # Check no endpoint-related warnings
        endpoint_warnings = [
            r for r in caplog.records if "endpoints" in r.message.lower()
        ]
        assert len(endpoint_warnings) == 0
