# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration contract for ModelHealthCheckResponse tuple serialization (OMN-11810).

Regression gate: strict=True was removed from ModelHealthCheckResponse.model_config
to fix /ready returning HTTP 503. The root cause was ModelEventBusReadiness
producing required_topics as tuple[str, ...] which Pydantic strict mode rejected
when the details dict was validated against ModelHealthCheckResponse.

These tests prove the fix holds at the model-contract level without a live runtime.
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.models.model_health_check_response import (
    ModelHealthCheckResponse,
)

pytestmark = pytest.mark.integration


def test_health_response_accepts_readiness_details_with_tuple_topics() -> None:
    """ModelHealthCheckResponse.success() must not raise when details contain tuples.

    Before the fix (strict=True), Pydantic raised ValidationError because
    ModelEventBusReadiness.required_topics typed as tuple[str, ...] was
    preserved by model_dump() without mode='json' and then rejected as
    non-list by strict mode validation.
    """
    readiness_details: dict[str, object] = {
        "ready": True,
        "event_bus_readiness": {
            "healthy": True,
            "required_topics": ("onex.cmd.test.v1", "onex.evt.test.v1"),
            "missing_topics": (),
            "extra_topics": (),
        },
    }
    resp = ModelHealthCheckResponse.success(
        status="healthy",
        version="1.0.0",
        details=readiness_details,  # type: ignore[arg-type]
    )
    assert resp.status == "healthy"
    assert resp.details is not None


def test_health_response_json_serializable_with_tuple_topics() -> None:
    """ModelHealthCheckResponse with tuple in details must produce valid JSON."""
    readiness_details: dict[str, object] = {
        "ready": True,
        "event_bus_readiness": {
            "healthy": True,
            "required_topics": ("onex.cmd.test.v1",),
        },
    }
    resp = ModelHealthCheckResponse.success(
        status="healthy",
        version="1.0.0",
        details=readiness_details,  # type: ignore[arg-type]
    )
    serialized = resp.model_dump_json()
    assert "healthy" in serialized
