# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for event forward effect models."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_event_forward_effect.models import (
    ModelEventForwardRequest,
    ModelEventForwardResult,
)


@pytest.mark.unit
class TestModelEventForwardRequest:
    def test_defaults(self) -> None:
        req = ModelEventForwardRequest(event_type="service.started")
        assert req.category == "generic"
        assert req.severity == "info"
        assert req.payload == {}
        assert req.metadata == {}
        assert req.correlation_id is not None

    def test_frozen(self) -> None:
        req = ModelEventForwardRequest(event_type="service.started")
        with pytest.raises(Exception):
            req.event_type = "changed"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        with pytest.raises(Exception):
            ModelEventForwardRequest(event_type="test", bogus="nope")  # type: ignore[call-arg]

    def test_category_enum(self) -> None:
        for cat in ("lifecycle", "system", "tool", "generic"):
            req = ModelEventForwardRequest(event_type="test", category=cat)  # type: ignore[arg-type]
            assert req.category == cat
        with pytest.raises(Exception):
            ModelEventForwardRequest(event_type="test", category="invalid")  # type: ignore[arg-type]


@pytest.mark.unit
class TestModelEventForwardResult:
    def test_success(self) -> None:
        result = ModelEventForwardResult(
            correlation_id=uuid4(),
            success=True,
            http_status=200,
            endpoint="/api/events/service-lifecycle",
        )
        assert result.success is True
        assert result.http_status == 200

    def test_failure(self) -> None:
        result = ModelEventForwardResult(
            correlation_id=uuid4(),
            success=False,
            error_message="connection refused",
        )
        assert result.success is False
        assert result.http_status == 0

    def test_frozen(self) -> None:
        result = ModelEventForwardResult(correlation_id=uuid4(), success=True)
        with pytest.raises(Exception):
            result.success = False  # type: ignore[misc]
