# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test validate_topic uses contract config for topic validation [OMN-6342]."""

from __future__ import annotations

import pytest

from omnibase_infra.errors import ProtocolConfigurationError


@pytest.mark.unit
def test_topic_validation_rejects_denied_patterns() -> None:
    """Topic names matching deny patterns must be rejected."""
    from omnibase_infra.runtime.event_bus_subcontract_wiring import (
        validate_topic,
    )

    with pytest.raises(ProtocolConfigurationError, match="denied"):
        validate_topic(
            "onex.${env.SECRET}",
            deny_patterns=(r"\$\{", r"\.\./"),
        )


@pytest.mark.unit
def test_topic_validation_allows_valid_topics() -> None:
    """Valid topic names must pass validation."""
    from omnibase_infra.runtime.event_bus_subcontract_wiring import (
        validate_topic,
    )

    # Should not raise
    validate_topic(
        "onex.evt.myapp.event.v1",
        deny_patterns=(r"\$\{",),
    )


@pytest.mark.unit
def test_topic_validation_empty_patterns_allows_all() -> None:
    """With no deny patterns, all topics pass."""
    from omnibase_infra.runtime.event_bus_subcontract_wiring import (
        validate_topic,
    )

    # Default: no patterns = allow all
    validate_topic("anything.goes.here")
