# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed lifecycle request preserves transitions and correlation IDs."""

import inspect
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumLifecycleTier, EnumValidationVerdict
from omnibase_infra.nodes.node_pattern_lifecycle_effect.handlers.handler_lifecycle_update import (
    HandlerLifecycleUpdate,
)
from omnibase_infra.nodes.node_pattern_lifecycle_effect.models import (
    ModelLifecycleState,
    ModelLifecycleUpdateRequest,
)


async def test_handle_request_promotes_pattern() -> None:
    pattern_id = uuid4()
    correlation_id = uuid4()
    state = ModelLifecycleState(pattern_id=pattern_id, consecutive_pass_count=1)
    request = ModelLifecycleUpdateRequest(
        pattern_id=pattern_id,
        verdict=EnumValidationVerdict.PASS,
        correlation_id=correlation_id,
        current_state=state,
    )

    result = await HandlerLifecycleUpdate().handle(request)

    assert result.pattern_id == pattern_id
    assert result.correlation_id == correlation_id
    assert result.previous_tier == EnumLifecycleTier.OBSERVED
    assert result.new_tier == EnumLifecycleTier.SUGGESTED
    assert result.tier_changed is True
    assert result.verdict_applied == EnumValidationVerdict.PASS
    assert result.error == ""


async def test_handle_request_defaults_create_state_and_correlation() -> None:
    request = ModelLifecycleUpdateRequest(
        pattern_id=uuid4(), verdict=EnumValidationVerdict.PASS
    )

    result = await HandlerLifecycleUpdate().handle(request)

    assert result.pattern_id == request.pattern_id
    assert isinstance(result.correlation_id, UUID)
    assert result.new_tier == result.previous_tier == EnumLifecycleTier.OBSERVED
    assert result.tier_changed is False
    assert result.error == ""


def test_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelLifecycleUpdateRequest.model_validate(
            {"pattern_id": uuid4(), "verdict": "PASS", "unknown_field": True}
        )


def test_handle_has_one_request_parameter() -> None:
    assert list(inspect.signature(HandlerLifecycleUpdate().handle).parameters) == [
        "request"
    ]
