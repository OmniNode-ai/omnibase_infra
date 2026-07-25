# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelLlmInferenceRequest resilience field bounds (OMN-15115).

Targets ``omnibase_infra.nodes.node_llm_inference_effect.models
.model_llm_inference_request.ModelLlmInferenceRequest`` specifically -- the
model actually consumed by ``HandlerLlmOpenaiCompatible`` (the HTTP
OpenAI-compatible path used by e.g. the hostile-reviewer local models).

Note: a differently-implemented, same-named ``ModelLlmInferenceRequest``
also exists at ``omnibase_infra.models.llm.model_llm_inference_request``
(consumed by the CLI-subprocess handler path) and already has its own
``timeout_seconds``/``max_retries`` bounds tests under this same test
directory (``test_model_llm_inference_request.py``) -- that file imports
the OTHER model despite living in this directory; do not confuse the two
when extending either.

Reference: OMN-15115 -- qwen3-review-b's timeout was pinned at this model's
previous ``le=600.0`` ceiling (the max the schema allowed), which made a
config-only fix structurally impossible; this raises the ceiling and adds
a ``max_retries`` override so a systematically-slow (not just occasionally
slow) local endpoint can be sized correctly without retrying a doomed call.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumLlmOperationType
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)

pytestmark = [pytest.mark.unit]


def _chat_kwargs(**overrides: object) -> dict[str, object]:
    """Minimal valid kwargs for a CHAT_COMPLETION request."""
    defaults: dict[str, object] = {
        "base_url": "http://192.168.86.201:8001",
        "operation_type": EnumLlmOperationType.CHAT_COMPLETION,
        "model": "test-model",
        "messages": ({"role": "user", "content": "hello"},),
    }
    defaults.update(overrides)
    return defaults


class TestTimeoutSecondsBounds:
    """OMN-15115: timeout_seconds ceiling raised 600.0 -> 1800.0."""

    def test_default_is_30(self) -> None:
        req = ModelLlmInferenceRequest(**_chat_kwargs())
        assert req.timeout_seconds == 30.0

    def test_below_minimum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(timeout_seconds=0.5))

    def test_previous_ceiling_600_now_accepted(self) -> None:
        """The old ceiling value must still be a valid input (no regression)."""
        req = ModelLlmInferenceRequest(**_chat_kwargs(timeout_seconds=600.0))
        assert req.timeout_seconds == 600.0

    def test_value_above_old_ceiling_now_accepted(self) -> None:
        """A value that would have been rejected before OMN-15115 now works --
        this is the entire point of the fix: qwen3-review-b needs 1200.0."""
        req = ModelLlmInferenceRequest(**_chat_kwargs(timeout_seconds=1200.0))
        assert req.timeout_seconds == 1200.0

    def test_new_ceiling_1800_accepted(self) -> None:
        req = ModelLlmInferenceRequest(**_chat_kwargs(timeout_seconds=1800.0))
        assert req.timeout_seconds == 1800.0

    def test_above_new_ceiling_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(timeout_seconds=1800.1))


class TestMaxRetriesBounds:
    """OMN-15115: max_retries is a new field, default 3 (matches the
    transport's historical hardcoded behavior)."""

    def test_default_is_3(self) -> None:
        req = ModelLlmInferenceRequest(**_chat_kwargs())
        assert req.max_retries == 3

    def test_explicit_override_honored(self) -> None:
        req = ModelLlmInferenceRequest(**_chat_kwargs(max_retries=1))
        assert req.max_retries == 1

    def test_zero_is_valid(self) -> None:
        req = ModelLlmInferenceRequest(**_chat_kwargs(max_retries=0))
        assert req.max_retries == 0

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(max_retries=-1))

    def test_above_ten_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelLlmInferenceRequest(**_chat_kwargs(max_retries=11))

    def test_ten_is_valid_boundary(self) -> None:
        req = ModelLlmInferenceRequest(**_chat_kwargs(max_retries=10))
        assert req.max_retries == 10
