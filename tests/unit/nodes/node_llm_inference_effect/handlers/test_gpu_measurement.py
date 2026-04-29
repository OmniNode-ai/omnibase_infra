# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""GPU usage evidence tests for node_llm_inference_effect (OMN-10338)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from omnibase_infra.enums import EnumLlmOperationType
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)

_CORRELATION_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


def _make_transport() -> MagicMock:
    transport = MagicMock(spec=MixinLlmHttpTransport)
    transport._execute_llm_http_call = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
    )
    transport._http_client = None
    transport._owns_http_client = True
    return transport


def _make_request(**overrides: Any) -> ModelLlmInferenceRequest:
    defaults: dict[str, Any] = {
        "base_url": "http://localhost:8000",
        "model": "qwen3-coder-30b-a3b",
        "operation_type": EnumLlmOperationType.CHAT_COMPLETION,
        "messages": ({"role": "user", "content": "Hello"},),
        "gpu_type": "rtx_5090",
        "gpu_count": 1,
    }
    defaults.update(overrides)
    return ModelLlmInferenceRequest(**defaults)


@pytest.mark.unit
class TestGpuMeasurement:
    @pytest.mark.asyncio
    async def test_gpu_seconds_and_config_are_added_to_metrics_extensions(
        self,
    ) -> None:
        """GPU seconds come from call wall-clock; GPU type/count from config."""
        handler = HandlerLlmOpenaiCompatible(_make_transport())
        with patch(
            "omnibase_infra.nodes.node_llm_inference_effect.handlers."
            "handler_llm_openai_compatible.time.perf_counter",
            side_effect=[10.0, 12.34567],
        ):
            await handler.handle(_make_request(), correlation_id=_CORRELATION_ID)

        metrics = handler.last_call_metrics
        assert metrics is not None
        assert metrics.extensions["gpu_seconds"] == 2.346
        assert metrics.extensions["gpu_type"] == "rtx_5090"
        assert metrics.extensions["gpu_count"] == 1
        assert metrics.extensions["compute_usage_source"] == "API"

    @pytest.mark.asyncio
    async def test_compute_usage_source_estimated_when_usage_is_estimated(
        self,
    ) -> None:
        """Without API usage, compute usage provenance degrades to ESTIMATED."""
        transport = _make_transport()
        transport._execute_llm_http_call.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "estimated"},
                    "finish_reason": "stop",
                }
            ]
        }
        handler = HandlerLlmOpenaiCompatible(transport)

        await handler.handle(_make_request(), correlation_id=_CORRELATION_ID)

        metrics = handler.last_call_metrics
        assert metrics is not None
        assert metrics.extensions["compute_usage_source"] == "ESTIMATED"

    def test_gpu_type_and_count_must_be_configured_together(self) -> None:
        with pytest.raises(ValueError, match="gpu_type and gpu_count"):
            _make_request(gpu_type="rtx_5090", gpu_count=None)
