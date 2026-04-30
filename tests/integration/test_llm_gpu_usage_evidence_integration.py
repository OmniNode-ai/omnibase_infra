# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration checks for LLM GPU usage evidence wiring (OMN-10338)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from omnibase_infra.enums import EnumLlmOperationType
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.models.pricing.model_pricing_table import ModelPricingTable
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)
from omnibase_infra.nodes.node_llm_inference_effect.services.service_llm_metrics_publisher import (
    ServiceLlmMetricsPublisher,
)

pytestmark = [pytest.mark.integration]

_CORRELATION_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRICING_MANIFEST = (
    _REPO_ROOT / "src" / "omnibase_infra" / "configs" / "pricing_manifest.yaml"
)


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


@pytest.mark.asyncio
async def test_gpu_usage_evidence_reaches_both_metric_events() -> None:
    """Measured GPU evidence is preserved from handler metrics into both events."""
    publisher = AsyncMock(return_value=True)
    handler = HandlerLlmOpenaiCompatible(_make_transport())
    service = ServiceLlmMetricsPublisher(handler=handler, publisher=publisher)

    with patch(
        "omnibase_infra.nodes.node_llm_inference_effect.handlers."
        "handler_llm_openai_compatible.time.perf_counter",
        side_effect=[10.0, 12.34567],
    ):
        await service.handle(_make_request(), correlation_id=_CORRELATION_ID)
    await asyncio.sleep(0)

    assert publisher.await_count == 2
    omni_payload = publisher.call_args_list[0].args[1]
    infra_payload = publisher.call_args_list[1].args[1]
    for payload in (omni_payload, infra_payload):
        assert payload["gpu_seconds"] == 2.346
        assert payload["gpu_type"] == "rtx_5090"
        assert payload["gpu_count"] == 1
        assert payload["compute_usage_source"] == "API"


def test_pricing_manifest_rates_gpu_usage_evidence() -> None:
    """Bundled pricing manifest can rate the GPU usage evidence emitted by runtime."""
    pricing = ModelPricingTable.from_yaml(_PRICING_MANIFEST)

    assert (
        pricing.estimate_compute_cost(
            gpu_type="rtx_5090",
            gpu_seconds=2.346,
            gpu_count=1,
        )
        is not None
    )
