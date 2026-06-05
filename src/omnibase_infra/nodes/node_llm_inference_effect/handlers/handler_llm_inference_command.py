# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command handler for contract-wired LLM inference requests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

from omnibase_core.enums.cost import EnumUsageSource
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.models.llm import ModelLlmInferenceResponse
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_call_completed_event import (
    ModelLlmCallCompletedEvent,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_call_completed_infra_event import (
    ModelLlmCallCompletedInfraEvent,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_command import (
    ModelLlmInferenceCommand,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)


class LlmInferenceCommandTransport(MixinLlmHttpTransport):
    """HTTP transport owned by the effect-node command handler."""

    def __init__(self) -> None:
        self._init_llm_http_transport(
            target_name="node-llm-inference-effect",
            max_timeout_seconds=120.0,
        )


class HandlerLlmInferenceCommand:
    """Execute typed LLM inference commands received through runtime wiring."""

    def __init__(
        self,
        inference_handler: HandlerLlmOpenaiCompatible | None = None,
    ) -> None:
        self._inference_handler = inference_handler or HandlerLlmOpenaiCompatible(
            LlmInferenceCommandTransport()
        )

    async def handle(
        self,
        command: ModelLlmInferenceCommand,
    ) -> ModelHandlerOutput[None]:
        """Run an inference command and return contract-declared output events."""
        request = self._build_request(command)
        response = await self._inference_handler.handle(
            request,
            correlation_id=command.correlation_id,
        )
        events = self._build_output_events(
            command=command,
            request=request,
            response=response,
        )
        return ModelHandlerOutput.for_effect(
            input_envelope_id=uuid4(),
            correlation_id=command.correlation_id,
            handler_id=type(self).__name__,
            events=events,
            processing_time_ms=response.latency_ms,
        )

    def _build_request(
        self,
        command: ModelLlmInferenceCommand,
    ) -> ModelLlmInferenceRequest:
        endpoint_url = _optional_str(
            command.endpoint_url or command.provider_value("endpoint_url")
        )
        base_url = _optional_str(
            command.base_url or command.provider_value("base_url") or endpoint_url
        )
        if base_url is None:
            raise ValueError(
                "LLM inference command requires endpoint_url or base_url; "
                f"model={command.model}"
            )

        api_key = _optional_str(command.api_key or command.provider_value("api_key"))
        compute_usage_source = None
        if command.compute_usage_source is not None:
            compute_usage_source = EnumUsageSource(command.compute_usage_source)
        return ModelLlmInferenceRequest(
            endpoint_url=endpoint_url.rstrip("/") if endpoint_url is not None else None,
            base_url=base_url.rstrip("/"),
            operation_type=command.operation_type,
            model=command.model,
            messages=command.messages,
            prompt=command.prompt,
            max_tokens=command.max_tokens,
            temperature=command.temperature,
            top_p=command.top_p,
            stop=command.stop,
            api_key=api_key,
            extra_headers=command.extra_headers,
            timeout_seconds=command.timeout_seconds,
            gpu_type=command.gpu_type,
            gpu_count=command.gpu_count,
            compute_usage_source=compute_usage_source,
        )

    def _build_output_events(
        self,
        *,
        command: ModelLlmInferenceCommand,
        request: ModelLlmInferenceRequest,
        response: ModelLlmInferenceResponse,
    ) -> tuple[
        ModelLlmInferenceResponse,
        ModelLlmCallCompletedEvent,
        ModelLlmCallCompletedInfraEvent,
    ]:
        metrics = self._inference_handler.last_call_metrics
        if metrics is None:
            raise RuntimeError(
                "HandlerLlmOpenaiCompatible completed without last_call_metrics"
            )

        metrics_payload = json.loads(metrics.model_dump_json())
        extensions = metrics_payload.get("extensions")
        flattened_extensions = _flatten_metric_extensions(extensions)
        full_event = ModelLlmCallCompletedEvent(
            **metrics_payload,
            **flattened_extensions,
        )
        infra_event = ModelLlmCallCompletedInfraEvent(
            model_id=metrics.model_id,
            endpoint_url=request.endpoint_url or request.base_url,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            total_tokens=metrics.total_tokens,
            latency_ms=metrics.latency_ms,
            success=True,
            timestamp=datetime.now(UTC).isoformat(),
            **flattened_extensions,
        )
        return response, full_event, infra_event


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _flatten_metric_extensions(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value[key]
        for key in (
            "gpu_seconds",
            "gpu_type",
            "gpu_count",
            "compute_usage_source",
        )
        if key in value
    }


__all__ = ["HandlerLlmInferenceCommand"]
