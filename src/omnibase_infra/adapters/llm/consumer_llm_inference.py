# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Kafka consumer for LLM inference commands.

Subscribes to ``onex.cmd.omnibase-infra.llm-inference-request.v1``,
deserializes to ``ModelLlmInferenceRequest``, routes to
``HandlerLlmOpenaiCompatible``, and emits the response to
``onex.evt.omnibase-infra.llm-call-completed.v1``.

This is the missing link between the LLM node contract (which declares
Kafka subscribe/publish topics) and the actual runtime consumer. Without
this module, the LLM node exists as a contract + handler but has no Kafka
ingestion path — commands silently go unconsumed.

Related:
    - OMN-7104: Spike — verify LLM node handles Kafka command e2e
    - OMN-7103: Node-Based LLM Delegation Workflow
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

from omnibase_infra.topics.platform_topic_suffixes import (
    SUFFIX_INTELLIGENCE_LLM_CALL_COMPLETED,
    SUFFIX_LLM_INFERENCE_REQUEST,
)

logger = logging.getLogger(__name__)

# Topic names via canonical constants (not hardcoded strings)
_SUBSCRIBE_TOPIC = SUFFIX_LLM_INFERENCE_REQUEST
_PUBLISH_TOPIC_INTELLIGENCE = SUFFIX_INTELLIGENCE_LLM_CALL_COMPLETED

# Consumer group ID
_CONSUMER_GROUP = "omnibase-infra.llm-inference-consumer"


async def start_llm_inference_consumer(
    *,
    event_bus: EventBusKafka,
    endpoints: dict[str, str],
    correlation_id: str,
) -> None:
    """Subscribe to LLM inference requests and route to the handler.

    This function runs as a long-lived asyncio task started by PluginLlm.
    It subscribes to the LLM inference command topic, deserializes each
    message into a ModelLlmInferenceRequest, calls the appropriate handler,
    and emits the response.

    Args:
        event_bus: The EventBusKafka instance from the runtime.
        endpoints: Dict of LLM_*_URL env var names → URLs.
        correlation_id: Runtime correlation ID for tracing.
    """
    from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
        TransportHolderLlmHttp,
    )
    from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
        HandlerLlmOpenaiCompatible,
    )
    from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
        ModelLlmInferenceRequest,
    )

    # Create transport holder (provides _execute_llm_http_call) and handler
    transport = TransportHolderLlmHttp(
        target_name="llm-inference-consumer",
        max_timeout_seconds=120.0,
    )
    handler = HandlerLlmOpenaiCompatible(transport)

    # Select default endpoint URL (prefer coder_fast > coder > deepseek)
    default_url = (
        endpoints.get("LLM_CODER_FAST_URL")
        or endpoints.get("LLM_CODER_URL")
        or endpoints.get("LLM_DEEPSEEK_R1_URL")
        or ""
    )

    if not default_url:
        logger.warning(
            "LLM inference consumer: no LLM endpoints configured, consumer will not start "
            "(correlation_id=%s)",
            correlation_id,
        )
        return

    logger.info(
        "LLM inference consumer: subscribing to %s (default_url=%s, correlation_id=%s)",
        _SUBSCRIBE_TOPIC,
        default_url[:40],
        correlation_id,
    )

    async def _handle_message(message: object) -> None:
        """Process a single LLM inference request message."""
        msg_correlation_id = str(uuid4())
        start_time = time.monotonic()

        try:
            # Parse the message payload
            if hasattr(message, "value") and message.value:
                payload = json.loads(message.value)
            elif isinstance(message, dict):
                payload = message
            else:
                logger.debug("LLM inference consumer: unrecognized message format")
                return

            msg_correlation_id = payload.get("correlation_id", msg_correlation_id)

            # Determine endpoint URL — use payload's base_url if provided,
            # otherwise use the model name to look up the right endpoint
            model_name = payload.get("model", "")
            base_url = payload.get("base_url") or payload.get(
                "provider_config", {}
            ).get("base_url", "")
            api_key = payload.get("api_key") or payload.get("provider_config", {}).get(
                "api_key", ""
            )

            if not base_url:
                # Route by model name heuristics
                model_lower = model_name.lower()
                if "glm" in model_lower:
                    base_url = os.environ.get("LLM_GLM_URL", "")
                    api_key = api_key or os.environ.get("LLM_GLM_API_KEY", "")
                elif "deepseek" in model_lower:
                    base_url = endpoints.get("LLM_DEEPSEEK_R1_URL", "")
                elif "coder" in model_lower or "qwen" in model_lower:
                    base_url = endpoints.get("LLM_CODER_URL", "") or endpoints.get(
                        "LLM_CODER_FAST_URL", ""
                    )
                else:
                    base_url = default_url

            if not base_url:
                logger.warning(
                    "LLM inference consumer: no endpoint for model=%s (correlation_id=%s)",
                    model_name,
                    msg_correlation_id,
                )
                return

            # Build the request model
            messages = payload.get("messages", [])
            request = ModelLlmInferenceRequest(
                base_url=base_url.rstrip("/"),
                model=model_name or "auto",
                operation_type="chat_completion",
                messages=messages,
                max_tokens=payload.get("max_tokens", 2048),
                temperature=payload.get("temperature", 0.3),
                api_key=api_key if api_key else None,
            )

            # Execute inference
            response = await handler.handle(request)

            latency_ms = int((time.monotonic() - start_time) * 1000)

            # Emit response event
            response_payload = {
                "correlation_id": msg_correlation_id,
                "model_used": getattr(response, "model_used", model_name),
                "generated_text": getattr(response, "generated_text", ""),
                "latency_ms": latency_ms,
                "usage": {
                    "prompt_tokens": getattr(
                        getattr(response, "usage", None), "prompt_tokens", 0
                    ),
                    "completion_tokens": getattr(
                        getattr(response, "usage", None), "completion_tokens", 0
                    ),
                    "total_tokens": getattr(
                        getattr(response, "usage", None), "total_tokens", 0
                    ),
                },
                "finish_reason": getattr(response, "finish_reason", "unknown"),
            }

            # Publish to intelligence LLM call completed topic
            if hasattr(event_bus, "publish"):
                await event_bus.publish(
                    _PUBLISH_TOPIC_INTELLIGENCE,
                    key=msg_correlation_id,
                    value=json.dumps(response_payload).encode(),
                )

            logger.info(
                "LLM inference consumer: completed request model=%s latency=%dms (correlation_id=%s)",
                model_name,
                latency_ms,
                msg_correlation_id,
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            logger.warning(
                "LLM inference consumer: failed error=%s latency=%dms (correlation_id=%s)",
                type(exc).__name__,
                latency_ms,
                msg_correlation_id,
            )

    # Subscribe to the command topic
    try:
        if hasattr(event_bus, "subscribe"):
            await event_bus.subscribe(
                topic=_SUBSCRIBE_TOPIC,
                on_message=_handle_message,
                group_id=_CONSUMER_GROUP,
            )
            logger.info(
                "LLM inference consumer: subscribed to %s (group=%s, correlation_id=%s)",
                _SUBSCRIBE_TOPIC,
                _CONSUMER_GROUP,
                correlation_id,
            )
        else:
            logger.warning(
                "LLM inference consumer: event_bus has no subscribe() method (correlation_id=%s)",
                correlation_id,
            )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning(
            "LLM inference consumer: failed to subscribe to %s: %s (correlation_id=%s)",
            _SUBSCRIBE_TOPIC,
            exc,
            correlation_id,
        )
