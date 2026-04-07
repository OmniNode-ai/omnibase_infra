# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lifecycle hooks for the LLM domain (OMN-7662).

Extracted from PluginLlm to enable contract-driven auto-wiring.
These hooks are referenced from node_llm_inference_effect/contract.yaml
and invoked by the LifecycleHookExecutor during kernel startup/shutdown.

on_start:
    Creates AdapterModelRouter, starts ServiceLlmEndpointHealth probe loop,
    and starts the LLM inference Kafka consumer.

on_shutdown:
    Stops health probes, cancels inference consumer task, clears state.

Related:
    - OMN-6600: LLM domain plugin
    - OMN-7104: LLM inference Kafka consumer
    - OMN-7662: Extract lightweight infra plugins
"""

from __future__ import annotations

import asyncio
import logging
import os

from omnibase_infra.runtime.auto_wiring.context import ModelAutoWiringContext
from omnibase_infra.runtime.auto_wiring.models import ModelLifecycleHookResult

logger = logging.getLogger(__name__)

# Environment variable names checked for LLM endpoint activation
_LLM_URL_ENV_VARS: tuple[str, ...] = (
    "LLM_CODER_URL",
    "LLM_CODER_FAST_URL",
    "LLM_EMBEDDING_URL",
    "LLM_DEEPSEEK_R1_URL",
    "LLM_SMALL_URL",
)

# Module-level state shared between on_start and on_shutdown.
# Using module state mirrors PluginLlm's instance state but is
# compatible with the stateless lifecycle hook callable contract.
_state: dict[str, object] = {}


async def on_start(context: ModelAutoWiringContext) -> ModelLifecycleHookResult:
    """Start LLM adapter, health probes, and inference consumer.

    Checks LLM_*_URL env vars. If none are set, returns success with no
    background workers (graceful skip). Otherwise creates health probe
    loop and inference consumer task.
    """
    endpoints: dict[str, str] = {}
    for var in _LLM_URL_ENV_VARS:
        url = os.environ.get(var)  # ONEX_FLAG_EXEMPT: activation gate
        if url:
            endpoints[var] = url

    if not endpoints:
        logger.debug(
            "lifecycle_llm.on_start: no LLM_*_URL env vars set, skipping "
            "(handler_id=%s)",
            context.handler_id,
        )
        return ModelLifecycleHookResult.succeeded(
            hook_name="on_start",
            background_workers=[],
        )

    background_workers: list[str] = []

    # Create AdapterModelRouter
    from omnibase_infra.adapters.llm.adapter_model_router import AdapterModelRouter

    router = AdapterModelRouter()
    _state["router"] = router
    _state["endpoints"] = endpoints

    logger.info(
        "lifecycle_llm.on_start: initialized AdapterModelRouter with %d endpoint(s) "
        "(handler_id=%s)",
        len(endpoints),
        context.handler_id,
    )

    # Start health probe loop
    from omnibase_infra.services.service_llm_endpoint_health import (
        ModelLlmEndpointHealthConfig,
        ServiceLlmEndpointHealth,
    )

    friendly_endpoints: dict[str, str] = {}
    for var_name, url in endpoints.items():
        friendly = var_name.removeprefix("LLM_").removesuffix("_URL").lower()
        friendly_endpoints[friendly] = url

    health_config = ModelLlmEndpointHealthConfig(endpoints=friendly_endpoints)
    event_bus = context.services.get("event_bus")
    health_service = ServiceLlmEndpointHealth(
        config=health_config,
        event_bus=event_bus,
    )
    await health_service.start()
    _state["health_service"] = health_service
    background_workers.append("llm-health-probe")

    logger.info(
        "lifecycle_llm.on_start: started health probe loop for %d endpoints "
        "(handler_id=%s)",
        len(friendly_endpoints),
        context.handler_id,
    )

    # Start inference consumer if event bus is available
    if event_bus is not None:
        from omnibase_infra.adapters.llm.consumer_llm_inference import (
            start_llm_inference_consumer,
        )

        inference_task = asyncio.create_task(
            start_llm_inference_consumer(
                event_bus=event_bus,
                endpoints=endpoints,
                correlation_id=context.metadata.get("correlation_id", "unknown"),
            ),
            name="llm-inference-consumer",
        )
        _state["inference_task"] = inference_task
        background_workers.append("llm-inference-consumer")

        logger.info(
            "lifecycle_llm.on_start: started LLM inference consumer (handler_id=%s)",
            context.handler_id,
        )

    return ModelLifecycleHookResult.succeeded(
        hook_name="on_start",
        background_workers=background_workers,
    )


async def on_shutdown(context: ModelAutoWiringContext) -> ModelLifecycleHookResult:
    """Stop health probes and inference consumer."""
    health_service = _state.pop("health_service", None)
    if health_service is not None and hasattr(health_service, "stop"):
        await health_service.stop()

    inference_task = _state.pop("inference_task", None)
    if inference_task is not None and isinstance(inference_task, asyncio.Task):
        if not inference_task.done():
            inference_task.cancel()
            try:
                await inference_task
            except asyncio.CancelledError:
                pass

    _state.pop("router", None)
    _state.pop("endpoints", None)

    logger.info(
        "lifecycle_llm.on_shutdown: complete (handler_id=%s)",
        context.handler_id,
    )

    return ModelLifecycleHookResult.succeeded(hook_name="on_shutdown")
