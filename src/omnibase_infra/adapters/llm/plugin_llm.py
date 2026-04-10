# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""LLM domain plugin for kernel-level initialization.

Wires the AdapterModelRouter (multi-provider LLM routing) and
ServiceLlmEndpointHealth (async health probe loop) into the kernel
lifecycle via the ProtocolDomainPlugin protocol.

Activation:
    The plugin activates when at least one ``LLM_*_URL`` environment variable
    is set (e.g. ``LLM_CODER_URL``, ``LLM_EMBEDDING_URL``).

Lifecycle:
    1. should_activate() — checks for any LLM_*_URL env var
    2. initialize() — creates AdapterModelRouter with routing-decided callback
    3. wire_handlers() — registers router in container for handler injection
    4. wire_dispatchers() — no-op (no dispatch routes)
    5. start_consumers() — starts ServiceLlmEndpointHealth probe loop
    6. shutdown() — stops health probes, clears state

Related:
    - OMN-6600: Create LLM domain plugin for service_kernel
    - OMN-2319: SPI LLM protocol adapters
    - OMN-8023: Wire routing-decided callback so routing decisions table populates
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.adapters.llm.adapter_model_router import AdapterModelRouter
from omnibase_infra.services.service_llm_endpoint_health import (
    ModelLlmEndpointHealthConfig,
    ServiceLlmEndpointHealth,
)
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
    from omnibase_infra.runtime.models import (
        ModelDomainPluginConfig,
        ModelDomainPluginResult,
    )

logger = logging.getLogger(__name__)


def _make_routing_decided_callback(
    event_bus: ProtocolEventBusLike,
) -> Callable[[dict[str, object]], Awaitable[None]]:
    """Return an async callback that emits routing-decided events to Kafka.

    The callback is bound to the provided event_bus and the resolved
    ROUTING_DECIDED topic.  Failures are logged at warning level and
    dropped — routing events are best-effort observability.
    """
    topic_registry = ServiceTopicRegistry.from_defaults()
    routing_topic = topic_registry.resolve(topic_keys.ROUTING_DECIDED)

    async def _on_routing_decided(event: dict[str, object]) -> None:
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=event,
            correlation_id=str(event.get("correlation_id") or ""),
            event_type="routing-decided",
            source_tool="AdapterModelRouter",
        )
        try:
            await event_bus.publish_envelope(envelope=envelope, topic=routing_topic)
        except Exception:  # noqa: BLE001 — best-effort; must not crash the router
            logger.warning(
                "PluginLlm: failed to publish routing-decided event to %s",
                routing_topic,
                exc_info=True,
            )

    return _on_routing_decided


# Environment variable prefixes checked for activation
_LLM_URL_ENV_VARS: tuple[str, ...] = (
    "LLM_CODER_URL",
    "LLM_CODER_FAST_URL",
    "LLM_EMBEDDING_URL",
    "LLM_DEEPSEEK_R1_URL",
    "LLM_SMALL_URL",
    "LLM_GLM_URL",
    "LLM_OPENROUTER_URL",
)


class PluginLlm:
    """LLM domain plugin — wires AdapterModelRouter + health probes.

    Follows the ProtocolDomainPlugin lifecycle contract. The plugin creates
    an AdapterModelRouter during initialization, registers it in the kernel
    container, and optionally starts a health probe loop for configured
    LLM endpoints.
    """

    def __init__(self) -> None:
        self._router: AdapterModelRouter | None = None
        self._health_service: ServiceLlmEndpointHealth | None = None
        self._health_task: asyncio.Task[None] | None = None
        self._inference_consumer_task: asyncio.Task[None] | None = None
        self._endpoints: dict[str, str] = {}

    @property
    def plugin_id(self) -> str:
        """Return unique identifier for this plugin."""
        return "llm"

    @property
    def display_name(self) -> str:
        """Return human-readable name for this plugin."""
        return "LLM"

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        """Activate when any LLM_*_URL env var is set."""
        for var in _LLM_URL_ENV_VARS:
            url = os.environ.get(var)  # ONEX_FLAG_EXEMPT: activation gate
            if url:
                self._endpoints[var] = url
        activated = bool(self._endpoints)
        if activated:
            logger.info(
                "PluginLlm: activating with %d endpoints (correlation_id=%s)",
                len(self._endpoints),
                config.correlation_id,
            )
        else:
            logger.debug(
                "PluginLlm: no LLM_*_URL env vars set, skipping (correlation_id=%s)",
                config.correlation_id,
            )
        return activated

    async def initialize(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Create AdapterModelRouter with configured endpoints and routing callback."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        event_bus = getattr(config, "event_bus", None)
        on_routing_decided = None
        if event_bus is not None:
            on_routing_decided = _make_routing_decided_callback(event_bus)
            logger.info(
                "PluginLlm: routing-decided callback wired to event_bus (correlation_id=%s)",
                config.correlation_id,
            )
        else:
            logger.debug(
                "PluginLlm: no event_bus available, routing-decided callback skipped "
                "(correlation_id=%s)",
                config.correlation_id,
            )

        self._router = AdapterModelRouter(on_routing_decided=on_routing_decided)

        logger.info(
            "PluginLlm: initialized AdapterModelRouter with %d endpoint(s) "
            "(correlation_id=%s)",
            len(self._endpoints),
            config.correlation_id,
        )

        return ModelDomainPluginResult(
            plugin_id=self.plugin_id,
            success=True,
            message=f"LLM router initialized with {len(self._endpoints)} endpoints",
            resources_created=["adapter_model_router"],
        )

    async def wire_handlers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Register LLM adapter and DelegationIntentBridge in the container."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        services: list[str] = ["AdapterModelRouter"]

        event_bus = getattr(config, "event_bus", None)
        if event_bus is not None and config.container is not None:
            from omnibase_core.enums import EnumInjectionScope
            from omnibase_infra.adapters.llm.adapter_llm_caller_delegation import (
                LlmCallerDelegation,
            )
            from omnibase_infra.nodes.node_delegation_orchestrator.delegation_intent_bridge import (
                DelegationIntentBridge,
            )

            bridge = DelegationIntentBridge(
                event_bus=event_bus,
                llm_caller=LlmCallerDelegation(),
            )
            if config.container.service_registry is not None:
                await config.container.service_registry.register_instance(
                    interface=DelegationIntentBridge,
                    instance=bridge,
                    scope=EnumInjectionScope.GLOBAL,
                    metadata={
                        "description": "Delegation intent bridge with local-model LLM caller",
                    },
                )
            services.append("DelegationIntentBridge")
            logger.info(
                "PluginLlm: DelegationIntentBridge registered with LlmCallerDelegation "
                "(correlation_id=%s)",
                config.correlation_id,
            )
        else:
            logger.debug(
                "PluginLlm: skipping DelegationIntentBridge registration — "
                "no event_bus or container (correlation_id=%s)",
                config.correlation_id,
            )

        return ModelDomainPluginResult(
            plugin_id=self.plugin_id,
            success=True,
            message="LLM handlers wired",
            services_registered=services,
        )

    async def wire_dispatchers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """No-op — LLM plugin has no dispatch routes."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message="LLM plugin has no dispatchers",
        )

    async def start_consumers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Start health probe loop and LLM inference command consumer."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        # --- Health probe loop (existing) ---
        friendly_endpoints: dict[str, str] = {}
        for var_name, url in self._endpoints.items():
            friendly = var_name.removeprefix("LLM_").removesuffix("_URL").lower()
            friendly_endpoints[friendly] = url

        health_config = ModelLlmEndpointHealthConfig(
            endpoints=friendly_endpoints,
        )
        event_bus = getattr(config, "event_bus", None)
        self._health_service = ServiceLlmEndpointHealth(
            config=health_config,
            event_bus=event_bus,
        )
        await self._health_service.start()

        logger.info(
            "PluginLlm: started health probe loop for %d endpoints (correlation_id=%s)",
            len(friendly_endpoints),
            config.correlation_id,
        )

        # --- LLM inference command consumer (new — OMN-7104) ---
        # Subscribe to the LLM inference request topic declared in
        # node_llm_inference_effect/contract.yaml so the node can receive
        # Kafka commands and route them to HandlerLlmOpenaiCompatible.
        if event_bus is not None:
            from omnibase_infra.adapters.llm.consumer_llm_inference import (
                start_llm_inference_consumer,
            )

            self._inference_consumer_task = asyncio.create_task(
                start_llm_inference_consumer(
                    event_bus=event_bus,
                    endpoints=self._endpoints,
                    correlation_id=str(config.correlation_id),
                ),
                name="llm-inference-consumer",
            )
            logger.info(
                "PluginLlm: started LLM inference command consumer (correlation_id=%s)",
                config.correlation_id,
            )
        else:
            logger.debug(
                "PluginLlm: no event_bus available, skipping inference consumer (correlation_id=%s)",
                config.correlation_id,
            )

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message=f"Health probes + inference consumer started for {len(friendly_endpoints)} endpoints",
        )

    async def shutdown(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Stop health probes, close connections."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        if self._health_service is not None:
            await self._health_service.stop()
            self._health_service = None

        self._router = None
        self._endpoints.clear()

        logger.info(
            "PluginLlm: shutdown complete (correlation_id=%s)",
            config.correlation_id,
        )

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message="LLM plugin shutdown complete",
        )


__all__ = [
    "PluginLlm",
]
