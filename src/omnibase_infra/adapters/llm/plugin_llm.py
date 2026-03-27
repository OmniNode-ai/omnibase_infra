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
    2. initialize() — creates AdapterModelRouter, registers configured providers
    3. wire_handlers() — registers router in container for handler injection
    4. wire_dispatchers() — no-op (no dispatch routes)
    5. start_consumers() — starts ServiceLlmEndpointHealth probe loop
    6. shutdown() — stops health probes, clears state

Related:
    - OMN-6600: Create LLM domain plugin for service_kernel
    - OMN-2319: SPI LLM protocol adapters
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from omnibase_infra.adapters.llm.adapter_model_router import AdapterModelRouter
from omnibase_infra.services.service_llm_endpoint_health import (
    ModelLlmEndpointHealthConfig,
    ServiceLlmEndpointHealth,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.models import (
        ModelDomainPluginConfig,
        ModelDomainPluginResult,
    )

logger = logging.getLogger(__name__)

# Environment variable prefixes checked for activation
_LLM_URL_ENV_VARS: tuple[str, ...] = (
    "LLM_CODER_URL",
    "LLM_CODER_FAST_URL",
    "LLM_EMBEDDING_URL",
    "LLM_DEEPSEEK_R1_URL",
    "LLM_SMALL_URL",
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
        """Create AdapterModelRouter with configured endpoints."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        self._router = AdapterModelRouter()

        logger.info(
            "PluginLlm: initialized AdapterModelRouter with %d endpoint(s) "
            "(correlation_id=%s)",
            len(self._endpoints),
            config.correlation_id,
        )

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message=f"LLM router initialized with {len(self._endpoints)} endpoints",
            resources_created=["adapter_model_router"],
        )

    async def wire_handlers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Register LLM adapter in container for handler injection."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        if self._router is not None and config.container is not None:
            config.container.register_service(
                "AdapterModelRouter",
                self._router,
            )
            logger.info(
                "PluginLlm: registered AdapterModelRouter in container "
                "(correlation_id=%s)",
                config.correlation_id,
            )

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message="LLM handlers wired",
            services_registered=["AdapterModelRouter"],
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
        """Start ServiceLlmEndpointHealth probe loop as asyncio task."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        # Build friendly endpoint names from env var names
        friendly_endpoints: dict[str, str] = {}
        for var_name, url in self._endpoints.items():
            # LLM_CODER_URL -> coder
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

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message=f"Health probes started for {len(friendly_endpoints)} endpoints",
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
