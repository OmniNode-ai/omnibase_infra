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
    - OMN-16900: partition auth-dead endpoints out of the health probe set
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
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

# Same repo-relative resolution the routing API uses for this contract.
_MODEL_REGISTRY_PATH = (
    Path(__file__).parents[4] / "docker" / "catalog" / "model_registry.yaml"
)


def _auth_env_by_url_env(registry_path: Path) -> dict[str, str]:
    """Map ``base_url_env`` -> ``api_key_env`` from the model registry.

    Read-only: the registry is the declaration of which endpoints are
    auth-gated, so the plugin derives that fact rather than hardcoding a
    second copy of it. Entries without an ``api_key_env`` are omitted.

    Args:
        registry_path: Path to the model registry contract YAML.

    Returns:
        Mapping of URL env-var name to the auth env-var name it requires.
        Empty when the registry is not present (pip-installed layouts ship
        the library without the operational ``docker/`` tree), which reduces
        to the pre-OMN-16900 behaviour of probing everything.
    """
    if not registry_path.exists():
        logger.warning(
            "Model registry not found at %s; LLM health probes cannot "
            "classify auth-gated endpoints (OMN-16900)",
            registry_path,
        )
        return {}

    import yaml  # guarded: pyyaml is a declared dep; import here avoids cost

    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        logger.warning(
            "Model registry at %s has no 'models' list; LLM health probes "
            "cannot classify auth-gated endpoints (OMN-16900)",
            registry_path,
        )
        return {}

    auth_env: dict[str, str] = {}
    for entry in raw["models"]:
        if not isinstance(entry, dict) or entry.get("transport") != "http":
            continue
        url_env = entry.get("base_url_env", "")
        api_key_env = entry.get("api_key_env", "")
        if url_env and api_key_env:
            auth_env[str(url_env)] = str(api_key_env)
    return auth_env


def _partition_endpoints_by_auth(
    endpoints: dict[str, str],
    auth_env_by_url_env: dict[str, str],
    resolved_config: dict[str, str] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Split ``LLM_*_URL`` endpoints into probeable and auth-dead (OMN-16900).

    Auth secrets are read from the kernel's **resolved overlay config**, the
    same seam ``PluginDlq`` uses — this plugin does not read the process
    environment for secrets.  When no overlay is loaded (legacy env-var boot)
    there is no authoritative view of which secrets resolve, so nothing is
    classified here and every endpoint stays probeable; a rejected credential
    is then caught one layer down by the service's terminal-``AUTH_FAILED``
    backoff instead of before the first probe.

    Args:
        endpoints: Mapping of URL env-var name (e.g. ``LLM_GLM_URL``) to URL.
        auth_env_by_url_env: Registry-derived URL-env -> auth-env mapping.
        resolved_config: The kernel's resolved overlay config, or ``None`` in
            legacy env-var mode.

    Returns:
        ``(probeable, unauthenticated)``, both keyed by the friendly endpoint
        name (``LLM_GLM_URL`` -> ``glm``).
    """
    probeable: dict[str, str] = {}
    unauthenticated: dict[str, str] = {}
    for var_name, url in endpoints.items():
        friendly = var_name.removeprefix("LLM_").removesuffix("_URL").lower()
        api_key_env = auth_env_by_url_env.get(var_name, "")
        if resolved_config is not None and api_key_env:
            if resolved_config.get(api_key_env):
                probeable[friendly] = url
            else:
                unauthenticated[friendly] = url
            continue
        probeable[friendly] = url
    return probeable, unauthenticated


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
        """Register the LLM model router in the container.

        The delegation chain runs as a pure Kafka chain (OMN-12294): the LLM
        call effect consumes inference intents directly off the bus, so there is
        no in-process bridge or LLM caller to register here.
        """
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        return ModelDomainPluginResult(
            plugin_id=self.plugin_id,
            success=True,
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
        """Start health probe loop and LLM inference command consumer."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        # --- Health probe loop ---
        # OMN-16900: an endpoint whose registry-declared auth secret is absent
        # can never answer a probe, so it is classified once and never probed
        # rather than 401-ing every 30s in every container, forever.
        friendly_endpoints, unauthenticated_endpoints = _partition_endpoints_by_auth(
            endpoints=self._endpoints,
            auth_env_by_url_env=_auth_env_by_url_env(_MODEL_REGISTRY_PATH),
            resolved_config=config.overlay_config,
        )

        health_config = ModelLlmEndpointHealthConfig(
            endpoints=friendly_endpoints,
            unauthenticated_endpoints=unauthenticated_endpoints,
        )
        event_bus = getattr(config, "event_bus", None)
        self._health_service = ServiceLlmEndpointHealth(
            config=health_config,
            event_bus=event_bus,
        )
        await self._health_service.start()

        logger.info(
            "PluginLlm: started health probe loop for %d endpoints "
            "(%d skipped, no resolvable auth secret) (correlation_id=%s)",
            len(friendly_endpoints),
            len(unauthenticated_endpoints),
            config.correlation_id,
        )

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message=(
                f"Health probes started for {len(friendly_endpoints)} endpoints "
                f"({len(unauthenticated_endpoints)} skipped, no auth)"
            ),
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
