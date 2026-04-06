# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Build loop domain plugin for kernel-level initialization.

PluginBuildLoop implements ProtocolDomainPlugin for the Build Loop
domain, encapsulating all build-loop-specific initialization that
wires the autonomous loop orchestrator into the runtime kernel.

The plugin handles:
    - Handler instantiation (HandlerLoopOrchestrator)
    - Dispatcher wiring into MessageDispatchEngine
    - Event consumer startup via EventBusSubcontractWiring

Design:
    Like PluginDelegation, this plugin has no PostgreSQL dependency.
    The build loop pipeline activates unconditionally.

Related:
    - OMN-7319: node_autonomous_loop_orchestrator
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from omnibase_infra.runtime.contract_topic_router import (
    build_topic_router_from_contract,
)
from omnibase_infra.runtime.models.model_handshake_result import (
    ModelHandshakeResult,
)
from omnibase_infra.runtime.protocol_domain_plugin import (
    ModelDomainPluginConfig,
    ModelDomainPluginResult,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.runtime.event_bus_subcontract_wiring import (
        EventBusSubcontractWiring,
    )

logger = logging.getLogger(__name__)

_CONTRACT_PATH = Path(__file__).parent / "contract.yaml"
try:
    _contract_raw = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))
except (OSError, yaml.YAMLError) as _contract_exc:
    logging.getLogger(__name__).warning(
        "Failed to load build loop contract at %s: %s. Using empty topic router.",
        _CONTRACT_PATH,
        _contract_exc,
    )
    _contract_raw = {}
_CONTRACT_DATA: dict[str, object] = (
    _contract_raw if isinstance(_contract_raw, dict) else {}
)
_TOPIC_ROUTER: dict[str, str] = build_topic_router_from_contract(_CONTRACT_DATA)


class PluginBuildLoop:
    """Build loop domain plugin for kernel initialization.

    Wires the autonomous loop orchestrator into the runtime kernel.
    Stateless -- no external resources beyond the event bus.
    """

    def __init__(self) -> None:
        self._wiring: EventBusSubcontractWiring | None = None
        self._handler_wiring_succeeded: bool = False
        self._dispatcher_wiring_succeeded: bool = False

    @property
    def plugin_id(self) -> str:
        return "build-loop"

    @property
    def display_name(self) -> str:
        return "Build Loop"

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        """Build loop plugin activates unconditionally."""
        logger.info(
            "[BUILD-LOOP] PluginBuildLoop.should_activate() called "
            "(node_identity=%s, correlation_id=%s)",
            config.node_identity,
            config.correlation_id,
        )
        return True

    async def initialize(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """No-op initialization -- build loop has no external resources."""
        return ModelDomainPluginResult(
            plugin_id=self.plugin_id,
            success=True,
            message="Build loop plugin initialized (no resources required)",
            resources_created=[],
            duration_seconds=0.0,
        )

    async def validate_handshake(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelHandshakeResult:
        """No handshake checks required for build loop."""
        return ModelHandshakeResult.default_pass(self.plugin_id)

    async def wire_handlers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Register build loop handlers with the container."""
        from omnibase_infra.nodes.node_autonomous_loop_orchestrator.wiring import (
            wire_build_loop_handlers,
        )

        start_time = time.time()
        try:
            result = await wire_build_loop_handlers(config.container)
            duration = time.time() - start_time

            logger.info(
                "Build loop handlers wired (correlation_id=%s)",
                config.correlation_id,
                extra={"services": result["services"]},
            )

            self._handler_wiring_succeeded = True

            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Build loop handlers wired",
                services_registered=result["services"],
                duration_seconds=duration,
            )

        except Exception as e:  # noqa: BLE001
            duration = time.time() - start_time
            logger.error(  # noqa: TRY400
                "Failed to wire build loop handlers: %s",
                sanitize_error_message(e),
                extra={"correlation_id": str(config.correlation_id)},
            )
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=sanitize_error_message(e),
                duration_seconds=duration,
            )

    async def wire_dispatchers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Wire build loop dispatchers into the MessageDispatchEngine."""
        start_time = time.time()

        if config.container.service_registry is None:
            logger.warning(
                "DEGRADED_MODE: ServiceRegistry not available, skipping "
                "build loop dispatcher wiring (correlation_id=%s)",
                config.correlation_id,
            )
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="ServiceRegistry not available",
            )

        if config.dispatch_engine is None:
            logger.warning(
                "DEGRADED_MODE: dispatch_engine not available, skipping "
                "build loop dispatcher wiring (correlation_id=%s)",
                config.correlation_id,
            )
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="dispatch_engine not available",
            )

        try:
            from omnibase_infra.nodes.node_autonomous_loop_orchestrator.wiring import (
                wire_build_loop_dispatchers,
            )

            dispatch_summary = await wire_build_loop_dispatchers(
                container=config.container,
                engine=config.dispatch_engine,
                correlation_id=config.correlation_id,
            )

            duration = time.time() - start_time
            logger.info(
                "Build loop dispatchers wired into engine (correlation_id=%s)",
                config.correlation_id,
                extra={
                    "dispatchers": dispatch_summary.get("dispatchers", []),
                    "routes": dispatch_summary.get("routes", []),
                },
            )

            self._dispatcher_wiring_succeeded = True
            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Build loop dispatchers wired into engine",
                resources_created=list(dispatch_summary.get("dispatchers", [])),
                duration_seconds=duration,
            )

        except Exception as e:  # noqa: BLE001
            duration = time.time() - start_time
            logger.error(  # noqa: TRY400
                "Failed to wire build loop dispatchers: %s",
                sanitize_error_message(e),
                extra={"correlation_id": str(config.correlation_id)},
            )
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=sanitize_error_message(e),
                duration_seconds=duration,
            )

    async def start_consumers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Start event consumers via EventBusSubcontractWiring."""
        logger.info(
            "[BUILD-LOOP] === PLUGIN start_consumers() ENTRY === "
            "(correlation_id=%s, node_identity=%s, "
            "handler_wired=%s, dispatcher_wired=%s, "
            "event_bus_type=%s, dispatch_engine=%s)",
            config.correlation_id,
            config.node_identity,
            self._handler_wiring_succeeded,
            self._dispatcher_wiring_succeeded,
            type(config.event_bus).__name__ if config.event_bus else "None",
            type(config.dispatch_engine).__name__ if config.dispatch_engine else "None",
        )
        start_time = time.time()
        correlation_id = config.correlation_id

        if not (self._handler_wiring_succeeded and self._dispatcher_wiring_succeeded):
            logger.warning(
                "Skipping consumer startup: handler/dispatcher wiring did not succeed "
                "for plugin '%s' (correlation_id=%s)",
                self.plugin_id,
                correlation_id,
            )
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="Handler/dispatcher wiring did not succeed -- consumers not started",
            )

        if config.dispatch_engine is None:
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="dispatch_engine not available",
            )

        from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
            ProtocolEventBusSubscriber,
        )

        if not isinstance(config.event_bus, ProtocolEventBusSubscriber):
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="Event bus does not support subscribe",
            )

        if config.node_identity is None:
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="node_identity not set (required for consumer subscription)",
            )

        wiring = None
        try:
            from omnibase_core.enums import EnumInjectionScope
            from omnibase_infra.runtime.event_bus_subcontract_wiring import (
                EventBusSubcontractWiring,
                load_event_bus_subcontract,
                load_published_events_map,
            )
            from omnibase_infra.runtime.service_dispatch_result_applier import (
                DispatchResultApplier,
            )

            contract_path = Path(__file__).parent / "contract.yaml"
            subcontract = load_event_bus_subcontract(contract_path, logger=logger)

            if subcontract is None:
                return ModelDomainPluginResult.skipped(
                    plugin_id=self.plugin_id,
                    reason=f"No event_bus subcontract in {contract_path}",
                )

            published_events_map = load_published_events_map(
                contract_path, logger=logger
            )

            logger.info(
                "Loaded published_events_map from %s: %d event-type->topic mappings",
                contract_path,
                len(published_events_map),
            )

            result_applier = DispatchResultApplier(
                event_bus=config.event_bus,  # type: ignore[arg-type]
                output_topic=config.output_topic,
                topic_router=_TOPIC_ROUTER,
                output_topic_map=published_events_map,
            )

            if config.container.service_registry is not None:
                await config.container.service_registry.register_instance(
                    interface=DispatchResultApplier,
                    instance=result_applier,
                    scope=EnumInjectionScope.GLOBAL,
                    metadata={
                        "description": "Dispatch result applier for build loop domain",
                        "plugin_id": self.plugin_id,
                    },
                )

            logger.info(
                "[BUILD-LOOP] Creating EventBusSubcontractWiring "
                "(env=%s, node_name=%s, service=%s, version=%s, "
                "subscribe_topics=%s)",
                config.node_identity.env,
                config.node_identity.node_name,
                config.node_identity.service,
                config.node_identity.version,
                subcontract.subscribe_topics,
            )

            wiring = EventBusSubcontractWiring(
                event_bus=config.event_bus,
                dispatch_engine=config.dispatch_engine,
                environment=config.node_identity.env,
                node_name=config.node_identity.node_name,
                service=config.node_identity.service,
                version=config.node_identity.version,
                result_applier=result_applier,
            )

            logger.info(
                "[BUILD-LOOP] Calling wire_subscriptions for "
                "node_autonomous_loop_orchestrator (correlation_id=%s)",
                correlation_id,
            )
            await wiring.wire_subscriptions(
                subcontract=subcontract,
                node_name="node_autonomous_loop_orchestrator",
            )

            self._wiring = wiring

            logger.info(
                "Build loop consumers started via EventBusSubcontractWiring "
                "(correlation_id=%s)",
                correlation_id,
                extra={
                    "subscribe_topics": subcontract.subscribe_topics,
                    "topic_count": len(subcontract.subscribe_topics)
                    if subcontract.subscribe_topics
                    else 0,
                },
            )

            duration = time.time() - start_time

            async def _cleanup_wiring() -> None:
                if self._wiring is not None:
                    await self._wiring.cleanup()
                    self._wiring = None

            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Build loop consumers started via EventBusSubcontractWiring",
                duration_seconds=duration,
                unsubscribe_callbacks=[_cleanup_wiring],
            )

        except Exception as e:  # noqa: BLE001
            duration = time.time() - start_time
            if wiring is not None:
                await wiring.cleanup()
            self._wiring = None
            logger.error(  # noqa: TRY400
                "Failed to start build loop consumers: %s",
                sanitize_error_message(e),
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=sanitize_error_message(e),
                duration_seconds=duration,
            )

    async def shutdown(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Clean up build loop resources."""
        if self._wiring is not None:
            await self._wiring.cleanup()
            self._wiring = None
        return ModelDomainPluginResult(
            plugin_id=self.plugin_id,
            success=True,
            message="Build loop plugin shut down",
            duration_seconds=0.0,
        )
