# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Emit daemon domain plugin for kernel-level initialization.

PluginEmitDaemon implements ProtocolDomainPlugin for the emit daemon
domain. Starts the omnimarket node_emit_daemon as a background async
task on kernel boot, providing event publishing to Kafka for all
connected clients (omniclaude hooks, IDE plugins, CLI tools).

Activation:
    Gated by ONEX_EMIT_DAEMON_ENABLED=true (default: false).
    The daemon binds a Unix socket; duplicate bind attempts result in
    FAILED (non-fatal) so only one daemon per host is active.

Design:
    - No Kafka consumer subscription (the daemon is a publisher, not consumer)
    - No handler/dispatcher wiring (standalone service, not a compute node)
    - Socket path from ONEX_EMIT_SOCKET_PATH or default resolution
    - Event registry loaded from omnimarket registries/claude_code.yaml

Related:
    - OMN-7640: PluginEmitDaemon for kernel
    - OMN-7628: OmniClaude Emit Daemon Extraction to OmniMarket
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from omnibase_infra.runtime.models.model_handshake_result import (
    ModelHandshakeResult,
)
from omnibase_infra.runtime.protocol_domain_plugin import (
    ModelDomainPluginConfig,
    ModelDomainPluginResult,
)

logger = logging.getLogger(__name__)


class PluginEmitDaemon:
    """Emit daemon domain plugin for kernel initialization.

    Starts the omnimarket node_emit_daemon as a background task.
    Activation gated by ONEX_EMIT_DAEMON_ENABLED=true (default false).
    """

    def __init__(self) -> None:
        self._daemon_task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event | None = None

    @property
    def plugin_id(self) -> str:
        return "emit-daemon"

    @property
    def display_name(self) -> str:
        return "Emit Daemon"

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        """Activate only when ONEX_EMIT_DAEMON_ENABLED=true."""
        enabled = os.environ.get("ONEX_EMIT_DAEMON_ENABLED", "false").lower() == "true"
        logger.info(
            "[EMIT-DAEMON] PluginEmitDaemon.should_activate() -> %s "
            "(ONEX_EMIT_DAEMON_ENABLED=%r, correlation_id=%s)",
            enabled,
            os.environ.get("ONEX_EMIT_DAEMON_ENABLED", ""),
            config.correlation_id,
        )
        return enabled

    async def initialize(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Start the emit daemon as a background async task."""
        start_time = time.time()

        try:
            from omnimarket.nodes.node_emit_daemon.event_queue import (
                BoundedEventQueue,
            )
            from omnimarket.nodes.node_emit_daemon.event_registry import (
                EventRegistry,
            )
            from omnimarket.nodes.node_emit_daemon.handlers.handler_emit_daemon import (
                HandlerEmitDaemon,
            )
            from omnimarket.nodes.node_emit_daemon.publisher_loop import (
                KafkaPublisherLoop,
            )
            from omnimarket.nodes.node_emit_daemon.socket_server import (
                EmitSocketServer,
            )
        except ImportError as e:
            duration = time.time() - start_time
            logger.warning(
                "[EMIT-DAEMON] omnimarket not installed, cannot start emit daemon: %s",
                e,
            )
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=f"omnimarket not installed: {e}",
                duration_seconds=duration,
            )

        from pathlib import Path

        # Resolve socket path
        socket_path = os.environ.get("ONEX_EMIT_SOCKET_PATH")
        if not socket_path:
            xdg = os.environ.get("XDG_RUNTIME_DIR")
            if xdg:
                socket_path = str(Path(xdg) / "onex" / "emit.sock")
            else:
                socket_path = "/tmp/onex-emit.sock"  # noqa: S108

        # Resolve spool dir
        spool_dir_str = os.environ.get("ONEX_EMIT_SPOOL_DIR")
        if not spool_dir_str:
            xdg = os.environ.get("XDG_RUNTIME_DIR")
            if xdg:
                spool_dir_str = str(Path(xdg) / "onex" / "event-spool")
            else:
                spool_dir_str = "/tmp/onex-event-spool"  # noqa: S108

        spool_dir = Path(spool_dir_str)

        # Load event registry
        registry_path_str = os.environ.get("ONEX_EMIT_REGISTRY_PATH")
        if registry_path_str:
            registry = EventRegistry.from_yaml(Path(registry_path_str))
        else:
            # Try default claude_code registry from omnimarket
            try:
                import omnimarket.nodes.node_emit_daemon as _pkg

                default_registry = (
                    Path(_pkg.__file__).parent / "registries" / "claude_code.yaml"
                )
                if default_registry.exists():
                    registry = EventRegistry.from_yaml(default_registry)
                else:
                    registry = EventRegistry()
            except Exception:  # noqa: BLE001 — boundary: registry load failure degrades to empty
                registry = EventRegistry()

        # Build publish function
        kafka_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")

        async def _noop_publish(
            topic: str,
            key: bytes | None,
            value: bytes,
            headers: dict[str, str],
        ) -> None:
            logger.debug(
                "[emit-daemon] no-kafka publish to %s (%d bytes)", topic, len(value)
            )

        publish_fn = _noop_publish

        if kafka_servers:
            try:
                from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
                from omnibase_infra.event_bus.models.config import (
                    ModelKafkaEventBusConfig,
                )

                kafka_config = ModelKafkaEventBusConfig(
                    bootstrap_servers=kafka_servers,
                )
                kafka_bus = EventBusKafka(config=kafka_config)
                await kafka_bus.connect()

                async def _kafka_publish(
                    topic: str,
                    key: bytes | None,
                    value: bytes,
                    headers: dict[str, str],
                ) -> None:
                    await kafka_bus.publish_raw(
                        topic=topic, key=key, value=value, headers=headers
                    )

                publish_fn = _kafka_publish
                logger.info("[EMIT-DAEMON] Kafka publishing enabled: %s", kafka_servers)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[EMIT-DAEMON] Failed to connect Kafka, running without: %s", e
                )

        # Create components
        handler = HandlerEmitDaemon()
        queue = BoundedEventQueue(spool_dir=spool_dir)
        server = EmitSocketServer(
            socket_path=socket_path,
            queue=queue,
            registry=registry,
        )
        publisher = KafkaPublisherLoop(queue=queue, publish_fn=publish_fn)

        self._shutdown_event = asyncio.Event()

        async def _run_daemon() -> None:
            """Background task that runs the emit daemon."""
            handler.transition_to_binding(socket_path, os.getpid())
            try:
                await queue.load_spool()
                await server.start()
                await publisher.start()
                handler.transition_to_listening()
            except OSError as e:
                if "Address already in use" in str(e):
                    handler.transition_to_failed(f"Socket already bound: {socket_path}")
                    logger.warning(
                        "[EMIT-DAEMON] Socket %s already bound — "
                        "another daemon is running. Non-fatal.",
                        socket_path,
                    )
                    return
                handler.transition_to_failed(str(e))
                raise
            except Exception as e:
                handler.transition_to_failed(str(e))
                raise

            logger.info(
                "[EMIT-DAEMON] Daemon running (socket=%s, pid=%d)",
                socket_path,
                os.getpid(),
            )

            assert self._shutdown_event is not None
            await self._shutdown_event.wait()

            handler.transition_to_draining()
            await server.stop()
            await publisher.stop()

            drained = await queue.drain_to_spool()
            if drained > 0:
                logger.info("[EMIT-DAEMON] Drained %d events to spool", drained)

            handler.transition_to_stopped(
                events_published=publisher.events_published,
                events_dropped=publisher.events_dropped,
            )

        try:
            self._daemon_task = asyncio.create_task(_run_daemon())
            # Give it a moment to bind the socket
            await asyncio.sleep(0.1)

            # Check if it failed immediately (e.g., socket already bound)
            if self._daemon_task.done():
                exc = self._daemon_task.exception()
                if exc:
                    duration = time.time() - start_time
                    return ModelDomainPluginResult.failed(
                        plugin_id=self.plugin_id,
                        error_message=str(exc),
                        duration_seconds=duration,
                    )

            duration = time.time() - start_time
            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message=f"Emit daemon started (socket={socket_path})",
                resources_created=["emit-daemon-socket", "emit-daemon-publisher"],
                duration_seconds=duration,
            )

        except Exception as e:  # noqa: BLE001
            duration = time.time() - start_time
            logger.error(  # noqa: TRY400
                "[EMIT-DAEMON] Failed to start: %s", e
            )
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=str(e),
                duration_seconds=duration,
            )

    async def validate_handshake(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelHandshakeResult:
        """No handshake checks required for emit daemon."""
        return ModelHandshakeResult.default_pass(self.plugin_id)

    async def wire_handlers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """No handler wiring needed -- emit daemon is a standalone service."""
        return ModelDomainPluginResult.skipped(
            plugin_id=self.plugin_id,
            reason="Emit daemon is a standalone service, no handler wiring needed",
        )

    async def wire_dispatchers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """No dispatcher wiring needed -- emit daemon is a standalone service."""
        return ModelDomainPluginResult.skipped(
            plugin_id=self.plugin_id,
            reason="Emit daemon is a standalone service, no dispatcher wiring needed",
        )

    async def start_consumers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """No consumer wiring needed -- emit daemon is a publisher, not consumer."""
        return ModelDomainPluginResult.skipped(
            plugin_id=self.plugin_id,
            reason="Emit daemon is a publisher, no consumer subscription needed",
        )

    async def shutdown(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Stop the emit daemon gracefully."""
        if self._shutdown_event is not None:
            self._shutdown_event.set()

        if self._daemon_task is not None:
            try:
                async with asyncio.timeout(10.0):
                    await self._daemon_task
            except TimeoutError:
                logger.warning("[EMIT-DAEMON] Shutdown timeout exceeded, cancelling")
                self._daemon_task.cancel()
            except asyncio.CancelledError:
                pass
            self._daemon_task = None

        return ModelDomainPluginResult(
            plugin_id=self.plugin_id,
            success=True,
            message="Emit daemon shut down",
            duration_seconds=0.0,
        )
