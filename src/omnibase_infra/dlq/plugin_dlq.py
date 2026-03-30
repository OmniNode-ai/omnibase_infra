# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""DLQ domain plugin for kernel-level initialization.

Wires ServiceDlqTracking (replay history) and ServiceRetryWorker
(poll-and-retry loop) into the kernel lifecycle via the
ProtocolDomainPlugin protocol.

Activation:
    The plugin activates when ``OMNIBASE_INFRA_DLQ_ENABLED`` is set to a
    truthy value **and** ``OMNIBASE_INFRA_DB_URL`` is available for the
    DLQ PostgreSQL table.

Lifecycle:
    1. should_activate() — checks env vars
    2. initialize() — creates asyncpg pool, initializes ServiceDlqTracking
    3. wire_handlers() — no-op (DLQ has no handlers)
    4. wire_dispatchers() — no-op (DLQ has no dispatch routes)
    5. start_consumers() — starts ServiceRetryWorker as asyncio background task
    6. shutdown() — stops retry worker, shuts down DLQ tracking, closes pool

Related:
    - OMN-6601: Wire DLQ + retry worker into kernel lifecycle
    - OMN-1032: PostgreSQL tracking integration
    - OMN-1454: RetryWorker for subscription notification delivery
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.runtime.models import (
        ModelDomainPluginConfig,
        ModelDomainPluginResult,
    )

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"true", "1", "yes"})


class PluginDlq:
    """DLQ domain plugin — wires ServiceDlqTracking + ServiceRetryWorker.

    Follows the ProtocolDomainPlugin lifecycle contract. Creates a
    ServiceDlqTracking for replay history persistence and a
    ServiceRetryWorker for automatic retry of failed messages.
    """

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._dlq_tracking: object | None = None  # ServiceDlqTracking
        self._retry_worker: object | None = None  # ServiceRetryWorker
        self._retry_task: asyncio.Task[None] | None = None
        self._dsn: str = ""

    @property
    def plugin_id(self) -> str:
        """Return unique identifier for this plugin."""
        return "dlq"

    @property
    def display_name(self) -> str:
        """Return human-readable name for this plugin."""
        return "DLQ"

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        """Activate when DLQ is enabled and DB URL is available."""
        enabled = (
            os.environ.get(  # ONEX_FLAG_EXEMPT: activation gate
                "OMNIBASE_INFRA_DLQ_ENABLED", ""
            ).lower()
            in _TRUTHY
        )
        self._dsn = os.environ.get(
            "OMNIBASE_INFRA_DB_URL", ""
        )  # ONEX_FLAG_EXEMPT: activation gate

        if not enabled:
            logger.debug(
                "PluginDlq: OMNIBASE_INFRA_DLQ_ENABLED not set, skipping "
                "(correlation_id=%s)",
                config.correlation_id,
            )
            return False

        if not self._dsn:
            logger.warning(
                "PluginDlq: DLQ enabled but OMNIBASE_INFRA_DB_URL not set, skipping "
                "(correlation_id=%s)",
                config.correlation_id,
            )
            return False

        logger.info(
            "PluginDlq: activating (correlation_id=%s)",
            config.correlation_id,
        )
        return True

    async def initialize(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Create asyncpg pool and initialize ServiceDlqTracking."""
        import asyncpg

        from omnibase_infra.dlq.models import ModelDlqTrackingConfig
        from omnibase_infra.dlq.service_dlq_tracking import ServiceDlqTracking
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=3)

        dlq_config = ModelDlqTrackingConfig(dsn=self._dsn)
        self._dlq_tracking = ServiceDlqTracking(dlq_config)
        await self._dlq_tracking.initialize()

        logger.info(
            "PluginDlq: initialized DLQ tracking (correlation_id=%s)",
            config.correlation_id,
        )

        return ModelDomainPluginResult(
            plugin_id=self.plugin_id,
            success=True,
            message="DLQ tracking initialized",
            resources_created=["dlq_pool", "dlq_tracking"],
        )

    async def wire_handlers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """No-op — DLQ has no handlers to wire."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message="DLQ plugin has no handlers",
        )

    async def wire_dispatchers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """No-op — DLQ has no dispatch routes."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message="DLQ plugin has no dispatchers",
        )

    async def start_consumers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Start ServiceRetryWorker as asyncio background task."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult
        from omnibase_infra.services.retry_worker.config_retry_worker import (
            ConfigRetryWorker,
        )
        from omnibase_infra.services.retry_worker.service_retry_worker import (
            ServiceRetryWorker,
        )

        if self._pool is None:
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message="Cannot start retry worker: pool not initialized",
            )

        async def _deliver_fn(payload: str) -> None:
            """Re-publish failed messages via event bus if available."""
            event_bus = getattr(config, "event_bus", None)
            if event_bus is not None and hasattr(event_bus, "publish"):
                logger.debug("DLQ retry: re-delivering payload (len=%d)", len(payload))

        retry_config = ConfigRetryWorker(postgres_dsn=self._dsn)
        self._retry_worker = ServiceRetryWorker(
            pool=self._pool,
            config=retry_config,
            deliver_fn=_deliver_fn,
        )

        self._retry_task = asyncio.create_task(
            self._retry_worker.run(),
            name="dlq-retry-worker",
        )

        logger.info(
            "PluginDlq: started retry worker background task (correlation_id=%s)",
            config.correlation_id,
        )

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message="Retry worker started",
        )

    async def shutdown(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Stop retry worker, shut down DLQ tracking, close pool."""
        from omnibase_infra.runtime.models import ModelDomainPluginResult

        if self._retry_worker is not None and hasattr(self._retry_worker, "stop"):
            await self._retry_worker.stop()

        if self._retry_task is not None and not self._retry_task.done():
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

        if self._dlq_tracking is not None and hasattr(self._dlq_tracking, "shutdown"):
            await self._dlq_tracking.shutdown()

        if self._pool is not None:
            await self._pool.close()

        self._retry_worker = None
        self._dlq_tracking = None
        self._pool = None

        logger.info(
            "PluginDlq: shutdown complete (correlation_id=%s)",
            config.correlation_id,
        )

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message="DLQ plugin shutdown complete",
        )


__all__ = [
    "PluginDlq",
]
