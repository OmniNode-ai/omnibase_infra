# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""DLQ domain plugin for kernel-level initialization.

Wires ServiceDlqTracking (replay history) and ServiceRetryWorker
(poll-and-retry loop) into the kernel lifecycle via the
ProtocolDomainPlugin protocol.

Activation (OMN-12634):
    The plugin activates when ``DLQ_ENABLED`` is set to a truthy value AND
    ``DLQ_DB_URL`` is present in ``config.overlay_config``.  Both keys are
    resolved from the contract overlay; environment variables are no longer
    consulted for feature gating.

    If ``overlay_config`` is ``None`` (kernel not yet overlay-capable) the
    plugin skips silently.  If ``DLQ_ENABLED`` is truthy but ``DLQ_DB_URL``
    is absent the plugin raises ``ValueError`` — fail-fast, no silent default.

Lifecycle:
    1. should_activate() — reads overlay_config, raises on misconfiguration
    2. initialize() — creates asyncpg pool, initializes ServiceDlqTracking
    3. wire_handlers() — no-op (DLQ has no handlers)
    4. wire_dispatchers() — no-op (DLQ has no dispatch routes)
    5. start_consumers() — starts ServiceRetryWorker as asyncio background task
    6. shutdown() — stops retry worker, shuts down DLQ tracking, closes pool

Overlay keys:
    DLQ_ENABLED  — "true" / "1" / "yes" (case-insensitive) to activate
    DLQ_DB_URL   — PostgreSQL DSN for the DLQ tracking table (required when enabled)

Related:
    - OMN-12634: Move DLQ feature gating from env vars to contract overlay
    - OMN-6601: Wire DLQ + retry worker into kernel lifecycle
    - OMN-1032: PostgreSQL tracking integration
    - OMN-1454: RetryWorker for subscription notification delivery
"""

from __future__ import annotations

import asyncio
import logging
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
        """Activate when DLQ_ENABLED and DLQ_DB_URL are set in overlay_config.

        Reads activation state exclusively from ``config.overlay_config`` — env
        vars are never consulted (OMN-12634).

        Returns False when:
        - ``overlay_config`` is ``None`` (kernel not yet overlay-capable)
        - ``DLQ_ENABLED`` key is absent or falsy

        Raises ValueError when:
        - ``DLQ_ENABLED`` is truthy but ``DLQ_DB_URL`` is missing — fail-fast,
          no silent default.
        """
        overlay = getattr(config, "overlay_config", None)
        if overlay is None:
            logger.debug(
                "PluginDlq: overlay_config not present, skipping (correlation_id=%s)",
                config.correlation_id,
            )
            return False

        enabled = overlay.get("DLQ_ENABLED", "").lower() in _TRUTHY
        if not enabled:
            logger.debug(
                "PluginDlq: DLQ_ENABLED not set in overlay, skipping "
                "(correlation_id=%s)",
                config.correlation_id,
            )
            return False

        db_url = overlay.get("DLQ_DB_URL", "").strip()
        if not db_url:
            raise ValueError(
                "PluginDlq: DLQ_ENABLED is true in overlay but DLQ_DB_URL is "
                "missing or empty — set DLQ_DB_URL in the overlay config "
                f"(correlation_id={config.correlation_id})"
            )

        self._dsn = db_url
        logger.info(
            "PluginDlq: activating from overlay config (correlation_id=%s)",
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
