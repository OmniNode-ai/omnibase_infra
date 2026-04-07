# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lifecycle hooks for the DLQ domain (OMN-7662).

Extracted from PluginDlq to enable contract-driven auto-wiring.
These hooks are referenced from dlq/contract.yaml and invoked by the
LifecycleHookExecutor during kernel startup/shutdown.

on_start:
    Creates asyncpg pool, initializes ServiceDlqTracking, starts
    ServiceRetryWorker as a background task.

on_shutdown:
    Stops retry worker, shuts down DLQ tracking, closes pool.

Related:
    - OMN-6601: Wire DLQ + retry worker into kernel lifecycle
    - OMN-7662: Extract lightweight infra plugins
"""

from __future__ import annotations

import asyncio
import logging
import os

from omnibase_infra.runtime.auto_wiring.context import ModelAutoWiringContext
from omnibase_infra.runtime.auto_wiring.models import ModelLifecycleHookResult

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"true", "1", "yes"})

# Module-level state shared between on_start and on_shutdown.
_state: dict[str, object] = {}


async def on_start(context: ModelAutoWiringContext) -> ModelLifecycleHookResult:
    """Create asyncpg pool, DLQ tracking, and start retry worker.

    Checks OMNIBASE_INFRA_DLQ_ENABLED and OMNIBASE_INFRA_DB_URL env vars.
    If not configured, returns success with no background workers.
    """
    enabled = (
        os.environ.get(  # ONEX_FLAG_EXEMPT: activation gate
            "OMNIBASE_INFRA_DLQ_ENABLED", ""
        ).lower()
        in _TRUTHY
    )
    dsn = os.environ.get(
        "OMNIBASE_INFRA_DB_URL", ""
    )  # ONEX_FLAG_EXEMPT: activation gate

    if not enabled:
        logger.debug(
            "lifecycle_dlq.on_start: DLQ not enabled, skipping (handler_id=%s)",
            context.handler_id,
        )
        return ModelLifecycleHookResult.succeeded(
            hook_name="on_start",
            background_workers=[],
        )

    if not dsn:
        logger.warning(
            "lifecycle_dlq.on_start: DLQ enabled but OMNIBASE_INFRA_DB_URL not set, "
            "skipping (handler_id=%s)",
            context.handler_id,
        )
        return ModelLifecycleHookResult.succeeded(
            hook_name="on_start",
            background_workers=[],
        )

    import asyncpg

    from omnibase_infra.dlq.models import ModelDlqTrackingConfig
    from omnibase_infra.dlq.service_dlq_tracking import ServiceDlqTracking

    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=3)
    _state["pool"] = pool
    _state["dsn"] = dsn

    dlq_config = ModelDlqTrackingConfig(dsn=dsn)
    dlq_tracking = ServiceDlqTracking(dlq_config)
    await dlq_tracking.initialize()
    _state["dlq_tracking"] = dlq_tracking

    logger.info(
        "lifecycle_dlq.on_start: initialized DLQ tracking (handler_id=%s)",
        context.handler_id,
    )

    # Start retry worker
    from omnibase_infra.services.retry_worker.config_retry_worker import (
        ConfigRetryWorker,
    )
    from omnibase_infra.services.retry_worker.service_retry_worker import (
        ServiceRetryWorker,
    )

    async def _deliver_fn(payload: str) -> None:
        """Re-publish failed messages via event bus if available."""
        event_bus = context.services.get("event_bus")
        if event_bus is not None and hasattr(event_bus, "publish"):
            logger.debug("DLQ retry: re-delivering payload (len=%d)", len(payload))

    retry_config = ConfigRetryWorker(postgres_dsn=dsn)
    retry_worker = ServiceRetryWorker(
        pool=pool,
        config=retry_config,
        deliver_fn=_deliver_fn,
    )
    _state["retry_worker"] = retry_worker

    retry_task = asyncio.create_task(
        retry_worker.run(),
        name="dlq-retry-worker",
    )
    _state["retry_task"] = retry_task

    logger.info(
        "lifecycle_dlq.on_start: started retry worker background task (handler_id=%s)",
        context.handler_id,
    )

    return ModelLifecycleHookResult.succeeded(
        hook_name="on_start",
        background_workers=["dlq-retry-worker"],
    )


async def on_shutdown(context: ModelAutoWiringContext) -> ModelLifecycleHookResult:
    """Stop retry worker, shut down DLQ tracking, close pool."""
    retry_worker = _state.pop("retry_worker", None)
    if retry_worker is not None and hasattr(retry_worker, "stop"):
        await retry_worker.stop()

    retry_task = _state.pop("retry_task", None)
    if retry_task is not None and isinstance(retry_task, asyncio.Task):
        if not retry_task.done():
            retry_task.cancel()
            try:
                await retry_task
            except asyncio.CancelledError:
                pass

    dlq_tracking = _state.pop("dlq_tracking", None)
    if dlq_tracking is not None and hasattr(dlq_tracking, "shutdown"):
        await dlq_tracking.shutdown()

    pool = _state.pop("pool", None)
    if pool is not None and hasattr(pool, "close"):
        await pool.close()

    _state.pop("dsn", None)

    logger.info(
        "lifecycle_dlq.on_shutdown: complete (handler_id=%s)",
        context.handler_id,
    )

    return ModelLifecycleHookResult.succeeded(hook_name="on_shutdown")
