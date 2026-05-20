# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection freshness SLA monitor (OMN-11200).

Periodically checks each declared ModelProjectionContract against a database
clock and emits ModelProjectionDegradedEvent or ModelProjectionRecoveredEvent
as freshness state transitions occur.

Rules:
- Freshness is determined solely by querying MAX({freshness_field}) FROM
  {freshness_source_table}. Never infer from arbitrary timestamp columns.
- If freshness_field or freshness_source_table is absent: state is UNKNOWN,
  skip the contract.
- Degraded → Recovered transitions emit a recovery event.
- Duplicate events are suppressed: one degraded event per breach, one recovery
  per resolution.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.health.model_projection_degraded_event import (
    ModelProjectionDegradedEvent,
)
from omnibase_infra.models.health.model_projection_recovered_event import (
    ModelProjectionRecoveredEvent,
)
from omnibase_infra.models.projection.model_projection_contract import (
    ModelProjectionContract,
)
from omnibase_infra.topics import topic_keys
from omnibase_infra.utils.correlation import generate_correlation_id

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
    from omnibase_infra.protocols.protocol_topic_registry import ProtocolTopicRegistry

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL: float = 60.0

_FreshnessPayload = ModelProjectionDegradedEvent | ModelProjectionRecoveredEvent

# Type alias for the async DB query callable injected by callers / tests.
# Signature: (table, field) -> datetime | None
QueryFn = Callable[[str, str], Coroutine[None, None, datetime | None]]


def _contract_hash(contract: ModelProjectionContract) -> str:
    """Return a short deterministic hash of the contract's identity fields."""
    payload = json.dumps(
        {
            "name": contract.projection_name,
            "table": contract.freshness_source_table,
            "field": contract.freshness_field,
            "sla": contract.freshness_sla_seconds,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class ServiceFreshnessMonitor:
    """Periodic projection freshness SLA monitor.

    Usage::

        async def query(table: str, field: str) -> datetime | None:
            row = await db.fetchrow(f"SELECT MAX({field}) FROM {table}")
            return row[0] if row else None

        monitor = ServiceFreshnessMonitor(
            contracts=PROJECTION_CONTRACTS,
            query_fn=query,
            event_bus=bus,
        )
        await monitor.start()
        ...
        await monitor.stop()
    """

    def __init__(
        self,
        contracts: tuple[ModelProjectionContract, ...],
        query_fn: QueryFn,
        event_bus: ProtocolEventBusLike | None = None,
        check_interval_seconds: float = _DEFAULT_INTERVAL,
        topic_registry: ProtocolTopicRegistry | None = None,
    ) -> None:
        if check_interval_seconds <= 0:
            raise ValueError(
                f"check_interval_seconds must be positive, got {check_interval_seconds}"
            )

        self._contracts = contracts
        self._query_fn = query_fn
        self._event_bus = event_bus
        self._check_interval = check_interval_seconds
        self._degraded: set[str] = set()
        self._task: asyncio.Task[None] | None = None
        self._running = False

        if topic_registry is None:
            from omnibase_infra.topics.service_topic_registry import (
                ServiceTopicRegistry,
            )

            topic_registry = ServiceTopicRegistry.from_defaults()

        self._topic_degraded = topic_registry.resolve(
            topic_keys.PROJECTION_FRESHNESS_DEGRADED
        )
        self._topic_recovered = topic_registry.resolve(
            topic_keys.PROJECTION_FRESHNESS_RECOVERED
        )

    async def start(self) -> None:
        """Start the background check loop. Idempotent."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="freshness-monitor")
        logger.info(
            "ServiceFreshnessMonitor started (contracts=%d interval=%ds)",
            len(self._contracts),
            int(self._check_interval),
        )

    async def stop(self) -> None:
        """Stop the background check loop. Idempotent."""
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        logger.info("ServiceFreshnessMonitor stopped")

    async def run_once(
        self,
    ) -> list[_FreshnessPayload]:
        """Run a single check cycle across all contracts.

        Returns:
            All events emitted during this cycle (degraded + recovered), in
            the order they were produced.
        """
        now = datetime.now(UTC)
        emitted: list[_FreshnessPayload] = []

        for contract in self._contracts:
            if not contract.freshness_field or not contract.freshness_source_table:
                logger.debug(
                    "Freshness monitor: skipping %s (no freshness_field or "
                    "freshness_source_table declared)",
                    contract.projection_name,
                )
                continue

            contract_hash = _contract_hash(contract)
            try:
                max_ts = await self._query_fn(
                    contract.freshness_source_table, contract.freshness_field
                )
            except Exception:  # noqa: BLE001 — boundary; never crash the loop
                logger.warning(
                    "Freshness monitor: query failed for %s (will retry next cycle)",
                    contract.projection_name,
                    exc_info=True,
                )
                continue

            if max_ts is None:
                logger.debug(
                    "Freshness monitor: %s returned NULL — no rows yet, skipping",
                    contract.projection_name,
                )
                continue

            # Ensure max_ts is timezone-aware for arithmetic.
            if max_ts.tzinfo is None:
                max_ts = max_ts.replace(tzinfo=UTC)

            staleness = (now - max_ts).total_seconds()
            is_stale = staleness > contract.freshness_sla_seconds
            was_degraded = contract.projection_name in self._degraded

            if is_stale and not was_degraded:
                self._degraded.add(contract.projection_name)
                event: _FreshnessPayload = ModelProjectionDegradedEvent(
                    projection_name=contract.projection_name,
                    sla_seconds=contract.freshness_sla_seconds,
                    actual_staleness_seconds=staleness,
                    degraded_behavior=contract.degraded_semantics.value,
                    observed_at=now,
                    source_contract_hash=contract_hash,
                )
                logger.warning(
                    "Projection %s is stale: staleness=%.1fs sla=%ds behavior=%s",
                    contract.projection_name,
                    staleness,
                    contract.freshness_sla_seconds,
                    contract.degraded_semantics,
                )
                await self._emit(event, self._topic_degraded)
                emitted.append(event)

            elif not is_stale and was_degraded:
                self._degraded.discard(contract.projection_name)
                recovered_event = ModelProjectionRecoveredEvent(
                    projection_name=contract.projection_name,
                    sla_seconds=contract.freshness_sla_seconds,
                    actual_staleness_seconds=staleness,
                    observed_at=now,
                    source_contract_hash=contract_hash,
                )
                logger.info(
                    "Projection %s recovered: staleness=%.1fs sla=%ds",
                    contract.projection_name,
                    staleness,
                    contract.freshness_sla_seconds,
                )
                await self._emit(recovered_event, self._topic_recovered)
                emitted.append(recovered_event)

        return emitted

    # -- Internal ----------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                await self.run_once()
            except Exception:  # noqa: BLE001 — never crash the loop
                logger.warning(
                    "ServiceFreshnessMonitor loop iteration failed (will retry in %ds)",
                    int(self._check_interval),
                    exc_info=True,
                )

    async def _emit(
        self,
        event: _FreshnessPayload,
        topic: str,
    ) -> None:
        if self._event_bus is None:
            return
        event_type = (
            "projection-freshness-degraded"
            if isinstance(event, ModelProjectionDegradedEvent)
            else "projection-freshness-recovered"
        )
        envelope: ModelEventEnvelope[_FreshnessPayload] = ModelEventEnvelope(
            payload=event,
            correlation_id=generate_correlation_id(),
            event_type=event_type,
            source_tool="ServiceFreshnessMonitor",
        )
        try:
            await self._event_bus.publish_envelope(envelope=envelope, topic=topic)
        except Exception:
            logger.exception(
                "ServiceFreshnessMonitor: failed to emit %s for %s",
                event_type,
                event.projection_name,
            )


__all__: list[str] = ["ServiceFreshnessMonitor"]
