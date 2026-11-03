# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane-mirror guard loop for the gateway forwarder (OMN-17201).

Moved verbatim from ``runtime/gateway_forwarder.py`` (OMN-21950) so the lane
mirror's own supervision lives beside the mirror it supervises.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path

from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_lane_mirror_health import (
    publish_lane_mirror_health,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror import (
    NodeLaneMirror,
)

logger = logging.getLogger(__name__)


async def run_lane_mirror_guard(
    *,
    mirror: NodeLaneMirror,
    producers: Mapping[str, KafkaTransport],
    shutdown_event: asyncio.Event,
    interval_seconds: int,
    health_path: Path | None,
) -> None:
    """Publish the mirror's counters and repair a lane it refused (OMN-17201).

    Two jobs on one cadence because they are one contract: the counters are how
    a refused lane becomes visible outside the process, and the recreate is the
    only thing that can end the refusal.

    The repair is deliberately a producer RECREATE, not a retry. A mirror lane
    is refused when a record this process published to it was read back off the
    SOURCE lane at the coordinate it was written to -- proof that the lane's
    producer is connected to the source broker. A Kafka client bootstraps at
    the configured address once and thereafter follows the ADVERTISED address
    it is handed, and both lane brokers can advertise the same bare name
    ``redpanda`` while this forwarder is joined to both lane networks.
    Retrying the same connection reaches the same broker; only dropping it and
    bootstrapping again at the configured address can land elsewhere.

    A recreate that lands on the wrong broker again re-refuses on the next echo,
    so this cannot paper over a genuinely mis-addressed lane:
    ``loop_detected_total`` keeps climbing and the healthcheck's verdict for a
    process that has ever detected a loop is sticky.
    """
    while not shutdown_event.is_set():
        publish_lane_mirror_health(mirror.health, health_path)
        for lane, reason in sorted(mirror.refused_lanes.items()):
            transport = producers.get(lane)
            # Construction guarantees a producer for every refusable lane.
            if transport is None:
                continue
            logger.error(
                "Lane mirror guard recreating lane=%s producer -- lane refused "
                "(%s). No source offset was committed while the lane was "
                "refused, so the retry loses nothing.",
                lane,
                reason,
            )
            try:
                await transport.close()
                await transport.start()
            except Exception:
                logger.exception(
                    "Lane mirror guard could not recreate lane=%s producer; the "
                    "lane stays refused and the source stays uncommitted",
                    lane,
                )
                continue
            mirror.clear_lane_refusal(lane)
            logger.warning(
                "Lane mirror guard recreated lane=%s producer and cleared the "
                "refusal; the held records redeliver from the source",
                lane,
            )
        publish_lane_mirror_health(mirror.health, health_path)
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue


__all__ = ["run_lane_mirror_guard"]
