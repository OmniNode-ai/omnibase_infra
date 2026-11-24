# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One ``boot summary:`` line per boot, written at the first healthy answer (OMN-26561).

Three boot-performance changes (each contract parsed once, the topic scan and route
discovery off the event loop, boot step 0 holds removed) reported nothing a boot log
could read back: the parse counter was never logged and the loop-stall detector
starts with the health listener, after boot step 0. This module is the instrument.

:class:`~omnibase_infra.models.node_announce.model_boot_clock.ModelBootClock` samples
how late the event loop wakes a short scheduled sleep from the first boot phase entered
(boot step 0 at the latest) and keeps the longest gap per phase. At the first
``/health`` 200 :func:`emit_boot_summary_once` hands :class:`HandlerBootSummary` the
contract parse count, the topic scan count, the seconds since the kernel started and
those gaps; the handler renders and logs the one line, and the clock stops.
"""

from __future__ import annotations

import logging
from typing import Final

from omnibase_infra.contracts.contract_read_cache import ContractReadCache
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.event_bus.model_contract_topic_scan import (
    ModelContractTopicScan,
)
from omnibase_infra.models.node_announce.model_boot_clock import (
    OTHER_PHASE,
    ModelBootClock,
)
from omnibase_infra.models.node_announce.model_boot_summary_request import (
    ModelBootSummaryRequest,
)
from omnibase_infra.models.node_announce.model_boot_summary_result import (
    ModelBootSummaryResult,
)

logger = logging.getLogger(__name__)

BOOT_PHASES: Final[tuple[str, ...]] = (
    "boot_step0",
    "provisioner",
    "wiring_pass",
    "local_ingress_start",
)
"""The boot phases whose longest loop gap the summary reports, in boot order."""

_HEALTHY_STATUS: Final = 200


class HandlerBootSummary:
    """Render and log the boot summary line from the counts it is given."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(self, request: ModelBootSummaryRequest) -> ModelBootSummaryResult:
        gaps = " ".join(
            f"gap_{name}="
            + (
                f"{request.phase_gap_seconds[name]:.3f}"
                if name in request.phase_gap_seconds
                else "none"
            )
            for name in (*BOOT_PHASES, OTHER_PHASE)
        )
        line = (
            f"boot summary: parse_runs={request.parse_runs} "
            f"topic_scans={request.topic_scans} "
            f"seconds_since_kernel_start={request.seconds_since_kernel_start:.1f} "
            f"{gaps}"
        )
        logger.info("%s", line)
        return ModelBootSummaryResult(line=line)


def emit_boot_summary_once(http_status: int) -> None:
    """Write the boot summary line at the first ``/health`` 200; later calls do nothing."""
    if http_status != _HEALTHY_STATUS:
        return
    gaps = ModelBootClock.finish()
    if gaps is None:
        return
    handler = HandlerBootSummary()
    handler.handle(
        ModelBootSummaryRequest(
            parse_runs=ContractReadCache.parse_runs(),
            topic_scans=ModelContractTopicScan.runs(),
            seconds_since_kernel_start=ModelBootClock.seconds_since_kernel_start(),
            phase_gap_seconds=gaps,
        )
    )


__all__ = ["BOOT_PHASES", "HandlerBootSummary", "emit_boot_summary_once"]
