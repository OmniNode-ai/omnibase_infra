# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A consumed record's handler outlived its per-dispatch deadline (OMN-19355).

The serial consume loop in ``EventBusKafka`` awaits each subscriber callback
before it polls again. Projection handlers run their blocking work through
``asyncio.to_thread``, so a handler that never returns parks a worker thread
and, with it, the loop. Measured on the .201 dev lane on 2026-09-23: the
``lab_lane_health`` group stopped polling at offset 51808, aiokafka evicted the
member at ``max_poll_interval_ms`` (1800000), and it never rejoined, because a
rejoin only happens inside the next ``getmany``. Auto-commit had already
committed the fetch position past the hung record, so records 51809 to 51832
were never processed. A ``/proc`` read showed one ``to_thread`` worker parked in
``epoll_wait`` with timeout -1.

This type is the REASON carried on the quarantine that ends such a dispatch. It
is not raised by handlers. The handler it names is abandoned, not cancelled: a
Python thread cannot be killed, so if the handler is parked in a worker thread
that thread stays parked until it returns or the process exits.
"""

from __future__ import annotations

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class DispatchDeadlineExceededError(RuntimeHostError):
    """A subscriber callback did not return within the per-dispatch deadline.

    The deadline is ``ModelKafkaEventBusConfig.effective_dispatch_deadline_seconds``.

    Example:
        >>> raise DispatchDeadlineExceededError(
        ...     "handler sub-1 did not return within 600.0s",
        ...     topic="onex.evt.omnibase-infra.lab-lane-health.v1",
        ...     partition=0,
        ...     offset=51808,
        ...     deadline_seconds=600.0,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize DispatchDeadlineExceededError.

        Args:
            message: Human-readable message naming the record and the deadline
            context: Bundled infrastructure context
            **extra_context: Additional context (topic, partition, offset,
                subscription_id, deadline_seconds, ...)
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.TIMEOUT,
            context=context,
            **extra_context,
        )


__all__ = ["DispatchDeadlineExceededError"]
