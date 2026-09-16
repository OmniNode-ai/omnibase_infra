# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""DLQ replay dependency lifecycle refusal (OMN-17137, second pass).

``HandlerDlqReplay`` starts and stops its Kafka dependencies inside the very
dispatch the outer auto-wired trigger consumer awaits. Those lifecycle calls sat
OUTSIDE ``max_run_duration_seconds``, which bounds only the record loop, so each
of them was an unbounded await on the critical path of a consumer that must keep
polling.

Measured on the .201 dev lane 2026-09-16: the node's three per-topic DLQ
consumers share the single ``onex-dlq-replay`` group and the handler starts and
stops one of them on EVERY trigger message, so the group rebalances without
pause -- generation 227,675 eight minutes after a cold boot. A lifecycle call
issued into that storm does not settle. The dispatch then never returns, the
outer consumer stops polling, aiokafka evicts it at ``max_poll_interval_ms``,
and nothing rejoins it, because a rejoin only happens on the next poll.

A START that exceeds its budget is a typed refusal rather than a silent skip:
the run has no consumer and must end, and the caller's existing ``except`` path
tears down what it did start. A STOP that exceeds its budget is deliberately NOT
raised -- see ``_stop_runtime_dependencies``; the batch it is tearing down has
already committed durable work, and raising would discard that result and have
every record in it redelivered.
"""

from __future__ import annotations

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class DlqDependencyLifecycleTimeoutError(RuntimeHostError):
    """Raised when a DLQ replay dependency's ``start()`` exceeds its budget.

    The budget is ``ModelDlqReplayEngineConfig.dependency_lifecycle_timeout_seconds``.
    Ending the run is the correct answer: a bounded dispatch that returns
    without draining keeps the outer trigger consumer polling, and the topic is
    retried on the next trigger. An unbounded one takes the consumer group down
    permanently.

    Example:
        >>> raise DlqDependencyLifecycleTimeoutError(
        ...     "DLQConsumer.start() did not complete within 15.00s",
        ...     dependency="DLQConsumer",
        ...     operation="start",
        ...     timeout_seconds=15.0,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize DlqDependencyLifecycleTimeoutError.

        Args:
            message: Human-readable message naming the dependency and the bound
            context: Bundled infrastructure context
            **extra_context: Additional context (dependency, operation,
                timeout_seconds, ...)
        """
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.TIMEOUT,
            context=context,
            **extra_context,
        )
