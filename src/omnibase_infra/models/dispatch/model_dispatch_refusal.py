# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Failure terminal for a command the dispatch engine refused as malformed (OMN-26026).

A command that fails its handler's input validation is classified
``publisher_malformed`` by the engine and dead-lettered. The caller waiting on
the contract's failure terminal never heard of it. This payload is that answer:
``status="failed"`` and ``failure_reason`` are the fields the existing terminal
readers (``resolve_terminal_verdict``, the Pattern B broker) already read.
``topic`` names the contract-declared failure topic, which the result applier
honors as the embedded output topic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.enums import EnumCoreErrorCode, EnumDispatchStatus
from omnibase_infra.enums import EnumDlqFailureClass
from omnibase_infra.models.dispatch.model_failure_terminal_bindings import (
    ModelFailureTerminalBindings,
)
from omnibase_infra.models.dispatch.model_no_dispatcher_refusal import (
    ModelNoDispatcherRefusal,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
        ModelDiscoveredContract,
    )

__all__ = [
    "PARTIAL_STATUSES",
    "bind_contract_failure_terminal",
    "select_dispatch_result_output_topic",
    "classify_no_dispatcher_refusal",
]

_VALIDATION_DETAIL_MAX_LENGTH = 500

# A refused command's terminal and a sibling handler's output are published even
# though the aggregate status is not SUCCESS (OMN-12416, OMN-26026).
PARTIAL_STATUSES = frozenset(
    {EnumDispatchStatus.HANDLER_ERROR, EnumDispatchStatus.NO_DISPATCHER}
)


def classify_no_dispatcher_refusal(
    *,
    type_scoped_candidate_rejected: bool,
    raw_validation_detail: str | None,
    message_type: str,
    topic: str,
    topic_category: object,
) -> ModelNoDispatcherRefusal:
    """Classify an empty dispatcher match: payload refused, or a wiring gap.

    When the type-scoping pass rejected a real candidate, a dispatcher IS
    registered for this event_type and the payload failed its event_model
    validation: a publisher defect, carrying the real validation detail.
    """
    if not type_scoped_candidate_rejected:
        return ModelNoDispatcherRefusal(
            failure_class=EnumDlqFailureClass.NO_DISPATCHER,
            error_code=EnumCoreErrorCode.ITEM_NOT_REGISTERED,
            error_message=(
                f"No dispatcher registered for category '{topic_category}' "
                f"and message type '{message_type}' matching topic '{topic}'."
            ),
        )
    detail = sanitize_error_message(
        ValueError(raw_validation_detail or "validation detail unavailable"),
        max_length=_VALIDATION_DETAIL_MAX_LENGTH,
    )
    return ModelNoDispatcherRefusal(
        failure_class=EnumDlqFailureClass.PUBLISHER_MALFORMED,
        error_code=EnumCoreErrorCode.ENVELOPE_VALIDATION_FAILED,
        validation_detail=detail,
        error_message=(
            f"Publisher-malformed payload for message type '{message_type}' "
            f"on topic '{topic}': a dispatcher IS registered for this "
            f"event_model, but the payload failed model_validate: {detail}"
        ),
    )


def select_dispatch_result_output_topic(
    contract: ModelDiscoveredContract,
) -> str | None:
    """Choose the fallback output topic for generic dispatch results.

    Prefer the contract's declared terminal_event when it is also publishable.
    Otherwise fall back to the first publish topic to preserve older contracts.
    """
    if contract.event_bus is None or not contract.event_bus.publish_topics:
        return None
    if (
        contract.terminal_event
        and contract.terminal_event in contract.event_bus.publish_topics
    ):
        return str(contract.terminal_event)
    return str(contract.event_bus.publish_topics[0])


def bind_contract_failure_terminal(
    dispatch_engine: object,
    contract: ModelDiscoveredContract,
    failure_topics: tuple[str, ...],
) -> None:
    """Bind a contract's subscribe topics to its single declared failure topic.

    With two or more declared failure terminals there is no basis to choose, so
    the contract keeps the DLQ-only behavior. A frozen or absent registry (a
    dynamic materialization, a non-engine dispatcher) is left alone.
    """
    bindings = getattr(dispatch_engine, "failure_terminals", None)
    if not isinstance(bindings, ModelFailureTerminalBindings) or bindings.is_frozen:
        return
    if len(failure_topics) != 1 or contract.event_bus is None:
        return
    for command_topic in contract.event_bus.subscribe_topics:
        bindings.bind(command_topic=command_topic, failure_topic=failure_topics[0])
