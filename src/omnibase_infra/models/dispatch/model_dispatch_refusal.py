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

import threading
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums import EnumCoreErrorCode, EnumDispatchStatus
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.enums import EnumDlqFailureClass
from omnibase_infra.utils import sanitize_error_message

__all__ = [
    "ModelFailureTerminalBindings",
    "ModelMalformedCommandTerminal",
    "ModelNoDispatcherRefusal",
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


class ModelMalformedCommandTerminal(BaseModel):
    """Typed terminal for a publisher-malformed command."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str = Field(..., min_length=1, description="Declared failure topic.")
    correlation_id: UUID = Field(..., description="Correlation of the refused command.")
    workflow_id: str = Field(
        default="", description="Workflow id carried by the command; empty if none."
    )
    status: Literal["failed"] = Field(default="failed")
    failure_class: Literal["publisher_malformed"] = Field(default="publisher_malformed")
    failure_code: str | None = Field(default=None)
    retryable: Literal[False] = Field(default=False)
    failure_reason: str = Field(..., description="Sanitized validation detail.")
    origin_topic: str = Field(..., description="Command topic that was refused.")


class ModelNoDispatcherRefusal(BaseModel):
    """Why the engine found no dispatcher for a message (OMN-14492)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_class: EnumDlqFailureClass
    error_code: EnumCoreErrorCode
    error_message: str
    validation_detail: str | None = None


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


class ModelFailureTerminalBindings:
    """Injected command topic -> declared failure topic map (OMN-26026).

    Bound at wiring time, before the engine freezes. A topic with no binding
    keeps the DLQ-only behavior.
    """

    def __init__(self) -> None:
        self._topics: dict[str, str] = {}
        self._frozen = False
        self._lock = threading.Lock()

    def bind(self, *, command_topic: str, failure_topic: str) -> None:
        """Bind one command topic to its failure topic; refused once frozen."""
        if not command_topic.strip() or not failure_topic.strip():
            raise ModelOnexError(
                message="failure terminal binding requires non-empty topics",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )
        with self._lock:
            if self._frozen:
                raise ModelOnexError(
                    message="Cannot bind failure terminal: engine is frozen.",
                    error_code=EnumCoreErrorCode.INVALID_STATE,
                )
            self._topics[command_topic] = failure_topic

    @property
    def is_frozen(self) -> bool:
        """Whether further bindings are refused."""
        return self._frozen

    def freeze(self) -> None:
        """Refuse further bindings."""
        with self._lock:
            self._frozen = True

    def answer(
        self,
        *,
        topic: str,
        refusal: ModelNoDispatcherRefusal,
        correlation_id: UUID,
        payload: object,
    ) -> list[BaseModel]:
        """Return the typed terminal for a malformed command, else no events."""
        failure_topic = self._topics.get(topic)
        if (
            failure_topic is None
            or refusal.failure_class is not EnumDlqFailureClass.PUBLISHER_MALFORMED
        ):
            return []
        raw = (
            payload.get("workflow_id")
            if isinstance(payload, dict)
            else getattr(payload, "workflow_id", None)
        )
        return [
            ModelMalformedCommandTerminal(
                topic=failure_topic,
                correlation_id=correlation_id,
                workflow_id="" if raw is None else str(raw),
                failure_code=refusal.error_code.value,
                failure_reason=refusal.error_message,
                origin_topic=topic,
            )
        ]


def select_dispatch_result_output_topic(contract: Any) -> str | None:
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
    dispatch_engine: object, contract: Any, failure_topics: tuple[str, ...]
) -> None:
    """Bind a contract's subscribe topics to its single declared failure topic.

    With two or more declared failure terminals there is no basis to choose, so
    the contract keeps the DLQ-only behavior. A frozen or absent registry (a
    dynamic materialization, a non-engine dispatcher) is left alone.
    """
    bindings = getattr(dispatch_engine, "failure_terminals", None)
    if not isinstance(bindings, ModelFailureTerminalBindings) or bindings.is_frozen:
        return
    if len(failure_topics) != 1:
        return
    for command_topic in contract.event_bus.subscribe_topics:
        bindings.bind(command_topic=command_topic, failure_topic=failure_topics[0])
