# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Injected command topic -> declared failure topic map (OMN-26026)."""

from __future__ import annotations

import threading
from uuid import UUID

from pydantic import BaseModel

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.enums import EnumDlqFailureClass
from omnibase_infra.models.dispatch.model_malformed_command_terminal import (
    ModelMalformedCommandTerminal,
)
from omnibase_infra.models.dispatch.model_no_dispatcher_refusal import (
    ModelNoDispatcherRefusal,
)

__all__ = ["ModelFailureTerminalBindings"]


def _customer_reason(refusal: ModelNoDispatcherRefusal) -> str:
    """The refused field's own message, never the dispatch trace (OMN-17397)."""
    detail = (refusal.validation_detail or "").removeprefix("ValueError: ").strip()
    return detail or "command payload failed validation"


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
                workflow_id=None if raw is None else str(raw),
                failure_code=refusal.error_code.value,
                failure_reason=_customer_reason(refusal),
                origin_topic=topic,
            )
        ]
