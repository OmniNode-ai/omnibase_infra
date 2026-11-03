# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""How a delegation that ended with no terminal is classed, and what it knew (OMN-22211).

A wait that ends while the command is still queued behind the delegate-skill
consumer is not a lost command and not a slow run: the consumer had not reached
it. Before this model the receipt said ``timeout`` for it, with no attempt and no
rung, and the queue depth that told the two apart sat in a block a caller had to
know to open. ``queued_not_started`` is the class when the broker showed records
ahead of the command; a depth of zero (the caller's own record alone) and a depth
that could not be observed stay ``timeout``, because neither is evidence the
command was still queued.

Route attribution is untouched: a run with no terminal still names no rung.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.cli.model_delegate_queue_depth import ModelDelegateQueueDepth
from omnibase_infra.enums.enum_delegate_incomplete_class import (
    EnumDelegateIncompleteClass,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

__all__ = ["ModelDelegateIncompleteTerminal"]


class ModelDelegateIncompleteTerminal(BaseModel):
    """The class, reason and known facts of a receipt written with no terminal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    terminal_class: EnumDelegateIncompleteClass = Field(
        ..., description="How the run ended without a delegation terminal."
    )
    failure_reason: str = Field(
        ..., description="Why there is no terminal, sanitized for the receipt."
    )
    known: dict[str, object] = Field(
        default_factory=dict,
        description="What the invocation can state about the run, and no guess.",
    )

    @classmethod
    def from_hard_timeout(
        cls, *, error: str, queue_depth: ModelDelegateQueueDepth
    ) -> ModelDelegateIncompleteTerminal:
        """Class the CLI's own hard timeout by the depth observed when it fired."""
        depth = queue_depth.records_ahead
        if depth is None or depth == 0:
            return cls(
                terminal_class=EnumDelegateIncompleteClass.TIMEOUT,
                failure_reason=sanitize_error_string(error),
            )
        return cls(
            terminal_class=EnumDelegateIncompleteClass.QUEUED_NOT_STARTED,
            failure_reason=(
                "the wait ended while the command was still queued: "
                f"{depth} record(s) were ahead of it, uncommitted by "
                f"{queue_depth.consumer_group} (observed backlog "
                f"{queue_depth.backlog_records}); the consumer had not started it"
            ),
        )

    @classmethod
    def from_envelope(
        cls,
        envelope: dict[str, object],
        *,
        reason: str,
        queue_depth_observer: Callable[[], ModelDelegateQueueDepth] | None = None,
    ) -> ModelDelegateIncompleteTerminal:
        """Classify a delegation receipt that carries no terminal.

        ``timeout`` when the runtime's own workflow result says so, ``failed``
        for every other way of ending without a terminal payload. The runtime's
        error and type ride along so the receipt names the cause the capture log
        holds rather than only the resolver's "no resolvable terminal". A
        runtime ``timeout`` is probed for the command queue's depth when an
        observer is given, and the observed depth rides in ``known``.
        """
        result = envelope.get("result")
        summary = result if isinstance(result, dict) else {}
        workflow_result = str(summary.get("workflow_result") or "")
        known: dict[str, object] = {
            "workflow_result": workflow_result or None,
            "runtime_error_type": str(summary.get("runtime_error_type") or "") or None,
            "runtime_error": sanitize_error_string(str(summary.get("error") or ""))
            or None,
            "terminal_payload": summary.get("terminal_payload"),
            "wire_correlation_id": summary.get("wire_correlation_id") or None,
            "resolution_error": sanitize_error_string(reason),
        }
        if workflow_result != "timeout":
            return cls(
                terminal_class=EnumDelegateIncompleteClass.FAILED,
                failure_reason=sanitize_error_string(reason),
                known=known,
            )
        if queue_depth_observer is None:
            return cls(
                terminal_class=EnumDelegateIncompleteClass.TIMEOUT,
                failure_reason=sanitize_error_string(reason),
                known=known,
            )
        queue_depth = queue_depth_observer()
        classed = cls.from_hard_timeout(error=reason, queue_depth=queue_depth)
        return cls(
            terminal_class=classed.terminal_class,
            failure_reason=classed.failure_reason,
            known={**known, "queue_depth": queue_depth.model_dump(mode="json")},
        )
