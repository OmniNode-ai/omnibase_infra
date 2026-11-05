# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dispatch terminal node's answer: the caller's result and the run's step messages.

OMN-19972. ``result`` is the flattened delegate-skill shape the dispatch port
returns to its caller. ``run_step_messages`` are the run's numbered step events,
each a serialized event envelope, in step order, for ``run_step_topic``; the
port sends them, because a handler holds no event bus. Both are empty when no
steps were asked for or none can be sent.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelDelegationDispatchTerminalResult(BaseModel):
    """The caller's flattened result and the run-step messages to send."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result: dict[str, JsonType]
    run_step_topic: str | None = None
    run_step_messages: tuple[bytes, ...] = Field(default_factory=tuple)


__all__ = ["ModelDelegationDispatchTerminalResult"]
