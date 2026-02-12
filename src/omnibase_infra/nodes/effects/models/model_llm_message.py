# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM chat message model for multi-turn conversations.

This module provides ModelLlmMessage, representing a single message in an LLM
chat conversation. Role-based invariants enforce correct field combinations
for each message type.

Related:
    - ModelLlmToolCall: Tool calls embedded in assistant messages
    - ModelLlmInferenceRequest: Request model that contains messages
    - OMN-2105: Phase 5 LLM inference request model
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.nodes.effects.models.model_llm_tool_call import ModelLlmToolCall


class ModelLlmMessage(BaseModel):
    """A single chat message in a multi-turn LLM conversation.

    Each message has a ``role`` that determines which fields are valid.
    A model validator enforces these invariants at construction time.

    Role invariants:
        - ``"system"`` / ``"user"``: ``tool_calls`` must be empty,
          ``tool_call_id`` must be None.
        - ``"assistant"``: ``tool_call_id`` must be None; ``content``
          and/or ``tool_calls`` are allowed.
        - ``"tool"``: ``tool_call_id`` must be set, ``tool_calls`` must
          be empty.

    Attributes:
        role: Message role identifier.
        content: Text content of the message.
        tool_calls: Tool calls requested by the assistant.
        tool_call_id: ID of the tool call this message responds to.

    Example:
        >>> msg = ModelLlmMessage(role="user", content="Hello!")
        >>> msg.role
        'user'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ...,
        description="Message role identifier.",
    )
    content: str | None = Field(
        default=None,
        description="Text content of the message.",
    )
    tool_calls: tuple[ModelLlmToolCall, ...] = Field(
        default_factory=tuple,
        description="Tool calls requested by the assistant.",
    )
    tool_call_id: str | None = Field(
        default=None,
        description="ID of the tool call this message responds to.",
    )

    @model_validator(mode="after")
    def _validate_role_field_invariants(self) -> ModelLlmMessage:
        """Enforce field constraints based on message role."""
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("tool_call_id is required when role is 'tool'.")
            if self.tool_calls:
                raise ValueError("tool_calls must be empty when role is 'tool'.")
        elif self.role in ("system", "user"):
            if self.content is None or not self.content.strip():
                raise ValueError(f"content must be non-empty for {self.role} messages.")
            if self.tool_calls:
                raise ValueError(
                    f"tool_calls must be empty when role is '{self.role}'."
                )
            if self.tool_call_id is not None:
                raise ValueError(
                    f"tool_call_id must be None when role is '{self.role}'."
                )
        elif self.role == "assistant":
            if self.tool_call_id is not None:
                raise ValueError("tool_call_id must be None when role is 'assistant'.")
            if self.content is None and not self.tool_calls:
                raise ValueError(
                    "assistant messages must have content or tool_calls (or both)."
                )
        return self


__all__ = ["ModelLlmMessage"]
