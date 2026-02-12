# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM tool choice model for caller constraints on tool selection.

This module provides ModelLlmToolChoice, a structured model that controls
how the LLM should use available tools. Uses a mode discriminator rather
than a plain string to preserve expressiveness across providers.

Related:
    - ModelLlmToolDefinition: Tool definitions constrained by this choice
    - OMN-2103: Phase 3 shared LLM models
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelLlmToolChoice(BaseModel):
    """Caller constraint on how the model should use tools.

    Modes:
        - ``"auto"``     - model decides whether to call a tool
        - ``"none"``     - model must NOT call any tool
        - ``"required"`` - model MUST call at least one tool
        - ``"function"`` - model MUST call the specific function named
          in ``function_name``

    Attributes:
        mode: Selection behaviour.
        function_name: Required when ``mode="function"``, forbidden otherwise.

    Example:
        >>> choice = ModelLlmToolChoice(mode="auto")
        >>> choice.mode
        'auto'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    mode: Literal["auto", "none", "required", "function"] = Field(
        ...,
        description="Tool selection behaviour.",
    )
    function_name: str | None = Field(
        default=None,
        min_length=1,
        description="Required when mode='function', forbidden otherwise.",
    )

    @model_validator(mode="after")
    def _validate_function_name_consistency(self) -> ModelLlmToolChoice:
        """Ensure function_name is present iff mode is 'function'."""
        if self.mode == "function" and self.function_name is None:
            raise ValueError("function_name is required when mode is 'function'.")
        if self.mode != "function" and self.function_name is not None:
            raise ValueError("function_name must be None when mode is not 'function'.")
        return self


__all__ = ["ModelLlmToolChoice"]
