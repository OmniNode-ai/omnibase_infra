# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lifecycle hook configuration model (OMN-7655)."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelLifecycleHookConfig(BaseModel):
    """Configuration for a single lifecycle hook callable.

    Represents a dotted-path reference to an async callable that will be
    invoked during the corresponding lifecycle phase. The callable receives
    a ModelAutoWiringContext and returns a ModelLifecycleHookResult.

    Attributes:
        callable_ref: Dotted import path to the async hook callable.
            Example: ``mypackage.hooks.on_start_handler``
        timeout_seconds: Maximum execution time before the hook is cancelled.
        required: If True, hook failure aborts the lifecycle phase.
            If False, failure is logged but does not block.
        idempotent: Assertion that this hook is safe to retry. Must be True.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    callable_ref: str = Field(
        ...,
        min_length=1,
        description="Dotted import path to the async hook callable",
    )
    timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=300.0,
        description="Maximum execution time in seconds",
    )
    required: bool = Field(
        default=True,
        description="Whether hook failure aborts the lifecycle phase",
    )
    idempotent: bool = Field(
        default=True,
        description="Assertion that this hook is safe to retry (must be True)",
    )

    @model_validator(mode="after")
    def validate_idempotent(self) -> Self:
        """Enforce that all lifecycle hooks declare themselves as idempotent."""
        if not self.idempotent:
            msg = (
                f"Lifecycle hook '{self.callable_ref}' must declare "
                "idempotent=True. Non-idempotent hooks are not permitted."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_callable_ref_format(self) -> Self:
        """Validate that callable_ref looks like a dotted Python import path."""
        parts = self.callable_ref.split(".")
        if len(parts) < 2:
            msg = (
                f"callable_ref '{self.callable_ref}' must be a dotted path "
                "with at least two segments (e.g., 'package.function')"
            )
            raise ValueError(msg)
        for part in parts:
            if not part.isidentifier():
                msg = (
                    f"callable_ref segment '{part}' in '{self.callable_ref}' "
                    "is not a valid Python identifier"
                )
                raise ValueError(msg)
        return self
