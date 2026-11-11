# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A consume-boundary failure terminal as the delegate CLI reads it (OMN-22216).

Read-side mirror of
:class:`~omnibase_infra.runtime.boundary_failure_terminal.ModelBoundaryFailureTerminal`,
the payload a deployed lane's consume boundary publishes when it fails a
delegate request for good (for example with a ``DatabaseError``). Only the
fields the CLI reports are declared, and ``extra="ignore"``, so a field the
runtime adds can never stop the failure reaching the caller.

``status`` is pinned to ``failed`` and ``failure_class`` and ``failure_reason``
are required: those three together are what identify this payload, and none
of them is a required field of a delegation terminal, whose own identifying
field is ``attempts``.

.. versionadded:: OMN-22216
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.cli.model_delegate_terminal import ModelDelegateTerminal

__all__ = ["ModelDelegateBoundaryFailureTerminal"]


class ModelDelegateBoundaryFailureTerminal(BaseModel):
    """The boundary's own failure: its class, its ONEX code and its text."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    status: Literal["failed"] = Field(...)
    failure_class: str = Field(..., min_length=1)
    failure_reason: str = Field(...)
    failure_code: str | None = Field(default=None)
    retryable: bool | None = Field(default=None)

    @property
    def cause(self) -> str:
        """The failure class, followed by its ONEX code when the boundary had one."""
        if self.failure_code:
            return f"{self.failure_class} {self.failure_code}"
        return self.failure_class

    def as_delegate_terminal(self) -> ModelDelegateTerminal:
        """This run's terminal: ``failed``, no attempts, the boundary's cause and text.

        No rung is invented: the boundary failed the request, so the terminal
        carries no attempts and its cause and reason are the boundary's own.
        """
        return ModelDelegateTerminal(
            attempts=(),
            status="failed",
            terminal_failure_cause=self.cause,
            terminal_failure_reason=self.failure_reason,
            error_message=self.failure_reason,
        )
