# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What a delegation that ended before dispatch tells the reader (OMN-19006).

``onex delegate`` exits before dispatch in a dozen places: an unknown flag, a
``--response-contract`` that is not JSON, a lane that does not resolve. Every
lane is told to read the outcome from ``.onex_state/runs/<run_id>/receipt.json``
and, on those exits, the instruction named a file that did not exist. The
``delegate-fanout`` workflow measured it: its ``NO_RECEIPT`` row is "no run
directory was created", and the only trace of the cause is the last line of a
stderr file.

This model is the ``refusal`` block of the receipt written for such an exit.
It states the cause in the words the command itself used and says what to do
about it, so the receipt is something a reader can act on and not a bare
classification.

.. versionadded:: OMN-19006
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_delegate_refusal_stage import EnumDelegateRefusalStage

__all__ = ["ModelDelegateRefusal"]


class ModelDelegateRefusal(BaseModel):
    """The cause of a delegation that ended without being dispatched."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    stage: EnumDelegateRefusalStage = Field(
        description="The stage the command ended at.",
    )
    error_type: str = Field(
        description="The class name of the error that ended the command.",
    )
    message: str = Field(
        description=(
            "The command's own words for the cause, sanitised: the text a "
            "terminal reader saw on stderr."
        ),
    )
    remedy: str = Field(
        description="What to change before running the command again.",
    )
    exit_code: int = Field(
        description="The process exit code the command ended with.",
    )
