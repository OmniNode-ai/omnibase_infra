# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""What ``onex delegate`` prints for a person by default (OMN-20124).

.. versionadded:: OMN-20124
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelDelegateHumanOutcome"]


class ModelDelegateHumanOutcome(BaseModel):
    """The default, non-JSON rendering of one delegation receipt.

    ``stdout`` is the answer text and nothing else, empty on a failure.
    ``stderr`` is one line: the receipt summary on success, the failure line on
    failure.
    """

    model_config = ConfigDict(frozen=True)

    succeeded: bool = Field(...)
    stdout: str = Field(default="")
    stderr: tuple[str, ...] = Field(default=())
