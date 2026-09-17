# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A delegation terminal as it arrives off the bus, inside its envelope (OMN-18569).

.. versionadded:: OMN-18569
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.cli.model_delegate_terminal import ModelDelegateTerminal

__all__ = ["ModelDelegateTerminalEnvelope"]


class ModelDelegateTerminalEnvelope(BaseModel):
    """The envelope carrier: a delegation terminal one level down, under ``payload``.

    Only the two fields that make this an envelope carrier are declared. The
    rest of ``ModelEventEnvelope`` -- routing, trace, tenancy -- is deliberately
    not mirrored: this model exists to say WHERE the terminal is, not to
    re-declare the envelope contract.

    Both declared fields are REQUIRED, and that is what makes carrier selection
    a typed question rather than a guess: a bare terminal has neither, so it
    cannot validate as an envelope, and an envelope has no top-level
    ``attempts``, so it cannot validate as a bare terminal.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    envelope_id: UUID = Field(...)
    payload: ModelDelegateTerminal = Field(...)
