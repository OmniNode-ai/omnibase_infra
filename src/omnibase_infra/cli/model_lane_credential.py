# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One machine's bus identity for one lane, resolved by reference (OMN-18432).

The ``.201`` dev-lane broker has required SASL/SCRAM since OMN-18012 phase B.
The sanctioned client store this machine already carries (OMN-15922) models
exactly one credential -- the gateway one -- so there was nowhere to put a BUS
identity except the ambient ``KAFKA_SASL_*`` environment, which is the surface
the by-reference rule exists to remove. This model is the resolved answer for
one lane.

``sasl_password_ref`` travels with the value deliberately. Every message this
credential appears in names the reference rather than the value, and a
resolved credential that has forgotten which reference produced it cannot say
what an operator would have to fix.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr

__all__ = ["ModelLaneCredential"]


class ModelLaneCredential(BaseModel):
    """A lane's SASL identity as this machine holds it."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    lane: str = Field(description="Lane id this identity authenticates to")
    sasl_username: str = Field(
        description="SASL principal name; a name, never a secret, so it is "
        "stored in the reference-only config file beside the reference"
    )
    sasl_password_ref: str = Field(
        description="Key the value is filed under in the 0600 credential file"
    )
    sasl_password: SecretStr = Field(
        description="The resolved value, wrapped so it cannot reach a log line "
        "or a traceback through an ordinary repr"
    )
