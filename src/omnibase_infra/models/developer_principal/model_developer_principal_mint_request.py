# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for the developer principal mint orchestrator (OMN-20823)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelDeveloperPrincipalMintRequest(BaseModel):
    """Mint (``converge``) or read (``audit``) one machine's broker principal.

    ``password_env`` is the NAME of an environment variable of the process that
    runs the node, never a value (plan D5): the request, the input file it is read
    from and the state root ``onex node`` writes carry no secret. ``converge``
    needs it; ``audit`` reads only and refuses one, so an audit can never set a
    password.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    mode: Literal["converge", "audit"] = Field(..., description="Mint or read only.")
    declaration_yaml: str = Field(
        ..., min_length=1, description="The grants declaration, verbatim."
    )
    derived_documents: dict[str, str] = Field(
        default_factory=dict,
        description="Derived grant documents the declaration names, by file name.",
    )
    login: str = Field(..., min_length=1, description="The machine's OS user.")
    device: str = Field(..., min_length=1, description="The machine's short host name.")
    admin_url: str = Field(..., min_length=1, description="The Redpanda Admin API URL.")
    password_env: str = Field(
        default="",
        pattern=r"^([A-Z][A-Z0-9_]*)?$",
        description="Name of the variable holding the new password (converge only).",
    )
    correlation_id: UUID | None = Field(
        default=None, description="Echoed on the result."
    )

    @model_validator(mode="after")
    def _password_env_matches_mode(self) -> ModelDeveloperPrincipalMintRequest:
        if self.mode == "converge" and not self.password_env:
            raise ValueError(
                "converge needs password_env, the name of the variable holding "
                "the new password"
            )
        if self.mode == "audit" and self.password_env:
            raise ValueError("audit reads only and takes no password_env")
        return self


__all__: list[str] = ["ModelDeveloperPrincipalMintRequest"]
