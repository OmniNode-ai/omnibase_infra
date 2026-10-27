# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to plan one developer machine's broker principal (OMN-20821)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDeveloperPrincipalPlanRequest(BaseModel):
    """Everything the plan reads, supplied by the caller so the handler does no I/O.

    ``declaration_yaml`` is the grants declaration's exact text, so its digest is
    the digest of the file the caller read. ``derived_documents`` maps each file
    name the declaration names in ``grants_from`` or ``principal_grants_from`` to
    that file's exact JSON text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    declaration_yaml: str = Field(..., description="The grants declaration, verbatim.")
    derived_documents: dict[str, str] = Field(
        default_factory=dict,
        description="Derived grant documents the declaration names, by file name.",
    )
    login: str = Field(..., description="The developer's login (local@domain).")
    device: str = Field(..., description="The machine's device name.")
    existing_principals: tuple[str, ...] = Field(
        default=(),
        description="Principals the broker already holds; decides create, rotate and the device cap.",
    )


__all__: list[str] = ["ModelDeveloperPrincipalPlanRequest"]
