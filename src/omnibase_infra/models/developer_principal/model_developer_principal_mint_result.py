# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of the developer principal mint orchestrator (OMN-20823)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelDeveloperPrincipalMintResult(BaseModel):
    """What one mint run did or found. It has no password field.

    ``status`` is what the local runtime classifies the run by: a ``failure``
    names the step that stopped it (``failed_step``) and which mutations had
    completed (``acls_applied``, ``password_set``). An audit reports
    ``action="audited"``, whether the principal exists, and every ACL outside
    the plan (``extra_acls``) and every planned ACL the broker lacks
    (``missing_acls``), as binding labels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    status: Literal["success", "failure"] = Field(..., description="Outcome.")
    mode: Literal["converge", "audit"] = Field(..., description="The request's mode.")
    principal: str = Field(default="", description="The planned principal.")
    action: Literal["create", "rotate", "refused", "audited", ""] = Field(
        default="", description="What converge did, or audited."
    )
    exists: bool = Field(default=False, description="The broker lists the principal.")
    verified: bool = Field(default=False, description="Verify passed (converge).")
    acls_applied: bool = Field(
        default=False, description="Every planned ACL read back."
    )
    password_set: bool = Field(default=False, description="The password was set.")
    grant_count: int = Field(default=0, ge=0, description="Planned ACL bindings.")
    declaration_digest: str = Field(default="", description="The plan's digest.")
    extra_acls: tuple[str, ...] = Field(
        default_factory=tuple, description="ACLs outside the plan, as labels."
    )
    missing_acls: tuple[str, ...] = Field(
        default_factory=tuple, description="Planned ACLs the broker lacks (audit)."
    )
    failed_step: Literal["", "list", "plan", "describe", "apply", "verify"] = Field(
        default="", description="The step that stopped a failed run."
    )
    reason: str = Field(default="", description="Why the run failed or was refused.")
    correlation_id: UUID | None = Field(default=None, description="The request's.")


__all__: list[str] = ["ModelDeveloperPrincipalMintResult"]
