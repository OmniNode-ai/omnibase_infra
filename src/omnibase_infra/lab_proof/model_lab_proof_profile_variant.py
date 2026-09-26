# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""One way a repository's pull requests are proved.

Ticket: OMN-19565
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.lab_proof.enum_lab_proof_budget_basis import (
    EnumLabProofBudgetBasis,
)
from omnibase_infra.lab_proof.enum_lab_proof_check import EnumLabProofCheck
from omnibase_infra.lab_proof.enum_lab_proof_execution import EnumLabProofExecution
from omnibase_infra.lab_proof.enum_lab_proof_kind import (
    EnumLabProofKind,
)
from omnibase_infra.lab_proof.enum_lab_proof_profile_status import (
    EnumLabProofProfileStatus,
)
from omnibase_infra.lab_proof.model_lab_proof_foundation_target import (
    ModelLabProofFoundationTarget,
)
from omnibase_infra.lab_proof.model_lab_proof_host_selector import (
    ModelLabProofHostSelector,
)
from omnibase_infra.lab_proof.model_lab_proof_match import ModelLabProofMatch
from omnibase_infra.lab_proof.model_lab_proof_negative_control import (
    ModelLabProofNegativeControl,
)

# A step names a node and one operation its contract routes: no shell text.
STEP_REF_PATTERN = re.compile(r"^node_[a-z0-9_]+:[a-z0-9_.]+$")


class ModelLabProofProfileVariant(BaseModel):
    """One proof kind for one repository (plan section 2).

    The rules below keep a row internally consistent. Whether its steps name
    nodes that exist is a property of the repository, not the row, and is
    checked by ``validate_steps_against_repo``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    variant_key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    proof_kind: EnumLabProofKind
    status: EnumLabProofProfileStatus
    status_reason: str = Field(min_length=1)
    execution: EnumLabProofExecution
    recipe_ref: str | None = None
    match: ModelLabProofMatch | None = None
    host_selector: ModelLabProofHostSelector | None = None
    steps: tuple[str, ...] = ()
    mandatory_checks: tuple[EnumLabProofCheck, ...] = ()
    negative_control: ModelLabProofNegativeControl | None = None
    foundation: ModelLabProofFoundationTarget | None = None
    timeout_minutes: int | None = Field(default=None, ge=1)
    budget_minutes: int | None = Field(default=None, ge=1)
    budget_basis: EnumLabProofBudgetBasis | None = None
    budget_source: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ModelLabProofProfileVariant:
        kind, status, execution = self.proof_kind, self.status, self.execution
        exempt_kind = kind is EnumLabProofKind.EXEMPT
        exempt_status = status is EnumLabProofProfileStatus.EXEMPT
        if exempt_kind != exempt_status:
            raise ValueError(
                f"variant {self.variant_key}: proof_kind exempt and status exempt go "
                f"together (got {kind}, {status})"
            )
        runnable = status in (
            EnumLabProofProfileStatus.LIVE,
            EnumLabProofProfileStatus.PILOT,
        )
        if not runnable and execution is not EnumLabProofExecution.NONE:
            raise ValueError(
                f"variant {self.variant_key}: status {status} runs nothing, so "
                f"execution must be none (got {execution})"
            )
        if runnable:
            if execution is EnumLabProofExecution.NONE:
                raise ValueError(
                    f"variant {self.variant_key}: status {status} needs execution "
                    "node or manual_recipe"
                )
            missing = [
                name
                for name, value in (
                    ("match", self.match),
                    ("host_selector", self.host_selector),
                    ("negative_control", self.negative_control),
                    ("timeout_minutes", self.timeout_minutes),
                    ("budget_minutes", self.budget_minutes),
                    ("budget_basis", self.budget_basis),
                )
                if value is None
            ]
            if not self.mandatory_checks:
                missing.append("mandatory_checks")
            if missing:
                raise ValueError(
                    f"variant {self.variant_key}: a {status} variant needs "
                    + ", ".join(missing)
                )
        if execution is EnumLabProofExecution.NODE:
            if not self.steps:
                raise ValueError(
                    f"variant {self.variant_key}: execution node needs steps"
                )
            for step in self.steps:
                if not STEP_REF_PATTERN.match(step):
                    raise ValueError(
                        f"variant {self.variant_key}: step {step!r} is not "
                        "'<node_name>:<operation>'"
                    )
        elif self.steps:
            raise ValueError(
                f"variant {self.variant_key}: steps are only valid with execution node"
            )
        if execution is EnumLabProofExecution.MANUAL_RECIPE and not self.recipe_ref:
            raise ValueError(
                f"variant {self.variant_key}: execution manual_recipe needs recipe_ref"
            )
        if (kind is EnumLabProofKind.FOUNDATION_OVERRIDE) != (
            self.foundation is not None
        ):
            raise ValueError(
                f"variant {self.variant_key}: a foundation block goes with proof_kind "
                "foundation_override and only with it"
            )
        if (
            self.budget_basis is EnumLabProofBudgetBasis.MEASURED
            and not self.budget_source
        ):
            raise ValueError(
                f"variant {self.variant_key}: a measured budget needs budget_source "
                "(the ledger row it was measured in)"
            )
        if (
            self.budget_minutes is not None
            and self.timeout_minutes is not None
            and self.budget_minutes > self.timeout_minutes
        ):
            raise ValueError(
                f"variant {self.variant_key}: budget_minutes exceeds timeout_minutes"
            )
        return self


__all__ = ["STEP_REF_PATTERN", "ModelLabProofProfileVariant"]
