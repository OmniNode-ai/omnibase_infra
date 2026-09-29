# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed diagnosis returned by delegation doctor checks."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.doctor.enum_delegation_doctor_fault import (
    EnumDelegationDoctorFault,
)


class ModelDelegationDiagnosis(BaseModel):
    """One named delegation fault, or a fault-free diagnostic detail."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fault: EnumDelegationDoctorFault | None
    detail: str = Field(min_length=1)
    fix: str = ""

    @model_validator(mode="after")
    def _fault_requires_fix(self) -> Self:
        if self.fault is not None and not self.fix.strip():
            raise ValueError("a delegation fault requires one concrete fix")
        return self


__all__ = ["ModelDelegationDiagnosis"]
