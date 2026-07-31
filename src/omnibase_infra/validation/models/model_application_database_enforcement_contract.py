# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed activation contract for application database domain gates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from omnibase_infra.validation.application_database_red_control_registry import (
    APPLICATION_DATABASE_RED_CONTROL_REGISTRY,
)
from omnibase_infra.validation.enums.enum_application_database_enforcement_gate import (
    EnumApplicationDatabaseEnforcementGate,
)
from omnibase_infra.validation.models.model_application_database_enforcement_gate_state import (
    ModelApplicationDatabaseEnforcementGateState,
)

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PIN_KEY = re.compile(r"^[a-z0-9_]+#[0-9]+$")


class ModelApplicationDatabaseEnforcementContract(BaseModel):
    """Complete gate family, immutable source pins, and activation truth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    ticket: Literal["OMN-15361"]
    predecessor_pins: Mapping[str, str] = Field(..., min_length=1)
    gates: Mapping[
        EnumApplicationDatabaseEnforcementGate,
        ModelApplicationDatabaseEnforcementGateState,
    ] = Field(..., min_length=1)

    @field_validator("predecessor_pins")
    @classmethod
    def validate_predecessor_pins(cls, pins: Mapping[str, str]) -> Mapping[str, str]:
        """Require immutable full Git revisions with unambiguous PR identities."""
        invalid_keys = sorted(key for key in pins if _PIN_KEY.fullmatch(key) is None)
        invalid_revisions = sorted(
            revision
            for revision in pins.values()
            if _REVISION.fullmatch(revision) is None
        )
        if invalid_keys:
            raise ValueError(f"invalid predecessor pin keys: {invalid_keys}")
        if invalid_revisions:
            raise ValueError(
                "predecessor pins must use full lowercase commit revisions"
            )
        if len(set(pins.values())) != len(pins):
            raise ValueError(
                "predecessor revisions must identify distinct source heads"
            )
        return MappingProxyType(dict(pins))

    @field_validator("gates")
    @classmethod
    def freeze_gates(
        cls,
        gates: Mapping[
            EnumApplicationDatabaseEnforcementGate,
            ModelApplicationDatabaseEnforcementGateState,
        ],
    ) -> Mapping[
        EnumApplicationDatabaseEnforcementGate,
        ModelApplicationDatabaseEnforcementGateState,
    ]:
        """Freeze the complete validated gate family against in-place mutation."""
        return MappingProxyType(dict(gates))

    @field_serializer("predecessor_pins")
    def serialize_predecessor_pins(self, pins: Mapping[str, str]) -> dict[str, str]:
        """Restore the YAML/JSON wire shape for the immutable mapping."""
        return dict(pins)

    @field_serializer("gates")
    def serialize_gates(
        self,
        gates: Mapping[
            EnumApplicationDatabaseEnforcementGate,
            ModelApplicationDatabaseEnforcementGateState,
        ],
    ) -> dict[
        EnumApplicationDatabaseEnforcementGate,
        ModelApplicationDatabaseEnforcementGateState,
    ]:
        """Restore the YAML/JSON wire shape for the immutable mapping."""
        return dict(gates)

    @model_validator(mode="after")
    def validate_complete_gate_set(self) -> ModelApplicationDatabaseEnforcementContract:
        """Reject omitted gates and RED claims without executable test bindings."""
        expected = set(EnumApplicationDatabaseEnforcementGate)
        actual = set(self.gates)
        if actual != expected:
            missing = sorted(gate.value for gate in expected - actual)
            extra = sorted(str(gate) for gate in actual - expected)
            raise ValueError(
                f"application database enforcement gate set drift: "
                f"missing={missing}, extra={extra}"
            )
        for gate, state in self.gates.items():
            declared_controls = set(state.seeded_red_controls)
            registered_controls = set(APPLICATION_DATABASE_RED_CONTROL_REGISTRY[gate])
            if declared_controls != registered_controls:
                missing = sorted(registered_controls - declared_controls)
                extra = sorted(declared_controls - registered_controls)
                raise ValueError(
                    "application database RED control registry drift: "
                    f"gate={gate.value}, missing={missing}, extra={extra}"
                )
        return self


__all__ = ["ModelApplicationDatabaseEnforcementContract"]
