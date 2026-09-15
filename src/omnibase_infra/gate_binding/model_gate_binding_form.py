# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One accepted spelling of a ``Gate:`` binding value."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.gate_binding.enum_gate_binding_probe import EnumGateBindingProbe


class ModelGateBindingForm(BaseModel):
    """A declared form: how it is spelled, and what it resolves to."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    pattern: str
    probe: EnumGateBindingProbe
