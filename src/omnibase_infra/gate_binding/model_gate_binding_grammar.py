# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The whole ``Gate:`` grammar declaration, as read from the contract file."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.gate_binding.model_gate_binding_fixtures import (
    ModelGateBindingFixtures,
)
from omnibase_infra.gate_binding.model_gate_binding_form import ModelGateBindingForm
from omnibase_infra.gate_binding.model_gate_binding_normalization import (
    ModelGateBindingNormalization,
)


class ModelGateBindingGrammar(BaseModel):
    """Typed shape of ``omnibase_infra/contracts/gate_binding_grammar.json``.

    The contract is the declaration; this model is only how this repository
    reads it. The other consumer -- omniclaude's ticket-creation admission
    guard -- reads the same file with the standard library alone, for the
    reason recorded in the contract's own ``$comment``.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    contract_version: str
    line_pattern_authoring: str
    line_pattern_reading: str
    normalizations: tuple[ModelGateBindingNormalization, ...]
    forms: tuple[ModelGateBindingForm, ...]
    fixtures: ModelGateBindingFixtures
