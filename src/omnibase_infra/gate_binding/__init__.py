# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The one declared grammar for a Linear description's ``Gate:`` key (OMN-18414).

The declaration itself is ``omnibase_infra/contracts/gate_binding_grammar.json``.
This package only reads it. omniclaude's ticket-creation admission guard reads
the same contract with the standard library alone, and a drift test on that side
fails when its transcription diverges.
"""

from omnibase_infra.gate_binding.enum_gate_binding_probe import EnumGateBindingProbe
from omnibase_infra.gate_binding.grammar import (
    declared_form_summary,
    gate_binding_line,
    load_gate_binding_grammar,
    normalize_gate_binding,
    resolve_gate_binding,
)
from omnibase_infra.gate_binding.model_gate_binding import ModelGateBinding
from omnibase_infra.gate_binding.model_gate_binding_form import ModelGateBindingForm
from omnibase_infra.gate_binding.model_gate_binding_grammar import (
    ModelGateBindingGrammar,
)
from omnibase_infra.gate_binding.model_gate_binding_normalization import (
    ModelGateBindingNormalization,
)

__all__ = [
    "EnumGateBindingProbe",
    "ModelGateBinding",
    "ModelGateBindingForm",
    "ModelGateBindingGrammar",
    "ModelGateBindingNormalization",
    "declared_form_summary",
    "gate_binding_line",
    "load_gate_binding_grammar",
    "normalize_gate_binding",
    "resolve_gate_binding",
]
