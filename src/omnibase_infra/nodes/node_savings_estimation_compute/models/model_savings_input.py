# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Top-level input model for the savings estimation compute handler."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_savings_estimation_compute.models.model_delegation_signal import (
    ModelDelegationSignal,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_injection_signal import (
    ModelInjectionSignal,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_llm_call_record import (
    ModelLlmCallRecord,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_rag_signal import (
    ModelRagSignal,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_savings_baseline_config import (
    ModelSavingsBaselineConfig,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_validator_catch_signal import (
    ModelValidatorCatchSignal,
)


class ModelSavingsInput(BaseModel):
    """Input for the savings estimation compute handler."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ONEX_EXCLUDE: pattern_validator - session_id is a wire-format string, not a UUID
    session_id: str
    # ONEX_EXCLUDE: pattern_validator - correlation_id is a wire-format string, not a UUID
    correlation_id: str
    llm_calls: list[ModelLlmCallRecord] = Field(default_factory=list)
    treatment_group: str
    injection_signals: list[ModelInjectionSignal] = Field(default_factory=list)
    validator_catches: list[ModelValidatorCatchSignal] = Field(default_factory=list)
    delegation_signals: list[ModelDelegationSignal] = Field(default_factory=list)
    rag_signals: list[ModelRagSignal] = Field(default_factory=list)
    baseline_config: ModelSavingsBaselineConfig = Field(
        default_factory=ModelSavingsBaselineConfig
    )


__all__: list[str] = ["ModelSavingsInput"]
