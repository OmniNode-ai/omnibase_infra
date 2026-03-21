# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Baseline configuration model for savings estimation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSavingsBaselineConfig(BaseModel):
    """Baseline configuration for counterfactual cost estimation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ONEX_EXCLUDE: pattern_validator - reference_model_id is a human-readable model name, not a UUID
    reference_model_id: str = "claude-opus-4-6"
    regen_multiplier: float = 3.0
    # ONEX_EXCLUDE: any_type - dict[str, int] needed for severity->token mapping
    tokens_per_fix_cycle_by_severity: dict[str, int] = Field(
        default_factory=lambda: {
            "error_pre_commit": 15000,
            "error_ci": 30000,
            "error_poly_enforcer": 10000,
            "error_code_review": 25000,
            "warning_pre_commit": 3000,
            "warning_code_review": 5000,
        }
    )
    avg_tokens_per_subagent_call: int = 20000
    rag_regen_multiplier: float = 2.0
    baseline_version: str = "v1.0"
    pricing_manifest_version: str = ""


__all__: list[str] = ["ModelSavingsBaselineConfig"]
