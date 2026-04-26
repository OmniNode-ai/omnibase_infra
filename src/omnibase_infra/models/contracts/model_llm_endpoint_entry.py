# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Single LLM inference endpoint slot model from contracts/llm_endpoints.yaml."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.models.contracts.enum_llm_endpoint_status import (
    EnumLlmEndpointStatus,
)


class ModelLlmEndpointEntry(BaseModel):
    """Single LLM inference endpoint slot from llm_endpoints.yaml.

    Fields mirror the STABLE-CANONICAL group defined in the contract header.
    OPERATOR-ANNOTATION fields (hardware, context_window_budgeted, etc.) are
    included as optional strings/ints for round-trip fidelity but are not
    load-bearing for routing logic.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    slot_id: str = Field(..., description="Slot identifier.")  # ONEX_EXCLUDE: pattern_validator - slot_id is a human-readable natural key, not a UUID  # fmt: skip
    host: str | None = Field(
        default=None,
        description="IP address; non-null when status=running.",
    )
    port: int | None = Field(
        default=None,
        description="Integer port; non-null when status=running.",
    )
    endpoint_url: str | None = Field(
        default=None,
        description="Resolved URL convenience field; non-null when status=running.",
    )
    url_env_var: str | None = Field(
        default=None,
        description="Model-named env var (legacy canonical read). Nullable for Docker-internal slots.",
    )
    role_env_alias: str | None = Field(
        default=None,
        description="Role-named env var alias. Nullable when not yet assigned.",
    )
    model_hf_id: str | None = Field(
        default=None,
        description="Exact HuggingFace model ID; non-null when status=running.",
    )
    role: str = Field(
        ..., description="Role taxonomy value (closed set in contract header)."
    )
    status: EnumLlmEndpointStatus = Field(
        ..., description="running | disabled | on_demand | planned."
    )
    hardware: str | None = Field(default=None, description="Human host description.")
    context_window_budgeted: int | None = Field(
        default=None, description="Current max-tokens cap."
    )
    launchd_unit_or_none: str | None = Field(
        default=None, description="launchd label or null."
    )
    notes: str = Field(default="", description="Free-text operator notes.")

    @model_validator(mode="after")
    def _validate_running_required_fields(self) -> ModelLlmEndpointEntry:
        if self.status == EnumLlmEndpointStatus.RUNNING:
            missing = [
                name
                for name, value in (
                    ("host", self.host),
                    ("port", self.port),
                    ("endpoint_url", self.endpoint_url),
                    ("model_hf_id", self.model_hf_id),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "running endpoints require non-null fields: " + ", ".join(missing)
                )
        return self


__all__ = ["ModelLlmEndpointEntry"]
