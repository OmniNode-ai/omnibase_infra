# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Single LLM inference endpoint slot model from contracts/llm_endpoints.yaml."""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    @field_validator("slot_id", "role")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("role")
    @classmethod
    def _validate_role_taxonomy(cls, value: str) -> str:
        allowed = {
            "coder_slow",
            "coder_fast",
            "embedding",
            "reasoning",
            "reasoning_fast",
            "reasoning_lightweight",
            "reasoning_transient",
            "vision",
            "stt",
            "tts",
            "reranker",
        }
        if value not in allowed:
            raise ValueError(f"role must be one of: {', '.join(sorted(allowed))}")
        return value

    @field_validator("host")
    @classmethod
    def _validate_host_ip(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("host must be a non-empty IP address when set")
        try:
            ip_address(value)
        except ValueError as exc:
            raise ValueError("host must be an IP address, not a hostname") from exc
        return value

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: int | None) -> int | None:
        if value is not None and not 1 <= value <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return value

    @field_validator("endpoint_url")
    @classmethod
    def _validate_endpoint_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("endpoint_url must be an http(s) URL")
        if (
            parsed.path not in {"", "/"}
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "endpoint_url must be a base URL without path/query/fragment"
            )
        return value.rstrip("/")

    @field_validator("url_env_var", "role_env_alias")
    @classmethod
    def _validate_endpoint_env_var(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith("LLM_") or not value.endswith("_URL"):
            raise ValueError("endpoint env vars must use LLM_*_URL naming")
        return value

    @field_validator("model_hf_id", "hardware", "launchd_unit_or_none")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must be a non-empty string when set")
        return value

    @field_validator("context_window_budgeted")
    @classmethod
    def _validate_context_window_budget(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("context_window_budgeted must be positive when set")
        return value

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
        if self.endpoint_url is not None:
            parsed = urlparse(self.endpoint_url)
            if self.host is not None and parsed.hostname != self.host:
                raise ValueError("endpoint_url host must match host")
            if self.port is not None and parsed.port != self.port:
                raise ValueError("endpoint_url port must match port")
        return self


__all__ = ["ModelLlmEndpointEntry"]
