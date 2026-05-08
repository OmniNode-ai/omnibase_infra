# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Pydantic models for bifrost_delegation.yaml.

Validates the delegation routing config that maps Claude Code task classes
to bifrost backend policies. Every response from the gateway must include
the matched rule_id and config_version for audit provenance (OMN-10637).

Related:
    - OMN-10637: Bifrost routing rules for delegation task classes
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelDelegationFallbackPolicy(BaseModel):
    """Fallback behavior when all backends in a rule are exhausted."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    action: str = Field(
        ...,
        description="Action on backend failure: 'escalate_to_next_tier' or 'return_error'.",
    )
    max_retries: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Maximum retry attempts across backends in this rule.",
    )
    on_exhaust: str = Field(
        default="return_error",
        description="Behavior when all retries are exhausted.",
    )


class ModelDelegationRoutingRule(BaseModel):
    """A single delegation routing rule mapping a task class to backends.

    The rule_id and config_version from the parent config are included
    in every Bifrost response for audit provenance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    rule_id: UUID = Field(
        ...,
        description="Stable UUID for audit logging and provenance.",
    )
    priority: int = Field(
        default=100,
        ge=0,
        description="Evaluation order — lower values evaluated first.",
    )
    task_class: str = Field(
        ...,
        description="Task class this rule targets (e.g. 'code_generation').",
    )
    task_class_contract_version: str = Field(
        ...,
        description="Version of the task class contract this rule was authored against.",
    )
    backend_policy_version: str = Field(
        ...,
        description="Version of the backend policy applied by this rule.",
    )
    match_operation_types: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Operation types this rule matches (empty = any).",
    )
    match_capabilities: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Capabilities the request must declare (empty = any).",
    )
    latency_sla_ms: int | None = Field(
        default=None,
        ge=1,
        description="SLA latency constraint — request must complete within this window.",
    )
    cost_ceiling_usd_per_1k_tokens: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum allowed cost per 1K tokens for this rule's backends.",
    )
    backend_ids: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Ordered backend IDs to try when this rule matches.",
    )
    fallback_policy: ModelDelegationFallbackPolicy = Field(
        ...,
        description="Failover behavior when backends are exhausted.",
    )
    shadow_policy_id: UUID = Field(
        ...,
        description="Shadow policy UUID for A/B evaluation of this rule.",
    )


class ModelDelegationBackendConfig(BaseModel):
    """Backend entry in the delegation routing config."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # ONEX_EXCLUDE: pattern_validator - backend_id is a human-readable slug
    # (e.g. "local-qwen-coder-30b"), not a UUID entity reference. Config usability trumps UUID convention.
    backend_id: str = Field(
        ...,
        min_length=1,
        description="Stable human-readable slug for audit logging.",
    )
    base_url_env: str | None = Field(
        default=None,
        description="Env var name holding the backend base URL (local backends).",
    )
    endpoint_url: str = Field(
        default="",
        description="Deployed full endpoint URL populated by install-delegation.sh.",
    )
    # ONEX_EXCLUDE: pattern_validator - model_name is a model identifier string
    # (e.g. "claude-sonnet-4-6"), not an entity reference requiring ID + display_name pattern.
    model_name: str = Field(
        ...,
        description="Model identifier sent in outbound requests.",
    )
    tier: str = Field(
        ...,
        description="Routing tier: 'local' or 'frontier_api'.",
    )
    timeout_ms: int = Field(
        default=30000,
        ge=100,
        le=600_000,
        description="Per-backend HTTP timeout in milliseconds.",
    )
    capabilities: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Capabilities this backend supports.",
    )


class ModelDelegationCircuitBreakerConfig(BaseModel):
    """Circuit breaker settings applying to all backends."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Consecutive failures that open the circuit.",
    )
    window_seconds: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Cooldown duration after circuit opens, in seconds.",
    )


class ModelDelegationFailoverConfig(BaseModel):
    """Gateway-level failover settings."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum backend attempts per request.",
    )
    backoff_base_ms: int = Field(
        default=500,
        ge=0,
        le=10_000,
        description="Base exponential backoff delay in milliseconds.",
    )


class ModelBifrostDelegationConfig(BaseModel):
    """Root model for bifrost_delegation.yaml.

    Validates the complete delegation routing config. The config_version
    field is recorded in every Bifrost response for audit provenance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    config_version: str = Field(
        ...,
        description="Semver config version — recorded in every gateway response.",
    )
    schema_version: str = Field(
        ...,
        description="Schema identifier for this config format.",
    )
    backends: tuple[ModelDelegationBackendConfig, ...] = Field(
        ...,
        min_length=1,
        description="Backend definitions with deployed endpoint URLs.",
    )
    routing_rules: tuple[ModelDelegationRoutingRule, ...] = Field(
        ...,
        min_length=1,
        description="Routing rules evaluated in ascending priority order.",
    )
    default_backends: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Fallback backend IDs when no routing rule matches.",
    )
    circuit_breaker: ModelDelegationCircuitBreakerConfig = Field(
        default_factory=ModelDelegationCircuitBreakerConfig,
        description="Circuit breaker settings applying to all backends.",
    )
    failover: ModelDelegationFailoverConfig = Field(
        default_factory=ModelDelegationFailoverConfig,
        description="Gateway-level failover settings.",
    )


__all__: list[str] = [
    "ModelBifrostDelegationConfig",
    "ModelDelegationBackendConfig",
    "ModelDelegationCircuitBreakerConfig",
    "ModelDelegationFailoverConfig",
    "ModelDelegationFallbackPolicy",
    "ModelDelegationRoutingRule",
]
