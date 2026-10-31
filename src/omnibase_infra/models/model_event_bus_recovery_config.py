# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-owned policy for runtime event bus recovery."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message


class ModelEventBusRecoveryConfig(BaseModel):
    """Bound each connect/attach attempt and cap the exponential retry delay."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    initial_backoff_seconds: float = Field(gt=0)
    max_backoff_seconds: float = Field(gt=0)
    backoff_multiplier: float = Field(ge=1)
    attempt_timeout_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> ModelEventBusRecoveryConfig:
        """Reject a cap smaller than the initial delay."""
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= initial_backoff_seconds")
        return self

    @classmethod
    def from_contract(cls) -> ModelEventBusRecoveryConfig:
        """Load the shipped policy; runtime overlays can replace this model."""
        resource = files("omnibase_infra").joinpath(
            "contracts/runtime_event_bus_recovery.yaml"
        )
        return cls.model_validate(yaml.safe_load(resource.read_text(encoding="utf-8")))

    @classmethod
    def from_runtime_config(
        cls, runtime_config: Mapping[str, object] | None
    ) -> ModelEventBusRecoveryConfig:
        """Read ``event_bus.recovery`` from the runtime config, else the contract.

        Raises:
            ProtocolConfigurationError: ``event_bus`` is not a mapping, or the
                ``recovery`` block is invalid.
        """
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="dynamic_contract_listener.recovery_config",
        )
        event_bus_config = (runtime_config or {}).get("event_bus", {})
        if not isinstance(event_bus_config, dict):
            raise ProtocolConfigurationError(
                "runtime config 'event_bus' must be a mapping", context=context
            )
        recovery_config = event_bus_config.get("recovery")
        try:
            if recovery_config is None:
                return cls.from_contract()
            return cls.model_validate(recovery_config)
        except ValidationError as exc:
            raise ProtocolConfigurationError(
                "invalid event_bus.recovery configuration for the dynamic "
                f"contract listener: {sanitize_error_message(exc)}",
                context=context,
            ) from exc


__all__ = ["ModelEventBusRecoveryConfig"]
