# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared runtime timeouts (OMN-21719).

The values live in the ``timeouts`` block of ``runtime_timeouts.yaml`` beside
this module. The model has no field defaults, so a block that omits a value
fails to load rather than falling back to a literal.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

_TIMEOUTS_PATH = Path(__file__).resolve().parent / "runtime_timeouts.yaml"


class ModelRuntimeTimeouts(BaseModel):
    """Waits and bounds that the runtime call sites apply.

    Attributes:
        capability_probe_ttl_seconds: Lifetime of a capability probe result.
        capability_probe_seconds: Wait of one socket, Kafka-host or HTTP probe.
        health_check_min_seconds: Lower bound of a handler health-check wait.
        health_check_max_seconds: Upper bound of a handler health-check wait.
        health_check_default_seconds: Default handler health-check wait.
        handler_shutdown_min_seconds: Lower bound of a handler shutdown wait.
        handler_shutdown_max_seconds: Upper bound of a handler shutdown wait.
        handler_shutdown_default_seconds: Default handler shutdown wait.
        kafka_session_timeout_ms: Consumer group session timeout.
        kafka_heartbeat_interval_ms: Consumer group heartbeat interval.
        kafka_max_poll_interval_ms: Consumer group max poll interval.
        snapshot_consumer_timeout_ms: Consumer wait of the snapshot publisher.
        circuit_breaker_reset_seconds: Wait before a circuit breaker resets.
        health_probe_seconds: Wait of the node invocation runtime health probe.
        local_dispatch_seconds: Wait for a whole local node dispatch.
        terminal_seconds: Wait for the terminal event after
            a local dispatch returns.
        llm_transport_max_seconds: Ceiling the LLM HTTP transport accepts.
        llm_post_seconds: Wait of one LLM HTTP POST.
        introspection_base_backoff_seconds: Base of the listener retry backoff.
        introspection_heartbeat_interval_seconds: Period of the heartbeat task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    capability_probe_ttl_seconds: int = Field(gt=0)
    capability_probe_seconds: float = Field(gt=0)
    health_check_min_seconds: float = Field(gt=0)
    health_check_max_seconds: float = Field(gt=0)
    health_check_default_seconds: float = Field(gt=0)
    handler_shutdown_min_seconds: float = Field(gt=0)
    handler_shutdown_max_seconds: float = Field(gt=0)
    handler_shutdown_default_seconds: float = Field(gt=0)
    kafka_session_timeout_ms: int = Field(gt=0)
    kafka_heartbeat_interval_ms: int = Field(gt=0)
    kafka_max_poll_interval_ms: int = Field(gt=0)
    snapshot_consumer_timeout_ms: int = Field(gt=0)
    circuit_breaker_reset_seconds: float = Field(gt=0)
    health_probe_seconds: float = Field(gt=0)
    local_dispatch_seconds: int = Field(gt=0)
    terminal_seconds: float = Field(gt=0)
    llm_transport_max_seconds: float = Field(gt=0)
    llm_post_seconds: float = Field(gt=0)
    introspection_base_backoff_seconds: float = Field(gt=0)
    introspection_heartbeat_interval_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def _defaults_lie_within_bounds(self) -> ModelRuntimeTimeouts:
        """Refuse a default outside its bounds and a heartbeat that outlasts the session."""
        if not (
            self.health_check_min_seconds
            <= self.health_check_default_seconds
            <= self.health_check_max_seconds
        ):
            raise ValueError("health_check_default_seconds is outside its bounds")
        if not (
            self.handler_shutdown_min_seconds
            <= self.handler_shutdown_default_seconds
            <= self.handler_shutdown_max_seconds
        ):
            raise ValueError("handler_shutdown_default_seconds is outside its bounds")
        if self.kafka_heartbeat_interval_ms >= self.kafka_session_timeout_ms:
            raise ValueError(
                "kafka_heartbeat_interval_ms must be below kafka_session_timeout_ms"
            )
        return self

    @classmethod
    def from_file(cls, timeouts_path: Path) -> ModelRuntimeTimeouts:
        """Load the timeouts from the ``timeouts`` mapping of a YAML file.

        Raises:
            ProtocolConfigurationError: If the file has no ``timeouts`` mapping.
        """
        with timeouts_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        block = document.get("timeouts") if isinstance(document, dict) else None
        if not isinstance(block, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_runtime_timeouts",
            )
            raise ProtocolConfigurationError(
                f"{timeouts_path.name} has no timeouts mapping",
                context=context,
            )
        return cls.model_validate(block)


@cache
def load_runtime_timeouts() -> ModelRuntimeTimeouts:
    """Return the timeouts declared beside this module."""
    return ModelRuntimeTimeouts.from_file(_TIMEOUTS_PATH)


__all__: list[str] = ["ModelRuntimeTimeouts", "load_runtime_timeouts"]
