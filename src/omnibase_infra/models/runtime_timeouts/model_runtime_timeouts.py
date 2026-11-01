# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared runtime timeouts (OMN-21719).

The values live in the ``timeouts`` block of ``runtime_timeouts.yaml`` beside
this module. The model has no field defaults, so a block that omits a value
fails to load rather than falling back to a literal. Every field is strict: a
fractional or float-typed value for an integer field, and a boolean or a
non-finite value for a float field, is refused rather than truncated or coerced.
The file is read through ``importlib.resources``, so the package loads from a
directory, a zipped wheel or a zip import alike. The file ships with the wheel because hatchling includes
every file under ``src/omnibase_infra`` (``[tool.hatch.build.targets.wheel]``).
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

_TIMEOUTS_RESOURCE: Traversable = files(__package__).joinpath("runtime_timeouts.yaml")


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
        kafka_session_ms: Consumer group session timeout.
        kafka_heartbeat_interval_ms: Consumer group heartbeat interval.
        kafka_max_poll_interval_ms: Consumer group max poll interval.
        snapshot_consumer_wait_ms: Consumer wait of the snapshot publisher.
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

    capability_probe_ttl_seconds: int = Field(gt=0, strict=True)
    capability_probe_seconds: float = Field(gt=0, strict=True, allow_inf_nan=False)
    health_check_min_seconds: float = Field(gt=0, strict=True, allow_inf_nan=False)
    health_check_max_seconds: float = Field(gt=0, strict=True, allow_inf_nan=False)
    health_check_default_seconds: float = Field(gt=0, strict=True, allow_inf_nan=False)
    handler_shutdown_min_seconds: float = Field(gt=0, strict=True, allow_inf_nan=False)
    handler_shutdown_max_seconds: float = Field(gt=0, strict=True, allow_inf_nan=False)
    handler_shutdown_default_seconds: float = Field(
        gt=0, strict=True, allow_inf_nan=False
    )
    kafka_session_ms: int = Field(gt=0, strict=True)
    kafka_heartbeat_interval_ms: int = Field(gt=0, strict=True)
    kafka_max_poll_interval_ms: int = Field(gt=0, strict=True)
    snapshot_consumer_wait_ms: int = Field(gt=0, strict=True)
    circuit_breaker_reset_seconds: float = Field(gt=0, strict=True, allow_inf_nan=False)
    health_probe_seconds: float = Field(gt=0, strict=True, allow_inf_nan=False)
    local_dispatch_seconds: int = Field(gt=0, strict=True)
    terminal_seconds: float = Field(gt=0, strict=True, allow_inf_nan=False)
    llm_transport_max_seconds: float = Field(gt=0, strict=True, allow_inf_nan=False)
    llm_post_seconds: float = Field(gt=0, strict=True, allow_inf_nan=False)
    introspection_base_backoff_seconds: float = Field(
        gt=0, strict=True, allow_inf_nan=False
    )
    introspection_heartbeat_interval_seconds: float = Field(
        gt=0, strict=True, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def _defaults_lie_within_bounds(self) -> ModelRuntimeTimeouts:
        """Refuse a default outside its bounds, an inverted bound pair, an LLM POST wait above the transport ceiling, a heartbeat that outlasts the session and a poll interval below it."""
        if self.health_check_min_seconds > self.health_check_max_seconds:
            raise ValueError(
                "health_check_min_seconds exceeds health_check_max_seconds"
            )
        if self.handler_shutdown_min_seconds > self.handler_shutdown_max_seconds:
            raise ValueError(
                "handler_shutdown_min_seconds exceeds handler_shutdown_max_seconds"
            )
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
        if self.llm_post_seconds > self.llm_transport_max_seconds:
            raise ValueError("llm_post_seconds exceeds llm_transport_max_seconds")
        if self.kafka_heartbeat_interval_ms >= self.kafka_session_ms:
            raise ValueError(
                "kafka_heartbeat_interval_ms must be below kafka_session_ms"
            )
        if self.kafka_max_poll_interval_ms < self.kafka_session_ms:
            raise ValueError(
                "kafka_max_poll_interval_ms must not be below kafka_session_ms"
            )
        return self

    @classmethod
    def from_file(cls, timeouts_path: Traversable) -> ModelRuntimeTimeouts:
        """Load the timeouts from the ``timeouts`` mapping of a YAML resource.

        The resource is read through ``importlib.resources`` semantics, so the
        file loads from a directory, a zipped wheel or a zip import alike.

        Raises:
            ProtocolConfigurationError: If the file cannot be read or parsed,
                has no ``timeouts`` mapping, or holds a value the model refuses.
        """
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation="load_runtime_timeouts",
        )
        try:
            document = yaml.safe_load(timeouts_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            raise ProtocolConfigurationError(
                f"{timeouts_path.name} cannot be read as YAML",
                context=context,
            ) from error
        block = document.get("timeouts") if isinstance(document, dict) else None
        if not isinstance(block, dict):
            raise ProtocolConfigurationError(
                f"{timeouts_path.name} has no timeouts mapping",
                context=context,
            )
        try:
            return cls.model_validate(block)
        except ValidationError as error:
            raise ProtocolConfigurationError(
                f"{timeouts_path.name} holds a value the runtime timeouts refuse: {error}",
                context=context,
            ) from error


@cache
def load_runtime_timeouts() -> ModelRuntimeTimeouts:
    """Return the timeouts declared beside this module."""
    return ModelRuntimeTimeouts.from_file(_TIMEOUTS_RESOURCE)


__all__: list[str] = ["ModelRuntimeTimeouts", "load_runtime_timeouts"]
