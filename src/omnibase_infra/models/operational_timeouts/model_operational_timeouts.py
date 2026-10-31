# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared operational timeouts (OMN-21802).

The values live in the ``timeouts`` block of ``operational_timeouts.yaml`` beside
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

_TIMEOUTS_PATH = Path(__file__).resolve().parent / "operational_timeouts.yaml"


class ModelOperationalTimeouts(BaseModel):
    """Waits, intervals and backoffs the service call sites apply.

    Attributes:
        outbox_poll_interval_seconds: Idle poll period of the outbox.
        outbox_shutdown_seconds: Wait for the outbox processor on stop.
        database_query_seconds: Wait of one outbox or projector database query.
        circuit_breaker_reset_seconds: Wait before an open circuit resets.
        scheduler_tick_interval_ms: Period of the runtime scheduler tick.
        tick_stop_seconds: Wait for the scheduler tick task on stop.
        scheduler_valkey_seconds: Wait of the scheduler's Valkey connection.
        freshness_interval_seconds: Period of the projection freshness sweep.
        optimistic_retry_initial_backoff_seconds: First backoff after a conflict.
        optimistic_retry_max_backoff_seconds: Ceiling of the conflict backoff.
        optimistic_retry_backoff_multiplier: Growth factor of the conflict backoff.
        consumer_restart_initial_backoff_seconds: First backoff of a restart.
        consumer_restart_max_backoff_seconds: Ceiling of the restart backoff.
        consumer_restart_backoff_factor: Growth factor of the restart backoff.
        kafka_admin_seconds: Wait of one Kafka admin call or future.
        kafka_watermark_seconds: Wait of one watermark offset query.
        dev_lane_ledger_seconds: Wait of a dev-lane ledger append.
        dev_lane_docker_seconds: Wait of one dev-lane docker call.
        dev_lane_restart_seconds: Wait of a dev-lane container restart.
        done_write_verify_seconds: Wait of one done-write verifier run.
        kill_wait_seconds: Wait for a killed verifier to exit.
        tool_execution_seconds: Wait of one tool execution orchestrator call.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    outbox_poll_interval_seconds: float = Field(gt=0)
    outbox_shutdown_seconds: float = Field(gt=0)
    database_query_seconds: float = Field(gt=0)
    circuit_breaker_reset_seconds: float = Field(gt=0)
    scheduler_tick_interval_ms: int = Field(gt=0)
    tick_stop_seconds: float = Field(gt=0)
    scheduler_valkey_seconds: float = Field(gt=0)
    freshness_interval_seconds: float = Field(gt=0)
    optimistic_retry_initial_backoff_seconds: float = Field(gt=0)
    optimistic_retry_max_backoff_seconds: float = Field(gt=0)
    optimistic_retry_backoff_multiplier: float = Field(ge=1)
    consumer_restart_initial_backoff_seconds: float = Field(gt=0)
    consumer_restart_max_backoff_seconds: float = Field(gt=0)
    consumer_restart_backoff_factor: float = Field(ge=1)
    kafka_admin_seconds: int = Field(gt=0)
    kafka_watermark_seconds: int = Field(gt=0)
    dev_lane_ledger_seconds: int = Field(gt=0)
    dev_lane_docker_seconds: float = Field(gt=0)
    dev_lane_restart_seconds: float = Field(gt=0)
    done_write_verify_seconds: float = Field(gt=0)
    kill_wait_seconds: int = Field(gt=0)
    tool_execution_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def _backoffs_are_ordered(self) -> ModelOperationalTimeouts:
        """Refuse an initial backoff above its ceiling."""
        if (
            self.optimistic_retry_initial_backoff_seconds
            > self.optimistic_retry_max_backoff_seconds
        ):
            raise ValueError("optimistic retry initial backoff exceeds its maximum")
        if (
            self.consumer_restart_initial_backoff_seconds
            > self.consumer_restart_max_backoff_seconds
        ):
            raise ValueError("consumer restart initial backoff exceeds its maximum")
        return self

    @classmethod
    def from_file(cls, timeouts_path: Path) -> ModelOperationalTimeouts:
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
                operation="load_operational_timeouts",
            )
            raise ProtocolConfigurationError(
                f"{timeouts_path.name} has no timeouts mapping",
                context=context,
            )
        return cls.model_validate(block)


@cache
def load_operational_timeouts() -> ModelOperationalTimeouts:
    """Return the timeouts declared beside this module."""
    return ModelOperationalTimeouts.from_file(_TIMEOUTS_PATH)


__all__: list[str] = ["ModelOperationalTimeouts", "load_operational_timeouts"]
