# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared endpoints and waits for ``onex-infra-test`` (OMN-21623).

The values live in the ``settings`` block of ``cli/infra_test/infra_test_settings.yaml``.
The model has no field defaults, so a block that omits a value fails to load
rather than falling back to a literal.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

_SETTINGS_PATH = (
    Path(__file__).resolve().parents[2]
    / "cli"
    / "infra_test"
    / "infra_test_settings.yaml"
)


class ModelInfraTestSettings(BaseModel):
    """Endpoints, timeouts and waits that the infra-test CLI runs under.

    Attributes:
        broker_address: Kafka bootstrap address used when neither the flag nor
            ``KAFKA_BOOTSTRAP_SERVERS`` names one.
        runtime_url: Runtime base URL used when ``ONEX_RUNTIME_URL`` is unset.
        compose_command_timeout_seconds: Wait for one ``docker compose`` call.
        compose_start_timeout_seconds: Wait for ``docker compose start``.
        docker_query_timeout_seconds: Wait for a ``docker compose`` status,
            listing or kill call.
        rpk_command_timeout_seconds: Wait for one ``rpk`` produce, list or
            describe call.
        rpk_consume_timeout_seconds: Wait for the single-record ``rpk`` consume.
        cli_subcommand_timeout_seconds: Wait for a nested ``onex-infra-test``
            subcommand.
        postgres_connect_timeout_seconds: PostgreSQL connection timeout.
        health_wait_timeout_seconds: Default wait for the stack to be healthy.
        health_poll_interval_seconds: Pause between health polls.
        registration_wait_seconds: Wait for a registration projection row.
        registration_backfill_wait_seconds: Wait for a registration row after
            the runtime restarts.
        registration_poll_interval_seconds: Pause between registration polls.
        idempotency_settle_seconds: Pause for consumer lag to clear after
            duplicate publishes.
        runtime_restart_wait_seconds: Pause for the runtime to come back.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    broker_address: str = Field(min_length=1)
    runtime_url: str = Field(min_length=1)
    compose_command_timeout_seconds: float = Field(gt=0)
    compose_start_timeout_seconds: float = Field(gt=0)
    docker_query_timeout_seconds: float = Field(gt=0)
    rpk_command_timeout_seconds: float = Field(gt=0)
    rpk_consume_timeout_seconds: float = Field(gt=0)
    cli_subcommand_timeout_seconds: float = Field(gt=0)
    postgres_connect_timeout_seconds: int = Field(gt=0)
    health_wait_timeout_seconds: int = Field(gt=0)
    health_poll_interval_seconds: float = Field(gt=0)
    registration_wait_seconds: int = Field(gt=0)
    registration_backfill_wait_seconds: int = Field(gt=0)
    registration_poll_interval_seconds: float = Field(gt=0)
    idempotency_settle_seconds: float = Field(gt=0)
    runtime_restart_wait_seconds: float = Field(gt=0)

    @classmethod
    def from_file(cls, settings_path: Path) -> ModelInfraTestSettings:
        """Load the settings from the ``settings`` mapping of a YAML file.

        Raises:
            ProtocolConfigurationError: If the file has no ``settings`` mapping.
        """
        with settings_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        block = document.get("settings") if isinstance(document, dict) else None
        if not isinstance(block, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_infra_test_settings",
            )
            raise ProtocolConfigurationError(
                f"{settings_path.name} has no settings mapping",
                context=context,
            )
        return cls.model_validate(block)


@cache
def load_infra_test_settings() -> ModelInfraTestSettings:
    """Return the settings declared beside the infra-test CLI."""
    return ModelInfraTestSettings.from_file(_SETTINGS_PATH)


__all__: list[str] = ["ModelInfraTestSettings", "load_infra_test_settings"]
