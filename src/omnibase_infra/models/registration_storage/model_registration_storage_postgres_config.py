# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Settings the PostgreSQL registration storage handler resolves from the container (OMN-22131).

The operational values (database, user, pool size, statement timeout, circuit
breaker) live in the ``registration_storage_postgres`` block of
``registration_storage_postgres.yaml`` beside this module. The deployment
values (DSN or host, port, password, schema auto-creation) are supplied by the
wiring site. The model has no field defaults, so an omitted value fails to load
rather than falling back to a literal.
"""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig

DECLARED_SETTINGS_PATH = files("omnibase_infra.models.registration_storage").joinpath(
    "registration_storage_postgres.yaml"
)
_BLOCK_NAME = "registration_storage_postgres"


class ModelRegistrationStoragePostgresConfig(BaseModel):
    """Connection, pool and resilience settings of HandlerRegistrationStoragePostgres.

    Attributes:
        dsn: PostgreSQL DSN. When non-empty it takes precedence over host,
            port, database, user and password.
        host: PostgreSQL server hostname, used when ``dsn`` is empty.
        port: PostgreSQL server port.
        database: Database name.
        user: Database user.
        password: Database password; empty means no password is sent.
        pool_size: Maximum connections in the asyncpg pool.
        timeout_seconds: Wait for one query or statement.
        auto_create_schema: Create the node_registrations table on first
            connection. Production deployments use migrations instead.
        circuit_breaker: Circuit breaker settings of the handler.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    dsn: SecretStr
    host: str
    port: int = Field(ge=1, le=65535)
    database: str = Field(min_length=1)
    user: str = Field(min_length=1)
    password: SecretStr
    pool_size: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    auto_create_schema: bool
    circuit_breaker: ModelCircuitBreakerConfig

    @model_validator(mode="after")
    def _dsn_or_host(self) -> ModelRegistrationStoragePostgresConfig:
        """Refuse a config that names neither a DSN nor a host."""
        if not self.dsn.get_secret_value() and not self.host:
            raise ValueError("registration storage config needs a dsn or a host")
        return self

    @classmethod
    def from_file(
        cls,
        settings_path: Traversable,
        *,
        dsn: str,
        host: str,
        port: int,
        password: str,
        auto_create_schema: bool,
    ) -> ModelRegistrationStoragePostgresConfig:
        """Load the declared block and add the deployment values.

        Package resources support both extracted wheels and zip imports without
        deriving a filesystem location from this module's source file.

        Raises:
            ProtocolConfigurationError: If the file has no settings mapping.
        """
        with settings_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        block = document.get(_BLOCK_NAME) if isinstance(document, dict) else None
        if not isinstance(block, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_registration_storage_postgres_config",
            )
            raise ProtocolConfigurationError(
                f"{settings_path.name} has no {_BLOCK_NAME} mapping",
                context=context,
            )
        return cls.model_validate(
            {
                **block,
                "dsn": dsn,
                "host": host,
                "port": port,
                "password": password,
                "auto_create_schema": auto_create_schema,
            }
        )


__all__: list[str] = [
    "DECLARED_SETTINGS_PATH",
    "ModelRegistrationStoragePostgresConfig",
]
