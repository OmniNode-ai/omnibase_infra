# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Effect boundary for registration bootstrap configuration, files and database I/O."""

from __future__ import annotations

import logging
from pathlib import Path

import asyncpg
import yaml
from asyncpg import Pool

from omnibase_core.errors.error_service_resolution import ServiceResolutionError
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ContainerWiringError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.registration.model_registration_bootstrap_result import (
    ModelRegistrationBootstrapResult,
)
from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)
from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs
from omnibase_infra.runtime.protocol_domain_plugin import ModelDomainPluginConfig

logger = logging.getLogger(__name__)


class HandlerRegistrationBootstrap:
    """Acquire resources for the plugin, which owns their eventual shutdown."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container

    @classmethod
    async def from_container(
        cls, container: ModelONEXContainer
    ) -> HandlerRegistrationBootstrap:
        """Resolve an injected handler, registering the default bootstrap binding on a miss."""
        registry = container.service_registry
        if registry is None:
            registry = container.initialize_service_registry()
        try:
            return await container.get_service_async(cls)
        except ServiceResolutionError:
            await registry.register_instance(interface=cls, instance=cls(container))
            return await container.get_service_async(cls)

    @property
    def handler_type(self) -> EnumHandlerType:
        """Classify bootstrap resource acquisition as infrastructure handling."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Resource acquisition belongs to the EFFECT layer."""
        return EnumHandlerTypeCategory.EFFECT

    @staticmethod
    def database_url() -> str:
        """Resolve the DSN at the effect boundary at the time of the operation."""
        return expand_contract_env_refs("${env.OMNIBASE_INFRA_DB_URL}").strip()

    @staticmethod
    def projector_contracts_dir(default: Path) -> Path:
        """Resolve the optional projector-directory override at call time."""
        return Path(
            expand_contract_env_refs(
                "${env.ONEX_PROJECTOR_CONTRACTS_DIR:" + str(default) + "}"
            )
        )

    @staticmethod
    def load_contract(path: Path) -> dict[str, object]:
        """Load routing declarations with the same best-effort fallback as bootstrap."""
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(
                "Failed to load registration contract at %s: %s. Using empty topic router.",
                path,
                exc,
            )
            raw = {}
        return raw if isinstance(raw, dict) else {}

    async def handle(
        self, request: ModelDomainPluginConfig
    ) -> ModelRegistrationBootstrapResult:
        """Validate the configured DSN and acquire the original 2..10 connection pool."""
        correlation_id = request.correlation_id
        postgres_dsn = self.database_url()

        # Shared error context for all DSN validation failures
        pool_error_context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.DATABASE,
            operation="create_postgres_pool",
        )

        if not postgres_dsn:
            raise ContainerWiringError(
                "OMNIBASE_INFRA_DB_URL is required but not set",
                context=pool_error_context,
            )

        # Validate DSN scheme and database name before pool creation
        try:
            ModelPostgresPoolConfig.validate_dsn(postgres_dsn)
        except ValueError as exc:
            raise ContainerWiringError(
                str(exc),
                context=pool_error_context,
            ) from exc

        pool = await asyncpg.create_pool(  # no-contract-check: effect handler owns the pool
            postgres_dsn,
            min_size=2,
            max_size=10,
        )
        return ModelRegistrationBootstrapResult(pool=pool)

    @staticmethod
    def read_schema(path: Path) -> str:
        """Read the schema file; the caller retains its existing failure policy."""
        return path.read_text()

    @staticmethod
    def is_duplicate_schema_error(error: Exception) -> bool:
        """Classify the two database-driver errors bootstrap treats as idempotent."""
        return isinstance(
            error,
            (
                asyncpg.exceptions.DuplicateTableError,
                asyncpg.exceptions.DuplicateObjectError,
            ),
        )


__all__ = ["HandlerRegistrationBootstrap", "Pool"]
