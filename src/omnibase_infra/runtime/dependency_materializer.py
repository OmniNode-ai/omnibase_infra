# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract dependency materializer for infrastructure resources.

Reads contract.dependencies declarations and auto-creates live DI providers
(asyncpg pools, Kafka producers, HTTP clients) without domain-specific boot code.

Architecture:
    - Contracts declare: what resources they need (type, name, required)
    - Materializer creates: shared resource instances from environment config
    - Container receives: materialized resources for handler injection
    - Handlers consume: resources via ModelResolvedDependencies

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import yaml

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.enums.enum_infra_resource_type import (
    INFRA_RESOURCE_TYPES,
    EnumInfraResourceType,
)
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.runtime.models.model_http_client_config import (
    ModelHttpClientConfig,
)
from omnibase_infra.runtime.models.model_kafka_producer_config import (
    ModelKafkaProducerConfig,
)
from omnibase_infra.runtime.models.model_materialized_resources import (
    ModelMaterializedResources,
)
from omnibase_infra.runtime.models.model_materializer_config import (
    ModelMaterializerConfig,
)
from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)
from omnibase_infra.runtime.providers.provider_http_client import ProviderHttpClient
from omnibase_infra.runtime.providers.provider_kafka_producer import (
    ProviderKafkaProducer,
)
from omnibase_infra.runtime.providers.provider_postgres_pool import (
    ProviderPostgresPool,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

# Maximum contract file size (10 MB) - consistent with HandlerPluginLoader
_MAX_CONTRACT_SIZE = 10 * 1024 * 1024

# Type alias for provider close functions
_CloseFunc = Callable[[Any], Awaitable[None]]


class DependencyMaterializer:
    """Materializes infrastructure resources from contract.dependencies.

    Reads contract YAML files, extracts infrastructure-type dependencies
    (postgres_pool, kafka_producer, http_client), creates shared resource
    instances, and provides them for handler injection.

    Resources are deduplicated by type: all contracts declaring the same
    resource type share a single instance (one pool per connection config).

    Example:
        >>> materializer = DependencyMaterializer()
        >>> resources = await materializer.materialize([
        ...     Path("nodes/my_node/contract.yaml"),
        ... ])
        >>> pool = resources.get("pattern_store")
        >>>
        >>> # On shutdown:
        >>> await materializer.shutdown()
    """

    def __init__(
        self,
        config: ModelMaterializerConfig | None = None,
    ) -> None:
        """Initialize the dependency materializer.

        Args:
            config: Materializer configuration with provider-specific settings.
                When ``None`` (the default), individual sub-configs are resolved
                lazily from environment variables at the point a specific
                resource type is requested in ``_create_resource()``.  This
                avoids requiring ALL provider env vars (e.g.
                ``OMNIBASE_INFRA_DB_URL``) just to construct the materializer
                when only a subset of resource types will actually be used.
        """
        self._config: ModelMaterializerConfig | None = config
        self._lock = asyncio.Lock()

        # Resource cache: type -> resource instance (deduplication)
        # ONEX_EXCLUDE: any_type - heterogeneous resource instances
        self._resource_by_type: dict[str, Any] = {}

        # Name -> type mapping for alias resolution
        self._name_to_type: dict[str, str] = {}

        # Close functions for each resource type
        self._close_funcs: dict[str, _CloseFunc] = {}

        # Track creation order for reverse shutdown
        self._creation_order: list[str] = []

    async def materialize(
        self,
        contract_paths: list[Path],
    ) -> ModelMaterializedResources:
        """Materialize infrastructure resources from contract dependencies.

        Scans all contracts, extracts infrastructure-type dependencies,
        creates shared resources, and returns them keyed by dependency name.

        Args:
            contract_paths: Paths to contract YAML files to scan.

        Returns:
            ModelMaterializedResources with created resources keyed by name.

        Raises:
            ProtocolConfigurationError: If a required resource fails to create.
        """
        # Single correlation_id for the entire materialization pass
        correlation_id = uuid4()

        # Collect all infrastructure dependencies from contracts
        infra_deps = self._collect_infra_deps(contract_paths, correlation_id)

        if not infra_deps:
            logger.debug("No infrastructure dependencies found in contracts")
            return ModelMaterializedResources()

        logger.info(
            "Materializing infrastructure dependencies",
            extra={"dependency_count": len(infra_deps)},
        )

        # ONEX_EXCLUDE: any_type - heterogeneous resource instances
        resources: dict[str, Any] = {}

        # Hold lock for entire materialization to prevent TOCTOU races
        # when concurrent callers process the same resource types.
        async with self._lock:
            for dep in infra_deps:
                dep_name = dep.name
                dep_type = dep.type
                dep_required = getattr(dep, "required", True)

                # Check if resource type already created (deduplication)
                if dep_type in self._resource_by_type:
                    resources[dep_name] = self._resource_by_type[dep_type]
                    self._name_to_type[dep_name] = dep_type
                    logger.debug(
                        "Reusing existing resource for dependency",
                        extra={"dep_name": dep_name, "dep_type": dep_type},
                    )
                    continue

                # Create new resource via provider
                try:
                    resource = await self._create_resource(dep_type, correlation_id)

                    self._resource_by_type[dep_type] = resource
                    self._name_to_type[dep_name] = dep_type
                    self._creation_order.append(dep_type)
                    resources[dep_name] = resource

                    logger.info(
                        "Materialized infrastructure resource",
                        extra={"dep_name": dep_name, "dep_type": dep_type},
                    )

                except ProtocolConfigurationError:
                    # Contract/configuration errors always propagate
                    raise
                except (TypeError, AttributeError) as e:
                    # Programming errors propagate immediately
                    raise
                except Exception as e:
                    # Infrastructure errors: OSError, TimeoutError, ImportError
                    # (missing optional package), and library-specific errors
                    # (KafkaConnectionError, asyncpg.PostgresError, etc.)
                    if dep_required:
                        context = ModelInfraErrorContext.with_correlation(
                            correlation_id=correlation_id,
                            transport_type=EnumInfraTransportType.RUNTIME,
                            operation="materialize_dependency",
                            target_name=dep_name,
                        )
                        safe_msg = sanitize_error_message(e)
                        raise ProtocolConfigurationError(
                            f"Failed to materialize required dependency "
                            f"'{dep_name}' (type={dep_type}): {safe_msg}",
                            context=context,
                        ) from e

                    logger.warning(
                        "Optional dependency materialization failed, skipping",
                        extra={
                            "dep_name": dep_name,
                            "dep_type": dep_type,
                            "error": sanitize_error_message(e),
                        },
                    )

        return ModelMaterializedResources(resources=resources)

    async def shutdown(self) -> None:
        """Close all materialized resources in reverse creation order.

        Holds lock for entire shutdown to prevent concurrent materialize()
        from adding resources mid-shutdown. Falls back to _resource_by_type
        keys if a resource was registered but not tracked in _creation_order.
        Errors during individual close operations are logged but do not
        propagate or halt remaining closes.
        """
        async with self._lock:
            # Build close list: _creation_order (reversed) + any untracked
            ordered = list(reversed(self._creation_order))
            extra = [
                rt for rt in self._resource_by_type if rt not in self._creation_order
            ]
            types_to_close = ordered + extra

            for resource_type in types_to_close:
                resource = self._resource_by_type.get(resource_type)
                close_func = self._close_funcs.get(resource_type)

                if resource is None or close_func is None:
                    continue

                try:
                    await close_func(resource)
                    logger.info(
                        "Closed materialized resource",
                        extra={"type": resource_type},
                    )
                except Exception as e:
                    logger.warning(
                        "Error closing materialized resource",
                        extra={"type": resource_type, "error": str(e)},
                    )

            self._resource_by_type.clear()
            self._name_to_type.clear()
            self._close_funcs.clear()
            self._creation_order.clear()

    def _collect_infra_deps(
        self,
        contract_paths: list[Path],
        correlation_id: UUID,
    ) -> list[SimpleNamespace]:
        """Extract infrastructure-type dependencies from contracts.

        Args:
            contract_paths: Paths to scan for contract YAML files.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            List of dependency objects with name, type, required fields.
        """
        deps: list[SimpleNamespace] = []
        seen_names: dict[str, str] = {}  # name -> type for conflict detection

        for path in contract_paths:
            try:
                contract_data = self._load_contract_yaml(path, correlation_id)
            except (OSError, yaml.YAMLError, ProtocolConfigurationError) as e:
                logger.warning(
                    "Failed to load contract for dependency scanning",
                    extra={"path": str(path), "error": str(e)},
                )
                continue

            dependencies = contract_data.get("dependencies", [])
            if not dependencies:
                continue

            for dep_data in dependencies:
                if not isinstance(dep_data, dict):
                    continue

                dep_type = dep_data.get("type", "")
                dep_name = dep_data.get("name", "")

                if dep_type not in INFRA_RESOURCE_TYPES:
                    continue

                if not dep_name:
                    logger.warning(
                        "Infrastructure dependency missing name, skipping",
                        extra={"dep_type": dep_type, "path": str(path)},
                    )
                    continue

                # Deduplicate by name (first declaration wins)
                if dep_name in seen_names:
                    if seen_names[dep_name] != dep_type:
                        context = ModelInfraErrorContext.with_correlation(
                            correlation_id=correlation_id,
                            transport_type=EnumInfraTransportType.RUNTIME,
                            operation="collect_infra_deps",
                            target_name=dep_name,
                        )
                        raise ProtocolConfigurationError(
                            f"Dependency name '{dep_name}' declared with conflicting "
                            f"types: '{seen_names[dep_name]}' vs '{dep_type}' "
                            f"(in {path}). Each dependency name must resolve to a "
                            f"single resource type.",
                            context=context,
                        )
                    continue
                seen_names[dep_name] = dep_type

                deps.append(
                    SimpleNamespace(
                        name=dep_name,
                        type=dep_type,
                        required=dep_data.get("required", True),
                    )
                )

        return deps

    # ONEX_EXCLUDE: any_type - returns heterogeneous resource instance
    async def _create_resource(self, resource_type: str, correlation_id: UUID) -> Any:
        """Create a resource using the appropriate provider.

        Args:
            resource_type: The resource type string (e.g., "postgres_pool").
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            The created resource instance.

        Raises:
            ProtocolConfigurationError: If the resource type has no registered provider.
        """
        if resource_type == EnumInfraResourceType.POSTGRES_POOL:
            pg_config = (
                self._config.postgres
                if self._config is not None
                else ModelPostgresPoolConfig.from_env()
            )
            provider = ProviderPostgresPool(pg_config)
            resource = await provider.create()
            self._close_funcs[resource_type] = ProviderPostgresPool.close
            return resource

        if resource_type == EnumInfraResourceType.KAFKA_PRODUCER:
            kafka_config = (
                self._config.kafka
                if self._config is not None
                else ModelKafkaProducerConfig.from_env()
            )
            provider_kafka = ProviderKafkaProducer(kafka_config)
            resource = await provider_kafka.create()
            self._close_funcs[resource_type] = ProviderKafkaProducer.close
            return resource

        if resource_type == EnumInfraResourceType.HTTP_CLIENT:
            http_config = (
                self._config.http
                if self._config is not None
                else ModelHttpClientConfig.from_env()
            )
            provider_http = ProviderHttpClient(http_config)
            resource = await provider_http.create()
            self._close_funcs[resource_type] = ProviderHttpClient.close
            return resource

        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="create_resource",
            target_name=resource_type,
        )
        raise ProtocolConfigurationError(
            f"No provider registered for resource type '{resource_type}'. "
            f"Supported types: {list(INFRA_RESOURCE_TYPES)}",
            context=context,
        )

    # ONEX_EXCLUDE: any_type - yaml.safe_load returns heterogeneous dict
    def _load_contract_yaml(self, path: Path, correlation_id: UUID) -> dict[str, Any]:
        """Load and parse a contract YAML file.

        Args:
            path: Path to the contract YAML file.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Parsed YAML content as a dictionary.

        Raises:
            ProtocolConfigurationError: If file cannot be loaded or parsed.
        """
        if not path.exists():
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_contract_yaml",
                target_name=str(path),
            )
            raise ProtocolConfigurationError(
                f"Contract file not found: {path}",
                context=context,
            )

        file_size = path.stat().st_size
        if file_size > _MAX_CONTRACT_SIZE:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_contract_yaml",
                target_name=str(path),
            )
            raise ProtocolConfigurationError(
                f"Contract file too large: {file_size} bytes "
                f"(max: {_MAX_CONTRACT_SIZE} bytes)",
                context=context,
            )

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data is None:
                    return {}
                if not isinstance(data, dict):
                    logger.warning(
                        "Contract YAML parsed to non-dict type, treating as empty",
                        extra={"path": str(path), "type": type(data).__name__},
                    )
                    return {}
                return data
        except yaml.YAMLError as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_contract_yaml",
                target_name=str(path),
            )
            raise ProtocolConfigurationError(
                f"Failed to parse contract YAML at {path}: {e}",
                context=context,
            ) from e


__all__ = ["DependencyMaterializer"]
