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
from omnibase_infra.runtime.models.model_materialized_resources import (
    ModelMaterializedResources,
)
from omnibase_infra.runtime.models.model_materializer_config import (
    ModelMaterializerConfig,
)
from omnibase_infra.runtime.providers.provider_http_client import ProviderHttpClient
from omnibase_infra.runtime.providers.provider_kafka_producer import (
    ProviderKafkaProducer,
)
from omnibase_infra.runtime.providers.provider_postgres_pool import (
    ProviderPostgresPool,
)

logger = logging.getLogger(__name__)


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
        self._config = config or ModelMaterializerConfig.from_env()
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
        # Collect all infrastructure dependencies from contracts
        infra_deps = self._collect_infra_deps(contract_paths)

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
                    resource = await self._create_resource(dep_type)

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
                except (OSError, TimeoutError) as e:
                    if dep_required:
                        context = ModelInfraErrorContext.with_correlation(
                            transport_type=EnumInfraTransportType.RUNTIME,
                            operation="materialize_dependency",
                            target_name=dep_name,
                        )
                        raise ProtocolConfigurationError(
                            f"Failed to materialize required dependency "
                            f"'{dep_name}' (type={dep_type}): {e}",
                            context=context,
                        ) from e

                    logger.warning(
                        "Optional dependency materialization failed, skipping",
                        extra={
                            "dep_name": dep_name,
                            "dep_type": dep_type,
                            "error": str(e),
                        },
                    )

        return ModelMaterializedResources(resources=resources)

    async def shutdown(self) -> None:
        """Close all materialized resources in reverse creation order.

        Falls back to _resource_by_type keys if a resource was registered
        but not tracked in _creation_order. Errors during shutdown are
        logged but do not propagate.
        """
        async with self._lock:
            # Use _creation_order for deterministic reverse shutdown,
            # but also include any resource types only in _resource_by_type
            ordered = list(reversed(self._creation_order))
            extra = [
                rt for rt in self._resource_by_type if rt not in self._creation_order
            ]
            types_to_close = ordered + extra

        for resource_type in types_to_close:
            async with self._lock:
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

        async with self._lock:
            self._resource_by_type.clear()
            self._name_to_type.clear()
            self._close_funcs.clear()
            self._creation_order.clear()

    def _collect_infra_deps(
        self,
        contract_paths: list[Path],
    ) -> list[SimpleNamespace]:
        """Extract infrastructure-type dependencies from contracts.

        Args:
            contract_paths: Paths to scan for contract YAML files.

        Returns:
            List of dependency objects with name, type, required fields.
        """
        deps: list[SimpleNamespace] = []
        seen_names: set[str] = set()

        for path in contract_paths:
            try:
                contract_data = self._load_contract_yaml(path)
            except Exception as e:
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
                    continue
                seen_names.add(dep_name)

                deps.append(
                    SimpleNamespace(
                        name=dep_name,
                        type=dep_type,
                        required=dep_data.get("required", True),
                    )
                )

        return deps

    # ONEX_EXCLUDE: any_type - returns heterogeneous resource instance
    async def _create_resource(self, resource_type: str) -> Any:
        """Create a resource using the appropriate provider.

        Args:
            resource_type: The resource type string (e.g., "postgres_pool").

        Returns:
            The created resource instance.

        Raises:
            ProtocolConfigurationError: If the resource type has no registered provider.
        """
        if resource_type == EnumInfraResourceType.POSTGRES_POOL:
            provider = ProviderPostgresPool(self._config.postgres)
            self._close_funcs[resource_type] = ProviderPostgresPool.close
            return await provider.create()

        if resource_type == EnumInfraResourceType.KAFKA_PRODUCER:
            provider_kafka = ProviderKafkaProducer(self._config.kafka)
            self._close_funcs[resource_type] = ProviderKafkaProducer.close
            return await provider_kafka.create()

        if resource_type == EnumInfraResourceType.HTTP_CLIENT:
            provider_http = ProviderHttpClient(self._config.http)
            self._close_funcs[resource_type] = ProviderHttpClient.close
            return await provider_http.create()

        context = ModelInfraErrorContext.with_correlation(
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
    def _load_contract_yaml(self, path: Path) -> dict[str, Any]:
        """Load and parse a contract YAML file.

        Args:
            path: Path to the contract YAML file.

        Returns:
            Parsed YAML content as a dictionary.

        Raises:
            ProtocolConfigurationError: If file cannot be loaded or parsed.
        """
        if not path.exists():
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_contract_yaml",
                target_name=str(path),
            )
            raise ProtocolConfigurationError(
                f"Contract file not found: {path}",
                context=context,
            )

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data is None:
                    return {}
                if not isinstance(data, dict):
                    return {}
                return data
        except yaml.YAMLError as e:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_contract_yaml",
                target_name=str(path),
            )
            raise ProtocolConfigurationError(
                f"Failed to parse contract YAML at {path}: {e}",
                context=context,
            ) from e


__all__ = ["DependencyMaterializer"]
