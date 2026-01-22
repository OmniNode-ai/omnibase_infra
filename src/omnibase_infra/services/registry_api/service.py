# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry Discovery Service.

Combines ProjectionReaderRegistration and HandlerServiceDiscoveryConsul
to provide a unified discovery interface for the Registry API.

Design Principles:
    - Partial success: Returns data even if one backend fails
    - Warnings array: Communicates backend failures without crashing
    - Async-first: All methods are async for non-blocking I/O
    - Correlation IDs: Full traceability across all operations
    - Container DI: Accepts ModelONEXContainer for dependency injection

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
    - OMN-1282: MCP Handler Contract-Driven Config
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.nodes.node_service_discovery_effect.models.enum_health_status import (
    EnumHealthStatus,
)
from omnibase_infra.services.registry_api.models import (
    ModelCapabilityWidgetMapping,
    ModelPaginationInfo,
    ModelRegistryDiscoveryResponse,
    ModelRegistryHealthResponse,
    ModelRegistryInstanceView,
    ModelRegistryNodeView,
    ModelRegistrySummary,
    ModelWarning,
    ModelWidgetDefaults,
    ModelWidgetMapping,
)

if TYPE_CHECKING:
    from omnibase_infra.handlers.service_discovery import HandlerServiceDiscoveryConsul
    from omnibase_infra.models.projection import ModelRegistrationProjection
    from omnibase_infra.projectors import ProjectionReaderRegistration

logger = logging.getLogger(__name__)

# Maximum records to fetch when node_type filtering requires in-memory pagination.
# The projection reader API doesn't support node_type filtering, so we fetch all
# records matching the state filter and apply node_type filter in-memory.
MAX_NODE_TYPE_FILTER_FETCH = 10000

# Default config path relative to this module
DEFAULT_WIDGET_MAPPING_PATH = (
    Path(__file__).parent.parent.parent / "configs" / "widget_mapping.yaml"
)


class ServiceRegistryDiscovery:
    """Registry discovery service combining projection and Consul data.

    Provides a unified interface for querying both registered nodes
    (from PostgreSQL projections) and live service instances (from Consul).

    Partial Success Pattern:
        If one backend fails, the service still returns data from the
        successful backend along with warnings indicating the failure.
        This allows dashboards to display partial data rather than
        showing complete errors.

    Dependency Injection:
        This service requires a ModelONEXContainer for ONEX-style dependency
        injection. Dependencies can also be provided directly via constructor
        parameters for testing flexibility.

    Thread Safety:
        This service is coroutine-safe. All methods are async and
        delegate to underlying services that handle their own
        concurrency requirements.

    Example:
        >>> # Using container for DI (container is required)
        >>> service = ServiceRegistryDiscovery(container=container)
        >>> response = await service.get_discovery()
        >>>
        >>> # With explicit dependencies (for testing)
        >>> service = ServiceRegistryDiscovery(
        ...     container=container,
        ...     projection_reader=reader,
        ...     consul_handler=handler,
        ... )
        >>> response = await service.get_discovery()
        >>> if response.warnings:
        ...     logger.warning("Partial data: %s", response.warnings)

    Attributes:
        projection_reader: Reader for node registration projections.
        consul_handler: Handler for Consul service discovery.
        widget_mapping_path: Path to widget mapping YAML configuration.
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        projection_reader: ProjectionReaderRegistration | None = None,
        consul_handler: HandlerServiceDiscoveryConsul | None = None,
        widget_mapping_path: Path | None = None,
    ) -> None:
        """Initialize the registry discovery service.

        Args:
            container: ONEX container for dependency injection. Required for
                ONEX DI pattern compliance.
            projection_reader: Optional projection reader for node registrations.
                If not provided, node queries will return empty results with warnings.
            consul_handler: Optional Consul handler for live instances.
                If not provided, instance queries will return empty results with warnings.
            widget_mapping_path: Path to widget mapping YAML file.
                Defaults to configs/widget_mapping.yaml relative to package.
        """
        self._container = container

        # Resolve projection_reader: direct param > None
        # NOTE: Container-based resolution removed in omnibase_core ^0.9.0.
        # The new ServiceRegistry uses async interface-based resolution which
        # doesn't fit the sync __init__ pattern. Use explicit dependency injection
        # via the projection_reader parameter instead.
        self._projection_reader = projection_reader

        # Resolve consul_handler: direct param > None
        # NOTE: Container-based resolution removed in omnibase_core ^0.9.0.
        # The new ServiceRegistry uses async interface-based resolution which
        # doesn't fit the sync __init__ pattern. Use explicit dependency injection
        # via the consul_handler parameter instead.
        self._consul_handler = consul_handler

        self._widget_mapping_path = widget_mapping_path or DEFAULT_WIDGET_MAPPING_PATH
        self._widget_mapping_cache: ModelWidgetMapping | None = None
        self._widget_mapping_mtime: float | None = None

        logger.info(
            "ServiceRegistryDiscovery initialized",
            extra={
                "has_projection_reader": self._projection_reader is not None,
                "has_consul_handler": self._consul_handler is not None,
                "widget_mapping_path": str(self._widget_mapping_path),
            },
        )

    @property
    def has_projection_reader(self) -> bool:
        """Check if projection reader is configured."""
        return self._projection_reader is not None

    @property
    def has_consul_handler(self) -> bool:
        """Check if Consul handler is configured."""
        return self._consul_handler is not None

    @property
    def consul_handler(self) -> HandlerServiceDiscoveryConsul | None:
        """Get the Consul handler for lifecycle management."""
        return self._consul_handler

    def invalidate_widget_mapping_cache(self) -> None:
        """Clear widget mapping cache, forcing reload on next access.

        Use this method when you know the widget mapping file has changed
        and want to force an immediate reload, rather than waiting for
        file modification time detection.

        Example:
            >>> service.invalidate_widget_mapping_cache()
            >>> mapping, warnings = service.get_widget_mapping()  # Fresh load
        """
        self._widget_mapping_cache = None
        self._widget_mapping_mtime = None
        logger.debug(
            "Widget mapping cache invalidated",
            extra={"widget_mapping_path": str(self._widget_mapping_path)},
        )

    async def list_nodes(
        self,
        limit: int = 100,
        offset: int = 0,
        state: EnumRegistrationState | None = None,
        node_type: str | None = None,
        correlation_id: UUID | None = None,
    ) -> tuple[list[ModelRegistryNodeView], ModelPaginationInfo, list[ModelWarning]]:
        """List registered nodes with pagination.

        Args:
            limit: Maximum number of nodes to return (1-1000).
            offset: Number of nodes to skip for pagination.
            state: Optional filter by registration state. When None, queries
                all active states (ACTIVE, ACCEPTED, AWAITING_ACK, ACK_RECEIVED).
            node_type: Optional filter by node type (effect, compute, reducer,
                orchestrator). Case-insensitive.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Tuple of (nodes, pagination_info, warnings).

        Note:
            When node_type filter is specified, all matching records are fetched
            to provide accurate pagination totals. For large datasets, consider
            using state filters to reduce the query scope.
        """
        correlation_id = correlation_id or uuid4()
        warnings: list[ModelWarning] = []
        nodes: list[ModelRegistryNodeView] = []
        total = 0

        if self._projection_reader is None:
            warnings.append(
                ModelWarning(
                    source="postgres",
                    message="Projection reader not configured",
                    code="NO_PROJECTION_READER",
                    timestamp=datetime.now(UTC),
                )
            )
        else:
            try:
                # Determine fetch limit based on whether node_type filter is applied
                # When node_type is specified, we need all records for accurate totals
                # since the projection reader doesn't support node_type filtering
                if node_type:
                    # Fetch all matching records to get accurate count after filtering
                    fetch_limit = MAX_NODE_TYPE_FILTER_FETCH
                else:
                    # No node_type filter - can use normal pagination
                    fetch_limit = limit + offset + 1  # +1 to detect has_more

                # Query projections based on state filter
                projections: list[ModelRegistrationProjection] = []

                if state is not None:
                    # Single state filter
                    projections = await self._projection_reader.get_by_state(
                        state=state,
                        limit=fetch_limit,
                        correlation_id=correlation_id,
                    )
                else:
                    # No state filter - query all active states and combine
                    # This provides results across all relevant states, not just ACTIVE
                    active_states = [
                        EnumRegistrationState.ACTIVE,
                        EnumRegistrationState.ACCEPTED,
                        EnumRegistrationState.AWAITING_ACK,
                        EnumRegistrationState.ACK_RECEIVED,
                        EnumRegistrationState.PENDING_REGISTRATION,
                    ]
                    all_projections: list[ModelRegistrationProjection] = []
                    for query_state in active_states:
                        state_projections = await self._projection_reader.get_by_state(
                            state=query_state,
                            limit=fetch_limit,
                            correlation_id=correlation_id,
                        )
                        all_projections.extend(state_projections)

                    # Sort combined results by updated_at descending
                    projections = sorted(
                        all_projections,
                        key=lambda p: p.updated_at,
                        reverse=True,
                    )

                # Apply node_type filter in-memory if specified
                # The projection reader API doesn't support node_type filtering
                node_type_filter = node_type.upper() if node_type else None
                if node_type_filter:
                    projections = [
                        p
                        for p in projections
                        if p.node_type.value.upper() == node_type_filter
                    ]

                # Calculate total from ALL filtered records (accurate count)
                total = len(projections)

                # Apply offset and limit for pagination
                projections_slice = projections[offset : offset + limit]

                # Convert to view models
                for proj in projections_slice:
                    # Map EnumNodeKind to API node_type string
                    node_type_str = proj.node_type.value.upper()
                    if node_type_str not in (
                        "EFFECT",
                        "COMPUTE",
                        "REDUCER",
                        "ORCHESTRATOR",
                    ):
                        node_type_str = "EFFECT"  # Fallback

                    nodes.append(
                        ModelRegistryNodeView(
                            node_id=proj.entity_id,
                            name=f"onex-{proj.node_type.value}",
                            service_name=f"onex-{proj.node_type.value}-{str(proj.entity_id)[:8]}",
                            namespace=proj.domain
                            if proj.domain != "registration"
                            else None,
                            display_name=None,
                            node_type=node_type_str,  # type: ignore[arg-type]
                            version=proj.node_version,
                            state=proj.current_state.value,
                            capabilities=proj.capability_tags,
                            registered_at=proj.registered_at,
                            last_heartbeat_at=proj.last_heartbeat_at,
                        )
                    )

            except Exception as e:
                logger.exception(
                    "Failed to query projections",
                    extra={"correlation_id": str(correlation_id)},
                )
                warnings.append(
                    ModelWarning(
                        source="postgres",
                        message=f"Failed to query projections: {type(e).__name__}",
                        code="PROJECTION_QUERY_FAILED",
                        timestamp=datetime.now(UTC),
                    )
                )

        pagination = ModelPaginationInfo(
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(nodes) < total,
        )

        return nodes, pagination, warnings

    async def get_node(
        self,
        node_id: UUID,
        correlation_id: UUID | None = None,
    ) -> tuple[ModelRegistryNodeView | None, list[ModelWarning]]:
        """Get a single node by ID.

        Args:
            node_id: Node UUID to retrieve.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Tuple of (node or None, warnings).
        """
        correlation_id = correlation_id or uuid4()
        warnings: list[ModelWarning] = []

        if self._projection_reader is None:
            warnings.append(
                ModelWarning(
                    source="postgres",
                    message="Projection reader not configured",
                    code="NO_PROJECTION_READER",
                    timestamp=datetime.now(UTC),
                )
            )
            return None, warnings

        try:
            proj = await self._projection_reader.get_entity_state(
                entity_id=node_id,
                correlation_id=correlation_id,
            )

            if proj is None:
                return None, warnings

            node_type_str = proj.node_type.value.upper()
            if node_type_str not in ("EFFECT", "COMPUTE", "REDUCER", "ORCHESTRATOR"):
                node_type_str = "EFFECT"

            node = ModelRegistryNodeView(
                node_id=proj.entity_id,
                name=f"onex-{proj.node_type.value}",
                service_name=f"onex-{proj.node_type.value}-{str(proj.entity_id)[:8]}",
                namespace=proj.domain if proj.domain != "registration" else None,
                display_name=None,
                node_type=node_type_str,  # type: ignore[arg-type]
                version=proj.node_version,
                state=proj.current_state.value,
                capabilities=proj.capability_tags,
                registered_at=proj.registered_at,
                last_heartbeat_at=proj.last_heartbeat_at,
            )

            return node, warnings

        except Exception as e:
            logger.exception(
                "Failed to get node",
                extra={"node_id": str(node_id), "correlation_id": str(correlation_id)},
            )
            warnings.append(
                ModelWarning(
                    source="postgres",
                    message=f"Failed to get node: {type(e).__name__}",
                    code="NODE_QUERY_FAILED",
                    timestamp=datetime.now(UTC),
                )
            )
            return None, warnings

    async def list_instances(
        self,
        service_name: str | None = None,
        include_unhealthy: bool = False,
        correlation_id: UUID | None = None,
    ) -> tuple[list[ModelRegistryInstanceView], list[ModelWarning]]:
        """List live Consul service instances.

        Args:
            service_name: Optional service name filter. If not provided,
                queries all services from the Consul catalog.
            include_unhealthy: Whether to include unhealthy instances.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Tuple of (instances, warnings).
        """
        correlation_id = correlation_id or uuid4()
        warnings: list[ModelWarning] = []
        instances: list[ModelRegistryInstanceView] = []

        if self._consul_handler is None:
            warnings.append(
                ModelWarning(
                    source="consul",
                    message="Consul handler not configured",
                    code="NO_CONSUL_HANDLER",
                    timestamp=datetime.now(UTC),
                )
            )
            return instances, warnings

        try:
            # Determine which services to query
            service_names_to_query: list[str] = []

            if service_name:
                # Single service specified
                service_names_to_query = [service_name]
            else:
                # Get all service names from Consul catalog
                try:
                    all_services = await self._consul_handler.list_all_services(
                        correlation_id=correlation_id,
                    )
                    service_names_to_query = list(all_services.keys())
                except Exception as e:
                    logger.warning(
                        "Failed to list all services, falling back to empty discovery",
                        extra={
                            "error": str(e),
                            "correlation_id": str(correlation_id),
                        },
                    )
                    warnings.append(
                        ModelWarning(
                            source="consul",
                            message=f"Failed to list all services: {type(e).__name__}",
                            code="CONSUL_CATALOG_FAILED",
                            timestamp=datetime.now(UTC),
                        )
                    )
                    return instances, warnings

            # Query each service for its instances
            for svc_name in service_names_to_query:
                try:
                    service_instances = (
                        await self._consul_handler.get_all_service_instances(
                            service_name=svc_name,
                            include_unhealthy=include_unhealthy,
                            correlation_id=correlation_id,
                        )
                    )

                    for svc in service_instances:
                        # Map EnumHealthStatus to API health_status string
                        health_status: str
                        if svc.health_status == EnumHealthStatus.HEALTHY:
                            health_status = "passing"
                        elif svc.health_status == EnumHealthStatus.UNHEALTHY:
                            health_status = "critical"
                        else:
                            health_status = "unknown"

                        instances.append(
                            ModelRegistryInstanceView(
                                node_id=svc.service_id,
                                service_name=svc.service_name,
                                service_id=svc.service_id,
                                instance_id=svc.service_id,
                                address=svc.address or "unknown",
                                port=svc.port or 0,
                                health_status=health_status,  # type: ignore[arg-type]
                                health_output=svc.health_output,
                                last_check_at=svc.last_check_at or svc.registered_at,
                                tags=list(svc.tags),
                                meta=svc.metadata,
                            )
                        )

                except Exception as e:
                    # Log but continue with other services (partial success)
                    logger.warning(
                        "Failed to query service instances",
                        extra={
                            "service_name": svc_name,
                            "error": str(e),
                            "correlation_id": str(correlation_id),
                        },
                    )
                    warnings.append(
                        ModelWarning(
                            source="consul",
                            message=f"Failed to query service '{svc_name}': {type(e).__name__}",
                            code="CONSUL_SERVICE_QUERY_FAILED",
                            timestamp=datetime.now(UTC),
                        )
                    )

        except Exception as e:
            logger.exception(
                "Failed to discover services",
                extra={"correlation_id": str(correlation_id)},
            )
            warnings.append(
                ModelWarning(
                    source="consul",
                    message=f"Failed to discover services: {type(e).__name__}",
                    code="CONSUL_QUERY_FAILED",
                    timestamp=datetime.now(UTC),
                )
            )

        return instances, warnings

    async def get_discovery(
        self,
        limit: int = 100,
        offset: int = 0,
        correlation_id: UUID | None = None,
    ) -> ModelRegistryDiscoveryResponse:
        """Get full dashboard payload with nodes, instances, and summary.

        This is the primary endpoint for dashboard consumption, providing
        all needed data in a single request.

        Args:
            limit: Maximum number of nodes to return.
            offset: Number of nodes to skip for pagination.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Complete discovery response with all data and any warnings.
        """
        correlation_id = correlation_id or uuid4()
        all_warnings: list[ModelWarning] = []

        # Fetch nodes
        nodes, pagination, node_warnings = await self.list_nodes(
            limit=limit,
            offset=offset,
            correlation_id=correlation_id,
        )
        all_warnings.extend(node_warnings)

        # Fetch instances
        instances, instance_warnings = await self.list_instances(
            include_unhealthy=True,
            correlation_id=correlation_id,
        )
        all_warnings.extend(instance_warnings)

        # Build summary
        by_node_type: dict[str, int] = {}
        by_state: dict[str, int] = {}
        active_count = 0

        for node in nodes:
            by_node_type[node.node_type] = by_node_type.get(node.node_type, 0) + 1
            by_state[node.state] = by_state.get(node.state, 0) + 1
            if node.state == "active":
                active_count += 1

        healthy_count = sum(1 for i in instances if i.health_status == "passing")
        unhealthy_count = len(instances) - healthy_count

        summary = ModelRegistrySummary(
            total_nodes=pagination.total,
            active_nodes=active_count,
            healthy_instances=healthy_count,
            unhealthy_instances=unhealthy_count,
            by_node_type=by_node_type,
            by_state=by_state,
        )

        return ModelRegistryDiscoveryResponse(
            timestamp=datetime.now(UTC),
            warnings=all_warnings,
            summary=summary,
            nodes=nodes,
            live_instances=instances,
            pagination=pagination,
        )

    def get_widget_mapping(
        self,
    ) -> tuple[ModelWidgetMapping | None, list[ModelWarning]]:
        """Load and return widget mapping configuration.

        Returns cached mapping if available and file unchanged, otherwise
        loads from YAML file.

        The cache is automatically invalidated when the file's modification
        time changes, enabling hot-reload of widget mappings without restart.

        Returns:
            Tuple of (widget_mapping or None, warnings).
        """
        warnings: list[ModelWarning] = []

        # Check if file has been modified since last cache
        current_mtime: float | None = None
        try:
            current_mtime = self._widget_mapping_path.stat().st_mtime
            if (
                self._widget_mapping_cache is not None
                and self._widget_mapping_mtime == current_mtime
            ):
                return self._widget_mapping_cache, warnings
        except OSError:
            # File doesn't exist or can't be accessed - will be handled below
            pass

        # Log cache invalidation due to file change (only when cache existed)
        if self._widget_mapping_cache is not None and current_mtime is not None:
            logger.info(
                "Widget mapping cache invalidated, reloading from file",
                extra={
                    "widget_mapping_path": str(self._widget_mapping_path),
                    "old_mtime": self._widget_mapping_mtime,
                    "new_mtime": current_mtime,
                },
            )

        if not self._widget_mapping_path.exists():
            warnings.append(
                ModelWarning(
                    source="config",
                    message=f"Widget mapping file not found: {self._widget_mapping_path}",
                    code="CONFIG_NOT_FOUND",
                    timestamp=datetime.now(UTC),
                )
            )
            return None, warnings

        try:
            with open(self._widget_mapping_path) as f:
                data = yaml.safe_load(f)

            # Parse capability mappings
            capability_mappings: dict[str, ModelCapabilityWidgetMapping] = {}
            for key, value in data.get("capability_mappings", {}).items():
                capability_mappings[key] = ModelCapabilityWidgetMapping(
                    widget_type=value.get("widget_type", "info_card"),
                    defaults=ModelWidgetDefaults(**value.get("defaults", {})),
                )

            # Parse semantic mappings
            semantic_mappings: dict[str, ModelCapabilityWidgetMapping] = {}
            for key, value in data.get("semantic_mappings", {}).items():
                semantic_mappings[key] = ModelCapabilityWidgetMapping(
                    widget_type=value.get("widget_type", "info_card"),
                    defaults=ModelWidgetDefaults(**value.get("defaults", {})),
                )

            # Parse fallback
            fallback_data = data.get("fallback", {})
            fallback = ModelCapabilityWidgetMapping(
                widget_type=fallback_data.get("widget_type", "info_card"),
                defaults=ModelWidgetDefaults(**fallback_data.get("defaults", {})),
            )

            self._widget_mapping_cache = ModelWidgetMapping(
                version=data.get("version", "1.0.0"),
                capability_mappings=capability_mappings,
                semantic_mappings=semantic_mappings,
                fallback=fallback,
            )
            self._widget_mapping_mtime = current_mtime

            logger.debug(
                "Widget mapping loaded",
                extra={
                    "widget_mapping_path": str(self._widget_mapping_path),
                    "mtime": current_mtime,
                    "version": data.get("version", "1.0.0"),
                },
            )

            return self._widget_mapping_cache, warnings

        except Exception as e:
            logger.exception(
                "Failed to load widget mapping",
                extra={"path": str(self._widget_mapping_path)},
            )
            warnings.append(
                ModelWarning(
                    source="config",
                    message=f"Failed to load widget mapping: {type(e).__name__}",
                    code="CONFIG_LOAD_FAILED",
                    timestamp=datetime.now(UTC),
                )
            )
            return None, warnings

    async def health_check(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelRegistryHealthResponse:
        """Perform health check on all backend components.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Health check response with component statuses.
        """
        correlation_id = correlation_id or uuid4()
        components: dict[str, JsonType] = {}
        overall_healthy = True

        # Check projection reader
        if self._projection_reader is None:
            components["postgres"] = {
                "healthy": False,
                "message": "Not configured",
            }
            overall_healthy = False
        else:
            try:
                # Simple query to verify connection
                await self._projection_reader.count_by_state(
                    correlation_id=correlation_id,
                )
                components["postgres"] = {
                    "healthy": True,
                    "message": "Connected",
                }
            except Exception as e:
                components["postgres"] = {
                    "healthy": False,
                    "message": f"Error: {type(e).__name__}",
                }
                overall_healthy = False

        # Check Consul handler
        if self._consul_handler is None:
            components["consul"] = {
                "healthy": False,
                "message": "Not configured",
            }
            overall_healthy = False
        else:
            try:
                result = await self._consul_handler.health_check(
                    correlation_id=correlation_id,
                )
                components["consul"] = {
                    "healthy": result.healthy,
                    "message": result.reason,
                }
                if not result.healthy:
                    overall_healthy = False
            except Exception as e:
                components["consul"] = {
                    "healthy": False,
                    "message": f"Error: {type(e).__name__}",
                }
                overall_healthy = False

        # Check widget mapping
        _, mapping_warnings = self.get_widget_mapping()
        if mapping_warnings:
            components["config"] = {
                "healthy": False,
                "message": mapping_warnings[0].message,
            }
        else:
            components["config"] = {
                "healthy": True,
                "message": "Loaded",
            }

        # Determine overall status
        unhealthy_count = sum(
            1
            for c in components.values()
            if isinstance(c, dict) and not c.get("healthy", False)
        )
        if unhealthy_count == 0:
            status = "healthy"
        elif unhealthy_count < len(components):
            status = "degraded"
        else:
            status = "unhealthy"

        return ModelRegistryHealthResponse(
            status=status,  # type: ignore[arg-type]
            timestamp=datetime.now(UTC),
            components=components,
            version="1.0.0",
        )


__all__ = ["ServiceRegistryDiscovery"]
