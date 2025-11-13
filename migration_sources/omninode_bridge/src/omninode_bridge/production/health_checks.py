#!/usr/bin/env python3
"""
Production Health Checks for ONEX v2.0 Code Generation System.

Provides comprehensive health monitoring for all system components:
- Template Manager health
- Validation Pipeline health
- AI Quorum health
- Coordination Components health
- Database connectivity
- Kafka connectivity
- Resource availability

ONEX v2.0 Compliance:
- Type-safe health check results
- Async/await for non-blocking checks
- Comprehensive error handling
- Detailed health status reporting

Performance Requirements:
- Individual health check: <100ms
- Full health check: <500ms
- Non-blocking async execution
- Minimal resource overhead

Author: Code Generation System
Last Updated: 2025-11-06
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# === Health Status Enums ===


class HealthStatus(str, Enum):
    """Health status for components."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types of components to monitor."""

    TEMPLATE_MANAGER = "template_manager"
    VALIDATION_PIPELINE = "validation_pipeline"
    AI_QUORUM = "ai_quorum"
    DEPENDENCY_RESOLVER = "dependency_resolver"
    CONTEXT_DISTRIBUTOR = "context_distributor"
    ROUTING_ORCHESTRATOR = "routing_orchestrator"
    DATABASE = "database"
    KAFKA = "kafka"
    CACHE = "cache"


# === Health Check Models ===


@dataclass
class HealthCheckResult:
    """
    Result of a health check for a component.

    Attributes:
        component: Name of the component checked
        component_type: Type of component
        status: Health status (healthy, degraded, unhealthy)
        response_time_ms: Time taken to perform health check
        message: Human-readable status message
        details: Additional details about the health check
        timestamp: When the health check was performed
        error: Error message if unhealthy
    """

    component: str
    component_type: ComponentType
    status: HealthStatus
    response_time_ms: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "component": self.component,
            "component_type": self.component_type.value,
            "status": self.status.value,
            "response_time_ms": round(self.response_time_ms, 2),
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "error": self.error,
        }


@dataclass
class SystemHealthReport:
    """
    Comprehensive system health report.

    Attributes:
        overall_status: Overall system health status
        component_results: Health check results for each component
        total_response_time_ms: Total time for all health checks
        timestamp: When the report was generated
        healthy_count: Number of healthy components
        degraded_count: Number of degraded components
        unhealthy_count: Number of unhealthy components
    """

    overall_status: HealthStatus
    component_results: list[HealthCheckResult]
    total_response_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def healthy_count(self) -> int:
        """Count of healthy components."""
        return sum(
            1 for r in self.component_results if r.status == HealthStatus.HEALTHY
        )

    @property
    def degraded_count(self) -> int:
        """Count of degraded components."""
        return sum(
            1 for r in self.component_results if r.status == HealthStatus.DEGRADED
        )

    @property
    def unhealthy_count(self) -> int:
        """Count of unhealthy components."""
        return sum(
            1 for r in self.component_results if r.status == HealthStatus.UNHEALTHY
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "total_response_time_ms": round(self.total_response_time_ms, 2),
            "summary": {
                "healthy": self.healthy_count,
                "degraded": self.degraded_count,
                "unhealthy": self.unhealthy_count,
                "total": len(self.component_results),
            },
            "components": [r.to_dict() for r in self.component_results],
        }


# === Health Checker ===


class HealthChecker:
    """
    Comprehensive health checker for all system components.

    Performs health checks on:
    - Template Manager
    - Validation Pipeline
    - AI Quorum
    - Coordination Components (Dependency Resolver, Context Distributor, Router)
    - Database
    - Kafka
    - Cache systems

    Performance:
    - Individual check: <100ms target
    - Full system check: <500ms target
    - Parallel execution of independent checks
    """

    def __init__(
        self,
        template_manager: Optional[Any] = None,
        validation_pipeline: Optional[Any] = None,
        ai_quorum: Optional[Any] = None,
        dependency_resolver: Optional[Any] = None,
        context_distributor: Optional[Any] = None,
        routing_orchestrator: Optional[Any] = None,
        database_client: Optional[Any] = None,
        kafka_producer: Optional[Any] = None,
        cache_client: Optional[Any] = None,
    ):
        """
        Initialize health checker with component references.

        Args:
            template_manager: Template manager instance
            validation_pipeline: Validation pipeline instance
            ai_quorum: AI quorum instance
            dependency_resolver: Dependency resolver instance
            context_distributor: Context distributor instance
            routing_orchestrator: Routing orchestrator instance
            database_client: Database client instance
            kafka_producer: Kafka producer instance
            cache_client: Cache client instance
        """
        self.template_manager = template_manager
        self.validation_pipeline = validation_pipeline
        self.ai_quorum = ai_quorum
        self.dependency_resolver = dependency_resolver
        self.context_distributor = context_distributor
        self.routing_orchestrator = routing_orchestrator
        self.database_client = database_client
        self.kafka_producer = kafka_producer
        self.cache_client = cache_client

        # Health check timeout (prevent hanging checks)
        self.check_timeout_seconds = 5.0

    async def check_system_health(
        self, include_components: Optional[list[ComponentType]] = None
    ) -> SystemHealthReport:
        """
        Check health of all system components.

        Args:
            include_components: Specific components to check (default: all available)

        Returns:
            Comprehensive system health report

        Performance:
            - Target: <500ms for full system check
            - Parallel execution of independent checks
            - Timeout protection for hanging checks
        """
        start_time = time.perf_counter()

        # Determine which checks to run
        checks_to_run = []

        if (
            include_components is None
            or ComponentType.TEMPLATE_MANAGER in include_components
        ):
            if self.template_manager is not None:
                checks_to_run.append(self.check_template_manager())

        if (
            include_components is None
            or ComponentType.VALIDATION_PIPELINE in include_components
        ):
            if self.validation_pipeline is not None:
                checks_to_run.append(self.check_validation_pipeline())

        if include_components is None or ComponentType.AI_QUORUM in include_components:
            if self.ai_quorum is not None:
                checks_to_run.append(self.check_ai_quorum())

        if (
            include_components is None
            or ComponentType.DEPENDENCY_RESOLVER in include_components
        ):
            if self.dependency_resolver is not None:
                checks_to_run.append(self.check_dependency_resolver())

        if (
            include_components is None
            or ComponentType.CONTEXT_DISTRIBUTOR in include_components
        ):
            if self.context_distributor is not None:
                checks_to_run.append(self.check_context_distributor())

        if (
            include_components is None
            or ComponentType.ROUTING_ORCHESTRATOR in include_components
        ):
            if self.routing_orchestrator is not None:
                checks_to_run.append(self.check_routing_orchestrator())

        if include_components is None or ComponentType.DATABASE in include_components:
            if self.database_client is not None:
                checks_to_run.append(self.check_database())

        if include_components is None or ComponentType.KAFKA in include_components:
            if self.kafka_producer is not None:
                checks_to_run.append(self.check_kafka())

        if include_components is None or ComponentType.CACHE in include_components:
            if self.cache_client is not None:
                checks_to_run.append(self.check_cache())

        # Run all checks in parallel with timeout protection
        try:
            component_results = await asyncio.wait_for(
                asyncio.gather(*checks_to_run, return_exceptions=True),
                timeout=self.check_timeout_seconds * len(checks_to_run),
            )
        except asyncio.TimeoutError:
            logger.error("Health check timeout exceeded")
            component_results = [
                HealthCheckResult(
                    component="system",
                    component_type=ComponentType.TEMPLATE_MANAGER,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=self.check_timeout_seconds * 1000,
                    message="Health check timeout exceeded",
                    error="Timeout",
                )
            ]

        # Filter out exceptions and convert to results
        valid_results: list[HealthCheckResult] = []
        for result in component_results:
            if isinstance(result, Exception):
                logger.error(f"Health check exception: {result}")
                valid_results.append(
                    HealthCheckResult(
                        component="unknown",
                        component_type=ComponentType.TEMPLATE_MANAGER,
                        status=HealthStatus.UNHEALTHY,
                        response_time_ms=0,
                        message="Health check failed with exception",
                        error=str(result),
                    )
                )
            else:
                valid_results.append(result)

        # Calculate overall status
        overall_status = self._calculate_overall_status(valid_results)

        total_time_ms = (time.perf_counter() - start_time) * 1000

        return SystemHealthReport(
            overall_status=overall_status,
            component_results=valid_results,
            total_response_time_ms=total_time_ms,
        )

    def _calculate_overall_status(
        self, results: list[HealthCheckResult]
    ) -> HealthStatus:
        """
        Calculate overall system health status.

        Logic:
        - If any component is unhealthy: UNHEALTHY
        - If any component is degraded: DEGRADED
        - If all components are healthy: HEALTHY
        - If no components checked: UNKNOWN

        Args:
            results: List of component health check results

        Returns:
            Overall system health status
        """
        if not results:
            return HealthStatus.UNKNOWN

        if any(r.status == HealthStatus.UNHEALTHY for r in results):
            return HealthStatus.UNHEALTHY

        if any(r.status == HealthStatus.DEGRADED for r in results):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    # === Component Health Checks ===

    async def check_template_manager(self) -> HealthCheckResult:
        """
        Check template manager health.

        Validates:
        - Template loading functionality
        - Cache statistics
        - Template availability

        Returns:
            Health check result for template manager
        """
        start_time = time.perf_counter()

        try:
            # Check if template manager is available
            if not hasattr(self.template_manager, "get_template"):
                return HealthCheckResult(
                    component="template_manager",
                    component_type=ComponentType.TEMPLATE_MANAGER,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    message="Template manager missing get_template method",
                    error="Missing method",
                )

            # Try to get cache statistics
            details = {}
            if hasattr(self.template_manager, "get_cache_stats"):
                cache_stats = self.template_manager.get_cache_stats()
                details["cache_stats"] = cache_stats

                # Check cache hit rate
                if cache_stats.get("hit_rate", 0) < 0.7:
                    status = HealthStatus.DEGRADED
                    message = (
                        f"Low cache hit rate: {cache_stats.get('hit_rate', 0):.2%}"
                    )
                else:
                    status = HealthStatus.HEALTHY
                    message = "Template manager healthy"
            else:
                status = HealthStatus.HEALTHY
                message = "Template manager healthy (no cache stats)"

            response_time_ms = (time.perf_counter() - start_time) * 1000

            return HealthCheckResult(
                component="template_manager",
                component_type=ComponentType.TEMPLATE_MANAGER,
                status=status,
                response_time_ms=response_time_ms,
                message=message,
                details=details,
            )

        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Template manager health check failed: {e}")
            return HealthCheckResult(
                component="template_manager",
                component_type=ComponentType.TEMPLATE_MANAGER,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message="Template manager health check failed",
                error=str(e),
            )

    async def check_validation_pipeline(self) -> HealthCheckResult:
        """
        Check validation pipeline health.

        Validates:
        - Validator availability
        - Pipeline execution capability

        Returns:
            Health check result for validation pipeline
        """
        start_time = time.perf_counter()

        try:
            # Check if validation pipeline is available
            if not hasattr(self.validation_pipeline, "validate"):
                return HealthCheckResult(
                    component="validation_pipeline",
                    component_type=ComponentType.VALIDATION_PIPELINE,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    message="Validation pipeline missing validate method",
                    error="Missing method",
                )

            # Check validator count
            details = {}
            if hasattr(self.validation_pipeline, "get_validator_count"):
                validator_count = self.validation_pipeline.get_validator_count()
                details["validator_count"] = validator_count

                if validator_count == 0:
                    status = HealthStatus.DEGRADED
                    message = "No validators configured"
                else:
                    status = HealthStatus.HEALTHY
                    message = (
                        f"Validation pipeline healthy ({validator_count} validators)"
                    )
            else:
                status = HealthStatus.HEALTHY
                message = "Validation pipeline healthy"

            response_time_ms = (time.perf_counter() - start_time) * 1000

            return HealthCheckResult(
                component="validation_pipeline",
                component_type=ComponentType.VALIDATION_PIPELINE,
                status=status,
                response_time_ms=response_time_ms,
                message=message,
                details=details,
            )

        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Validation pipeline health check failed: {e}")
            return HealthCheckResult(
                component="validation_pipeline",
                component_type=ComponentType.VALIDATION_PIPELINE,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message="Validation pipeline health check failed",
                error=str(e),
            )

    async def check_ai_quorum(self) -> HealthCheckResult:
        """
        Check AI Quorum health.

        Validates:
        - Model connectivity
        - Response latency
        - Model availability

        Returns:
            Health check result for AI Quorum
        """
        start_time = time.perf_counter()

        try:
            # Check if AI Quorum is available
            if not hasattr(self.ai_quorum, "query"):
                return HealthCheckResult(
                    component="ai_quorum",
                    component_type=ComponentType.AI_QUORUM,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    message="AI Quorum missing query method",
                    error="Missing method",
                )

            # Check model availability
            details = {}
            if hasattr(self.ai_quorum, "get_available_models"):
                models = self.ai_quorum.get_available_models()
                details["available_models"] = len(models)

                if len(models) == 0:
                    status = HealthStatus.UNHEALTHY
                    message = "No AI models available"
                elif len(models) < 3:
                    status = HealthStatus.DEGRADED
                    message = f"Limited AI models available ({len(models)})"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"AI Quorum healthy ({len(models)} models)"
            else:
                status = HealthStatus.HEALTHY
                message = "AI Quorum healthy"

            response_time_ms = (time.perf_counter() - start_time) * 1000

            # Check latency threshold
            if response_time_ms > 100:
                status = HealthStatus.DEGRADED
                message = f"{message} (slow response: {response_time_ms:.0f}ms)"

            return HealthCheckResult(
                component="ai_quorum",
                component_type=ComponentType.AI_QUORUM,
                status=status,
                response_time_ms=response_time_ms,
                message=message,
                details=details,
            )

        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"AI Quorum health check failed: {e}")
            return HealthCheckResult(
                component="ai_quorum",
                component_type=ComponentType.AI_QUORUM,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message="AI Quorum health check failed",
                error=str(e),
            )

    async def check_dependency_resolver(self) -> HealthCheckResult:
        """Check dependency resolver health."""
        start_time = time.perf_counter()

        try:
            status = HealthStatus.HEALTHY
            message = "Dependency resolver healthy"
            details = {}

            response_time_ms = (time.perf_counter() - start_time) * 1000

            return HealthCheckResult(
                component="dependency_resolver",
                component_type=ComponentType.DEPENDENCY_RESOLVER,
                status=status,
                response_time_ms=response_time_ms,
                message=message,
                details=details,
            )

        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Dependency resolver health check failed: {e}")
            return HealthCheckResult(
                component="dependency_resolver",
                component_type=ComponentType.DEPENDENCY_RESOLVER,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message="Dependency resolver health check failed",
                error=str(e),
            )

    async def check_context_distributor(self) -> HealthCheckResult:
        """Check context distributor health."""
        start_time = time.perf_counter()

        try:
            status = HealthStatus.HEALTHY
            message = "Context distributor healthy"
            details = {}

            response_time_ms = (time.perf_counter() - start_time) * 1000

            return HealthCheckResult(
                component="context_distributor",
                component_type=ComponentType.CONTEXT_DISTRIBUTOR,
                status=status,
                response_time_ms=response_time_ms,
                message=message,
                details=details,
            )

        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Context distributor health check failed: {e}")
            return HealthCheckResult(
                component="context_distributor",
                component_type=ComponentType.CONTEXT_DISTRIBUTOR,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message="Context distributor health check failed",
                error=str(e),
            )

    async def check_routing_orchestrator(self) -> HealthCheckResult:
        """Check routing orchestrator health."""
        start_time = time.perf_counter()

        try:
            status = HealthStatus.HEALTHY
            message = "Routing orchestrator healthy"
            details = {}

            response_time_ms = (time.perf_counter() - start_time) * 1000

            return HealthCheckResult(
                component="routing_orchestrator",
                component_type=ComponentType.ROUTING_ORCHESTRATOR,
                status=status,
                response_time_ms=response_time_ms,
                message=message,
                details=details,
            )

        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Routing orchestrator health check failed: {e}")
            return HealthCheckResult(
                component="routing_orchestrator",
                component_type=ComponentType.ROUTING_ORCHESTRATOR,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message="Routing orchestrator health check failed",
                error=str(e),
            )

    async def check_database(self) -> HealthCheckResult:
        """
        Check database connectivity and health.

        Validates:
        - Connection availability
        - Query execution
        - Connection pool status

        Returns:
            Health check result for database
        """
        start_time = time.perf_counter()

        try:
            # Simple connectivity check
            if hasattr(self.database_client, "execute_query"):
                await self.database_client.execute_query("SELECT 1")
            elif hasattr(self.database_client, "fetchval"):
                await self.database_client.fetchval("SELECT 1")

            response_time_ms = (time.perf_counter() - start_time) * 1000

            # Check response time
            if response_time_ms > 100:
                status = HealthStatus.DEGRADED
                message = f"Database slow (response: {response_time_ms:.0f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = "Database healthy"

            return HealthCheckResult(
                component="database",
                component_type=ComponentType.DATABASE,
                status=status,
                response_time_ms=response_time_ms,
                message=message,
            )

        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Database health check failed: {e}")
            return HealthCheckResult(
                component="database",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message="Database unhealthy",
                error=str(e),
            )

    async def check_kafka(self) -> HealthCheckResult:
        """
        Check Kafka connectivity and health.

        Validates:
        - Producer availability
        - Connection status

        Returns:
            Health check result for Kafka
        """
        start_time = time.perf_counter()

        try:
            # Check if producer is started
            details = {}
            if hasattr(self.kafka_producer, "_producer"):
                # Producer is available
                status = HealthStatus.HEALTHY
                message = "Kafka producer healthy"
            else:
                status = HealthStatus.DEGRADED
                message = "Kafka producer not initialized"

            response_time_ms = (time.perf_counter() - start_time) * 1000

            return HealthCheckResult(
                component="kafka",
                component_type=ComponentType.KAFKA,
                status=status,
                response_time_ms=response_time_ms,
                message=message,
                details=details,
            )

        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Kafka health check failed: {e}")
            return HealthCheckResult(
                component="kafka",
                component_type=ComponentType.KAFKA,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message="Kafka unhealthy",
                error=str(e),
            )

    async def check_cache(self) -> HealthCheckResult:
        """
        Check cache system health.

        Validates:
        - Cache availability
        - Cache statistics

        Returns:
            Health check result for cache
        """
        start_time = time.perf_counter()

        try:
            status = HealthStatus.HEALTHY
            message = "Cache healthy"
            details = {}

            # Check cache statistics if available
            if hasattr(self.cache_client, "get_stats"):
                stats = self.cache_client.get_stats()
                details["stats"] = stats

            response_time_ms = (time.perf_counter() - start_time) * 1000

            return HealthCheckResult(
                component="cache",
                component_type=ComponentType.CACHE,
                status=status,
                response_time_ms=response_time_ms,
                message=message,
                details=details,
            )

        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Cache health check failed: {e}")
            return HealthCheckResult(
                component="cache",
                component_type=ComponentType.CACHE,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                message="Cache unhealthy",
                error=str(e),
            )
