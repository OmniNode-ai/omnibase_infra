"""Infrastructure Health Monitoring Service.

Centralized health monitoring service for PostgreSQL-RedPanda event bus integration.
Aggregates health statistics from connection pools, circuit breakers, and provides
Prometheus metrics integration for production monitoring.

Following ONEX infrastructure monitoring patterns with strongly typed models.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.core.errors.onex_error import CoreErrorCode

from .postgres_connection_manager import PostgresConnectionManager, get_connection_manager
from .kafka_producer_pool import KafkaProducerPool, get_producer_pool
from .event_bus_circuit_breaker import EventBusCircuitBreaker
from ..models.infrastructure.model_circuit_breaker_environment_config import ModelCircuitBreakerEnvironmentConfig


@dataclass
class InfrastructureHealthMetrics:
    """Aggregated infrastructure health metrics."""
    
    # Overall health
    overall_status: str  # healthy, degraded, unhealthy
    timestamp: float
    environment: str
    
    # Component statuses
    postgres_healthy: bool
    kafka_healthy: bool
    circuit_breaker_healthy: bool
    
    # Detailed metrics
    postgres_metrics: Dict[str, Any]
    kafka_metrics: Dict[str, Any]
    circuit_breaker_metrics: Dict[str, Any]
    
    # Aggregate statistics
    total_connections: int
    total_messages_processed: int
    total_events_queued: int
    error_rate_percent: float
    
    # Performance indicators
    avg_db_response_time_ms: float
    avg_kafka_throughput_mps: float
    circuit_breaker_success_rate: float


class InfrastructureHealthMonitor:
    """
    Centralized infrastructure health monitoring service.
    
    Provides:
    - Aggregated health status from all infrastructure components
    - Prometheus metrics integration
    - Alerting threshold management
    - Environment-specific monitoring configuration
    - Real-time health status reporting
    """
    
    def __init__(self, environment: Optional[str] = None):
        """Initialize infrastructure health monitor.
        
        Args:
            environment: Target environment for configuration (optional, auto-detected if None)
        """
        self.environment = environment or self._detect_environment()
        self.logger = logging.getLogger(f"{__name__}.InfrastructureHealthMonitor")
        
        # Component references (lazy loaded)
        self._postgres_manager: Optional[PostgresConnectionManager] = None
        self._kafka_pool: Optional[KafkaProducerPool] = None
        self._circuit_breaker: Optional[EventBusCircuitBreaker] = None
        
        # Monitoring configuration
        self.monitoring_interval_seconds = 30
        self.health_history: List[InfrastructureHealthMetrics] = []
        self.max_history_size = 100
        
        # Alerting thresholds (environment-specific)
        self.alerting_thresholds = self._get_alerting_thresholds()
        
        # State tracking
        self.last_health_check: Optional[float] = None
        self.consecutive_failures = 0
        self.is_monitoring = False
        
    def _detect_environment(self) -> str:
        """Detect current deployment environment."""
        env_vars = ["ENVIRONMENT", "ENV", "DEPLOYMENT_ENV", "NODE_ENV", "OMNIBASE_ENV"]
        for var in env_vars:
            value = os.getenv(var)
            if value:
                return value.lower()
        return "development"
    
    def _get_alerting_thresholds(self) -> Dict[str, Any]:
        """Get environment-specific alerting thresholds."""
        thresholds = {
            "production": {
                "error_rate_threshold": 1.0,  # 1% error rate triggers alert
                "response_time_threshold": 500.0,  # 500ms average response time
                "connection_utilization_threshold": 80.0,  # 80% pool utilization
                "circuit_breaker_failure_threshold": 0.5,  # 0.5% circuit breaker failures
                "queue_size_threshold": 500  # 500 queued events
            },
            "staging": {
                "error_rate_threshold": 2.0,  # 2% error rate
                "response_time_threshold": 1000.0,  # 1s response time
                "connection_utilization_threshold": 85.0,  # 85% utilization
                "circuit_breaker_failure_threshold": 1.0,  # 1% failures
                "queue_size_threshold": 200  # 200 queued events
            },
            "development": {
                "error_rate_threshold": 5.0,  # 5% error rate
                "response_time_threshold": 2000.0,  # 2s response time
                "connection_utilization_threshold": 90.0,  # 90% utilization
                "circuit_breaker_failure_threshold": 2.0,  # 2% failures
                "queue_size_threshold": 50  # 50 queued events
            }
        }
        return thresholds.get(self.environment, thresholds["development"])
    
    async def get_comprehensive_health_status(self) -> InfrastructureHealthMetrics:
        """Get comprehensive infrastructure health status.
        
        Returns:
            Aggregated infrastructure health metrics
        """
        timestamp = time.time()
        
        try:
            # Collect health metrics from all components
            postgres_health = await self._get_postgres_health()
            kafka_health = await self._get_kafka_health()
            circuit_breaker_health = await self._get_circuit_breaker_health()
            
            # Determine component health status
            postgres_healthy = postgres_health.get("status") == "healthy"
            kafka_healthy = kafka_health.get("status") == "healthy"
            circuit_breaker_healthy = circuit_breaker_health.get("is_healthy", False)
            
            # Calculate aggregate metrics
            total_connections = (
                postgres_health.get("connection_pool", {}).get("size", 0) +
                kafka_health.get("pool_stats", {}).get("total_producers", 0)
            )
            
            total_messages = (
                postgres_health.get("performance", {}).get("total_queries", 0) +
                kafka_health.get("pool_stats", {}).get("total_messages_sent", 0)
            )
            
            total_queued = circuit_breaker_health.get("queued_events", 0)
            
            # Calculate error rates
            postgres_errors = postgres_health.get("performance", {}).get("failed_connections", 0)
            kafka_errors = kafka_health.get("pool_stats", {}).get("total_messages_failed", 0)
            circuit_breaker_errors = circuit_breaker_health.get("metrics", {}).get("failed_events", 0)
            
            total_operations = max(
                total_messages + postgres_errors + kafka_errors + circuit_breaker_errors, 1
            )
            error_rate = ((postgres_errors + kafka_errors + circuit_breaker_errors) / total_operations) * 100
            
            # Performance indicators
            avg_db_response = postgres_health.get("performance", {}).get("average_response_time_ms", 0.0)
            avg_kafka_throughput = kafka_health.get("pool_stats", {}).get("average_throughput_mps", 0.0)
            circuit_success_rate = circuit_breaker_health.get("metrics", {}).get("success_rate", 100.0)
            
            # Determine overall health status
            overall_status = self._determine_overall_health(
                postgres_healthy, kafka_healthy, circuit_breaker_healthy, error_rate
            )
            
            # Create health metrics
            metrics = InfrastructureHealthMetrics(
                overall_status=overall_status,
                timestamp=timestamp,
                environment=self.environment,
                postgres_healthy=postgres_healthy,
                kafka_healthy=kafka_healthy,
                circuit_breaker_healthy=circuit_breaker_healthy,
                postgres_metrics=postgres_health,
                kafka_metrics=kafka_health,
                circuit_breaker_metrics=circuit_breaker_health,
                total_connections=total_connections,
                total_messages_processed=total_messages,
                total_events_queued=total_queued,
                error_rate_percent=error_rate,
                avg_db_response_time_ms=avg_db_response,
                avg_kafka_throughput_mps=avg_kafka_throughput,
                circuit_breaker_success_rate=circuit_success_rate
            )
            
            # Store in history for trend analysis
            self._add_to_history(metrics)
            
            self.last_health_check = timestamp
            self.consecutive_failures = 0
            
            return metrics
            
        except Exception as e:
            self.consecutive_failures += 1
            self.logger.error(f"Health check failed (consecutive failures: {self.consecutive_failures}): {e}")
            
            # Return degraded health status on failure
            return InfrastructureHealthMetrics(
                overall_status="unhealthy",
                timestamp=timestamp,
                environment=self.environment,
                postgres_healthy=False,
                kafka_healthy=False,
                circuit_breaker_healthy=False,
                postgres_metrics={"error": str(e)},
                kafka_metrics={"error": str(e)},
                circuit_breaker_metrics={"error": str(e)},
                total_connections=0,
                total_messages_processed=0,
                total_events_queued=0,
                error_rate_percent=100.0,
                avg_db_response_time_ms=0.0,
                avg_kafka_throughput_mps=0.0,
                circuit_breaker_success_rate=0.0
            )
    
    async def _get_postgres_health(self) -> Dict[str, Any]:
        """Get PostgreSQL connection manager health status."""
        try:
            if not self._postgres_manager:
                self._postgres_manager = get_connection_manager()
            
            return await self._postgres_manager.health_check()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _get_kafka_health(self) -> Dict[str, Any]:
        """Get Kafka producer pool health status."""
        try:
            if not self._kafka_pool:
                self._kafka_pool = get_producer_pool()
            
            health = await self._kafka_pool.health_check()
            
            # Add detailed pool statistics
            pool_stats = self._kafka_pool.get_pool_stats()
            health["pool_stats"] = pool_stats.model_dump()
            
            return health
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _get_circuit_breaker_health(self) -> Dict[str, Any]:
        """Get circuit breaker health status."""
        try:
            if not self._circuit_breaker:
                # Create environment-specific circuit breaker if not exists
                env_config = ModelCircuitBreakerEnvironmentConfig.create_default_config()
                self._circuit_breaker = EventBusCircuitBreaker.from_environment(
                    env_config, self.environment
                )
            
            return self._circuit_breaker.get_health_status()
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    def _determine_overall_health(
        self, 
        postgres_healthy: bool, 
        kafka_healthy: bool, 
        circuit_breaker_healthy: bool, 
        error_rate: float
    ) -> str:
        """Determine overall infrastructure health status."""
        # Check if all components are healthy
        if postgres_healthy and kafka_healthy and circuit_breaker_healthy:
            # Check error rate against threshold
            if error_rate <= self.alerting_thresholds["error_rate_threshold"]:
                return "healthy"
            else:
                return "degraded"
        
        # Check if critical components are working
        elif postgres_healthy and (kafka_healthy or circuit_breaker_healthy):
            return "degraded"
        
        # Critical failure if PostgreSQL is down
        else:
            return "unhealthy"
    
    def _add_to_history(self, metrics: InfrastructureHealthMetrics) -> None:
        """Add metrics to historical tracking."""
        self.health_history.append(metrics)
        
        # Maintain history size limit
        if len(self.health_history) > self.max_history_size:
            self.health_history.pop(0)
    
    def get_health_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get health trends for the specified time period.
        
        Args:
            hours: Number of hours of history to analyze
            
        Returns:
            Trend analysis including error rates, response times, and availability
        """
        if not self.health_history:
            return {"error": "No health history available"}
        
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.health_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": f"No health data available for the last {hours} hour(s)"}
        
        # Calculate trends
        total_checks = len(recent_metrics)
        healthy_checks = len([m for m in recent_metrics if m.overall_status == "healthy"])
        degraded_checks = len([m for m in recent_metrics if m.overall_status == "degraded"])
        unhealthy_checks = len([m for m in recent_metrics if m.overall_status == "unhealthy"])
        
        avg_error_rate = sum(m.error_rate_percent for m in recent_metrics) / total_checks
        avg_response_time = sum(m.avg_db_response_time_ms for m in recent_metrics) / total_checks
        avg_throughput = sum(m.avg_kafka_throughput_mps for m in recent_metrics) / total_checks
        
        return {
            "period_hours": hours,
            "total_checks": total_checks,
            "availability": {
                "healthy_percent": (healthy_checks / total_checks) * 100,
                "degraded_percent": (degraded_checks / total_checks) * 100,
                "unhealthy_percent": (unhealthy_checks / total_checks) * 100
            },
            "performance": {
                "average_error_rate_percent": avg_error_rate,
                "average_response_time_ms": avg_response_time,
                "average_throughput_mps": avg_throughput
            },
            "current_status": recent_metrics[-1].overall_status if recent_metrics else "unknown"
        }
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus metrics format for scraping.
        
        Returns:
            Prometheus metrics in text format
        """
        if not self.health_history:
            return "# No metrics available\n"
        
        latest = self.health_history[-1]
        
        metrics = [
            "# HELP omnibase_infrastructure_health Overall infrastructure health status (1=healthy, 0.5=degraded, 0=unhealthy)",
            "# TYPE omnibase_infrastructure_health gauge",
            f'omnibase_infrastructure_health{{environment="{self.environment}"}} {1.0 if latest.overall_status == "healthy" else 0.5 if latest.overall_status == "degraded" else 0.0}',
            "",
            "# HELP omnibase_postgres_connections PostgreSQL connection pool size",
            "# TYPE omnibase_postgres_connections gauge", 
            f'omnibase_postgres_connections{{environment="{self.environment}"}} {latest.postgres_metrics.get("connection_pool", {}).get("size", 0)}',
            "",
            "# HELP omnibase_kafka_producers Kafka producer pool size",
            "# TYPE omnibase_kafka_producers gauge",
            f'omnibase_kafka_producers{{environment="{self.environment}"}} {latest.kafka_metrics.get("pool_stats", {}).get("total_producers", 0)}',
            "",
            "# HELP omnibase_error_rate_percent Infrastructure error rate percentage",
            "# TYPE omnibase_error_rate_percent gauge",
            f'omnibase_error_rate_percent{{environment="{self.environment}"}} {latest.error_rate_percent}',
            "",
            "# HELP omnibase_messages_processed_total Total messages processed",
            "# TYPE omnibase_messages_processed_total counter",
            f'omnibase_messages_processed_total{{environment="{self.environment}"}} {latest.total_messages_processed}',
            "",
            "# HELP omnibase_events_queued Current number of queued events", 
            "# TYPE omnibase_events_queued gauge",
            f'omnibase_events_queued{{environment="{self.environment}"}} {latest.total_events_queued}',
            "",
            "# HELP omnibase_response_time_ms Average database response time in milliseconds",
            "# TYPE omnibase_response_time_ms gauge",
            f'omnibase_response_time_ms{{environment="{self.environment}"}} {latest.avg_db_response_time_ms}',
            ""
        ]
        
        return "\n".join(metrics)
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.logger.info(f"Starting infrastructure health monitoring (interval: {self.monitoring_interval_seconds}s)")
        
        while self.is_monitoring:
            try:
                await self.get_comprehensive_health_status()
                await asyncio.sleep(self.monitoring_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.is_monitoring = False
        self.logger.info("Infrastructure health monitoring stopped")


# Global health monitor instance
_health_monitor: Optional[InfrastructureHealthMonitor] = None


def get_health_monitor(environment: Optional[str] = None) -> InfrastructureHealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = InfrastructureHealthMonitor(environment)
    return _health_monitor