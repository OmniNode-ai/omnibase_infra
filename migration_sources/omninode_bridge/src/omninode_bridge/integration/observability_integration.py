"""Integration example showing how to use all observability features together."""

import os
from typing import Any

from fastapi import FastAPI

from ..health import (
    HealthChecker,
    HealthCheckType,
    check_disk_space,
    check_memory_usage,
)
from ..health.infrastructure_checks import (
    check_external_service_health,
    check_kafka_health,
    check_postgresql_health,
)
from ..middleware import add_request_correlation_middleware
from ..monitoring import initialize_production_monitoring, start_alert_monitoring_loop
from ..security.audit_logger import AuditEventType, AuditSeverity, get_audit_logger
from ..tracing import initialize_opentelemetry


async def initialize_comprehensive_observability(
    app: FastAPI,
    service_name: str,
    service_version: str = "1.0.0",
    environment: str = None,
) -> dict[str, Any]:
    """Initialize comprehensive observability for a service.

    This function sets up:
    1. Request correlation tracking
    2. OpenTelemetry distributed tracing
    3. Health checking system
    4. Production monitoring and alerting
    5. Enhanced audit logging

    Args:
        app: FastAPI application instance
        service_name: Name of the service
        service_version: Version of the service
        environment: Environment name (defaults to ENVIRONMENT env var)

    Returns:
        Dictionary with initialized components
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    # 1. Initialize request correlation middleware
    add_request_correlation_middleware(
        app=app,
        service_name=service_name,
        generate_request_id=True,
        log_requests=True,
        propagate_headers=True,
    )

    # 2. Initialize OpenTelemetry distributed tracing
    tracing_initialized = initialize_opentelemetry(
        service_name=service_name,
        service_version=service_version,
    )

    # 3. Initialize health checking system
    health_checker = HealthChecker(
        service_name=service_name,
        service_version=service_version,
        environment=environment,
        timeout_seconds=30.0,
        enable_metrics=True,
    )

    # Register default system health checks
    health_checker.register_liveness_check(
        name="memory_usage",
        check_func=lambda: check_memory_usage(threshold_percent=90.0),
    )

    health_checker.register_liveness_check(
        name="disk_space",
        check_func=lambda: check_disk_space(threshold_percent=90.0),
    )

    # 4. Initialize production monitoring and alerting
    monitoring_config = initialize_production_monitoring(
        service_name=service_name,
        environment=environment,
        enable_custom_metrics=True,
    )

    # 5. Initialize audit logger
    audit_logger = get_audit_logger(service_name, service_version)

    # Log initialization
    audit_logger.log_event(
        event_type=AuditEventType.SERVICE_STARTUP,
        severity=AuditSeverity.LOW,
        message=f"Service {service_name} v{service_version} started with comprehensive observability",
        additional_data={
            "environment": environment,
            "tracing_enabled": tracing_initialized,
            "health_checks_enabled": True,
            "monitoring_enabled": True,
        },
    )

    return {
        "health_checker": health_checker,
        "monitoring_config": monitoring_config,
        "audit_logger": audit_logger,
        "tracing_initialized": tracing_initialized,
        "service_info": {
            "name": service_name,
            "version": service_version,
            "environment": environment,
        },
    }


def register_infrastructure_health_checks(
    health_checker: HealthChecker,
    postgres_client: Any = None,
    kafka_bootstrap_servers: str = None,
    external_services: dict[str, str] = None,
) -> None:
    """Register infrastructure-specific health checks.

    Args:
        health_checker: HealthChecker instance
        postgres_client: PostgreSQL client instance
        kafka_bootstrap_servers: Kafka bootstrap servers string
        external_services: Dictionary of service_name -> service_url
    """

    # PostgreSQL health check
    if postgres_client:
        health_checker.register_readiness_check(
            name="postgresql",
            check_func=lambda: check_postgresql_health(postgres_client),
        )

    # Kafka health check
    if kafka_bootstrap_servers:
        health_checker.register_readiness_check(
            name="kafka",
            check_func=lambda: check_kafka_health(kafka_bootstrap_servers),
        )

    # External service health checks
    if external_services:
        for service_name, service_url in external_services.items():
            health_checker.register_readiness_check(
                name=f"external_{service_name}",
                check_func=lambda svc_name=service_name, svc_url=service_url: check_external_service_health(
                    svc_name, svc_url
                ),
            )


def add_comprehensive_health_endpoints(
    app: FastAPI, health_checker: HealthChecker
) -> None:
    """Add comprehensive health check endpoints to FastAPI app.

    Args:
        app: FastAPI application instance
        health_checker: HealthChecker instance
    """

    @app.get("/health/live")
    async def liveness_probe():
        """Kubernetes liveness probe endpoint."""
        health_status = await health_checker.get_service_health(
            check_types=[HealthCheckType.LIVENESS]
        )

        if health_status.status.value in ["healthy", "degraded"]:
            return health_status.to_dict()
        else:
            # Return 503 for unhealthy status
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail=health_status.to_dict())

    @app.get("/health/ready")
    async def readiness_probe():
        """Kubernetes readiness probe endpoint."""
        health_status = await health_checker.get_service_health(
            check_types=[HealthCheckType.READINESS]
        )

        if health_status.status.value == "healthy":
            return health_status.to_dict()
        else:
            # Return 503 for non-healthy status
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail=health_status.to_dict())

    @app.get("/health/startup")
    async def startup_probe():
        """Kubernetes startup probe endpoint."""
        health_status = await health_checker.get_service_health(
            check_types=[HealthCheckType.STARTUP]
        )

        if health_status.status.value in ["healthy", "degraded"]:
            return health_status.to_dict()
        else:
            # Return 503 for unhealthy status
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail=health_status.to_dict())

    @app.get("/health")
    async def comprehensive_health():
        """Comprehensive health check endpoint."""
        health_status = await health_checker.get_service_health()
        return health_status.to_dict()

    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        from ..monitoring import get_monitoring_config

        monitoring_config = get_monitoring_config()

        if monitoring_config:
            return monitoring_config.get_prometheus_metrics()
        else:
            return "# No metrics available\n"

    @app.get("/alerts")
    async def alert_status():
        """Alert status endpoint."""
        from ..monitoring import get_monitoring_config

        monitoring_config = get_monitoring_config()

        if monitoring_config:
            return monitoring_config.get_alert_status()
        else:
            return {"alerting_enabled": False, "message": "Monitoring not initialized"}


# Example usage function
async def setup_service_observability_example(
    app: FastAPI,
    service_name: str = "omninode-bridge-example",
    postgres_client: Any = None,
    kafka_bootstrap_servers: str = None,
) -> None:
    """Complete example of setting up observability for a service.

    Args:
        app: FastAPI application instance
        service_name: Name of the service
        postgres_client: PostgreSQL client (optional)
        kafka_bootstrap_servers: Kafka servers (optional)
    """

    # Initialize comprehensive observability
    observability_components = await initialize_comprehensive_observability(
        app=app,
        service_name=service_name,
        service_version="1.0.0",
    )

    health_checker = observability_components["health_checker"]
    audit_logger = observability_components["audit_logger"]

    # Register infrastructure health checks
    external_services = {
        "example_api": "https://api.example.com",
    }

    register_infrastructure_health_checks(
        health_checker=health_checker,
        postgres_client=postgres_client,
        kafka_bootstrap_servers=kafka_bootstrap_servers,
        external_services=external_services,
    )

    # Add health endpoints
    add_comprehensive_health_endpoints(app, health_checker)

    # Start alert monitoring in background
    import asyncio

    asyncio.create_task(start_alert_monitoring_loop(interval_seconds=60))

    # Log successful setup
    audit_logger.log_event(
        event_type=audit_logger.AuditEventType.SERVICE_STARTUP,
        severity=audit_logger.AuditSeverity.LOW,
        message=f"Comprehensive observability setup completed for {service_name}",
        additional_data={
            "health_checks_registered": len(health_checker._liveness_checks)
            + len(health_checker._readiness_checks),
            "infrastructure_components": (
                ["postgresql", "kafka", "external_services"]
                if postgres_client and kafka_bootstrap_servers
                else []
            ),
        },
    )


# Docker Compose configuration example for local development
DOCKER_COMPOSE_MONITORING_STACK = """
# Add this to your docker-compose.yml for local monitoring stack
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:1.50
    ports:
      - "16686:16686"      # Jaeger UI
      - "4317:4317"        # OTLP gRPC receiver
      - "4318:4318"        # OTLP HTTP receiver
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  prometheus:
    image: prom/prometheus:v2.47.0
    ports:
      - "9090:9090"        # Prometheus UI
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:10.1.0
    ports:
      - "3000:3000"        # Grafana UI
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
"""


# Kubernetes deployment configuration example
KUBERNETES_MONITORING_CONFIG = """
# Add these environment variables to your Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: omninode-bridge-service
spec:
  template:
    spec:
      containers:
      - name: app
        env:
        # OpenTelemetry configuration
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://jaeger-collector:4317"
        - name: ENABLE_OTEL_TRACING
          value: "true"
        - name: TRACE_SAMPLING_RATE
          value: "0.1"  # 10% sampling in production

        # Monitoring configuration
        - name: ENABLE_ALERTING
          value: "true"
        - name: ALERT_WEBHOOK_URL
          value: "https://alerts.example.com/webhook"
        - name: METRICS_PORT
          value: "9090"

        # Health check configuration
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          failureThreshold: 30
"""
