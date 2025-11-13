"""
Production pattern generators for Phase 2 code generation.

This package provides pattern libraries for generating production-ready code:
- Health checks (comprehensive monitoring)
- Consul integration (service discovery)
- OnexEnvelopeV1 events (production event format)
- Metrics collection (observability)
- Lifecycle management (startup/shutdown)

Each pattern generator produces working implementations that can be directly
integrated into generated node code, reducing manual completion from 50% → 10%.
"""

# Consul integration patterns
from .consul_integration import (
    ConsulPatternGenerator,
    ConsulRegistrationConfig,
    generate_consul_deregistration,
    generate_consul_discovery,
    generate_consul_registration,
)

# Event publishing patterns (Workstream 3)
from .event_publishing import (
    EventPublishingPatternGenerator,
    generate_event_publishing_methods,
    generate_operation_completed_event,
    generate_operation_failed_event,
    generate_operation_started_event,
    get_event_type_catalog,
)

# Health check patterns
from .health_checks import (
    HealthCheckGenerator,
    generate_consul_health_check,
    generate_database_health_check,
    generate_health_check_method,
    generate_http_service_health_check,
    generate_kafka_health_check,
    generate_self_health_check,
)

# Lifecycle management patterns (Workstream 5)
from .lifecycle import (
    LifecyclePatternGenerator,
    generate_helper_methods,
    generate_init_method,
    generate_runtime_monitoring,
    generate_shutdown_method,
    generate_startup_method,
)

# Metrics collection patterns (Workstream 4)
from .metrics import (
    MetricsConfiguration,
    generate_business_metrics_tracking,
    generate_complete_metrics_class,
    generate_example_usage,
    generate_metrics_decorator,
    generate_metrics_initialization,
    generate_metrics_publishing,
    generate_operation_metrics_tracking,
    generate_resource_metrics_collection,
)

__all__ = [
    # Consul integration
    "ConsulPatternGenerator",
    "ConsulRegistrationConfig",
    "generate_consul_registration",
    "generate_consul_discovery",
    "generate_consul_deregistration",
    # Event publishing (Workstream 3)
    "EventPublishingPatternGenerator",
    "generate_event_publishing_methods",
    "generate_operation_started_event",
    "generate_operation_completed_event",
    "generate_operation_failed_event",
    "get_event_type_catalog",
    # Lifecycle management (Workstream 5)
    "LifecyclePatternGenerator",
    "generate_init_method",
    "generate_startup_method",
    "generate_shutdown_method",
    "generate_runtime_monitoring",
    "generate_helper_methods",
    # Metrics collection (Workstream 4)
    "MetricsConfiguration",
    "generate_metrics_initialization",
    "generate_operation_metrics_tracking",
    "generate_resource_metrics_collection",
    "generate_business_metrics_tracking",
    "generate_metrics_publishing",
    "generate_metrics_decorator",
    "generate_complete_metrics_class",
    "generate_example_usage",
    # Health checks
    "HealthCheckGenerator",
    "generate_health_check_method",
    "generate_self_health_check",
    "generate_database_health_check",
    "generate_kafka_health_check",
    "generate_consul_health_check",
    "generate_http_service_health_check",
]
