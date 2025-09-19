"""Service-specific health details models implementing ProtocolHealthDetails."""

from omnibase_infra.models.core.health.services.model_circuit_breaker_health_details import (
    ModelCircuitBreakerHealthDetails,
)
from omnibase_infra.models.core.health.services.model_kafka_health_details import (
    ModelKafkaHealthDetails,
)
from omnibase_infra.models.core.health.services.model_postgres_health_details import (
    ModelPostgresHealthDetails,
)
from omnibase_infra.models.core.health.services.model_system_health_details import (
    ModelSystemHealthDetails,
)

__all__ = [
    "ModelCircuitBreakerHealthDetails",
    "ModelKafkaHealthDetails",
    "ModelPostgresHealthDetails",
    "ModelSystemHealthDetails",
]
