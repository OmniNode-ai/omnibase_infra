"""Configurable circuit breaker patterns for OmniNode Bridge services."""

import os
from enum import Enum

from circuitbreaker import circuit
from pydantic import BaseModel, Field


class ServiceCriticality(Enum):
    """Service criticality levels for circuit breaker configuration."""

    CRITICAL = "critical"  # Core infrastructure (database, auth)
    HIGH = "high"  # Primary business logic (workflow coordination)
    MEDIUM = "medium"  # Supporting services (monitoring, caching)
    LOW = "low"  # Non-essential services (analytics, logging)


class CircuitBreakerConfig(BaseModel):
    """
    Circuit breaker configuration for different service criticalities.

    Provides validated configuration with sensible constraints for production use.

    Attributes:
        failure_threshold: Number of consecutive failures to open circuit (1-100)
        recovery_timeout: Seconds to wait before attempting recovery (5-600)
        expected_exception: Exception type to trigger circuit breaker

    Example:
        ```python
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=DatabaseError
        )
        ```
    """

    failure_threshold: int = Field(
        ge=1,
        le=100,
        description="Number of consecutive failures required to open circuit (1-100)",
    )
    recovery_timeout: int = Field(
        ge=5,
        le=600,
        description="Seconds to wait before attempting recovery (5-600 seconds, max 10 minutes)",
    )
    expected_exception: type = Field(
        default=Exception, description="Exception type that triggers circuit breaker"
    )

    model_config = {"arbitrary_types_allowed": True}


class ConfigurableCircuitBreaker:
    """Configurable circuit breaker factory based on service criticality."""

    def __init__(self):
        """Initialize with environment-based circuit breaker configurations."""
        self.environment = os.getenv("ENVIRONMENT", "development").lower()
        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load circuit breaker configurations based on environment."""
        if self.environment == "production":
            # Production: More lenient to avoid service disruption
            self.configs = {
                ServiceCriticality.CRITICAL: CircuitBreakerConfig(
                    failure_threshold=int(
                        os.getenv("CB_CRITICAL_FAILURE_THRESHOLD", "5"),
                    ),
                    recovery_timeout=int(
                        os.getenv("CB_CRITICAL_RECOVERY_TIMEOUT", "60"),
                    ),
                ),
                ServiceCriticality.HIGH: CircuitBreakerConfig(
                    failure_threshold=int(os.getenv("CB_HIGH_FAILURE_THRESHOLD", "4")),
                    recovery_timeout=int(os.getenv("CB_HIGH_RECOVERY_TIMEOUT", "45")),
                ),
                ServiceCriticality.MEDIUM: CircuitBreakerConfig(
                    failure_threshold=int(
                        os.getenv("CB_MEDIUM_FAILURE_THRESHOLD", "3"),
                    ),
                    recovery_timeout=int(os.getenv("CB_MEDIUM_RECOVERY_TIMEOUT", "30")),
                ),
                ServiceCriticality.LOW: CircuitBreakerConfig(
                    failure_threshold=int(os.getenv("CB_LOW_FAILURE_THRESHOLD", "2")),
                    recovery_timeout=int(os.getenv("CB_LOW_RECOVERY_TIMEOUT", "15")),
                ),
            }
        elif self.environment == "staging":
            # Staging: Balanced configuration for testing
            self.configs = {
                ServiceCriticality.CRITICAL: CircuitBreakerConfig(
                    failure_threshold=int(
                        os.getenv("CB_CRITICAL_FAILURE_THRESHOLD", "4"),
                    ),
                    recovery_timeout=int(
                        os.getenv("CB_CRITICAL_RECOVERY_TIMEOUT", "45"),
                    ),
                ),
                ServiceCriticality.HIGH: CircuitBreakerConfig(
                    failure_threshold=int(os.getenv("CB_HIGH_FAILURE_THRESHOLD", "3")),
                    recovery_timeout=int(os.getenv("CB_HIGH_RECOVERY_TIMEOUT", "30")),
                ),
                ServiceCriticality.MEDIUM: CircuitBreakerConfig(
                    failure_threshold=int(
                        os.getenv("CB_MEDIUM_FAILURE_THRESHOLD", "3"),
                    ),
                    recovery_timeout=int(os.getenv("CB_MEDIUM_RECOVERY_TIMEOUT", "20")),
                ),
                ServiceCriticality.LOW: CircuitBreakerConfig(
                    failure_threshold=int(os.getenv("CB_LOW_FAILURE_THRESHOLD", "2")),
                    recovery_timeout=int(os.getenv("CB_LOW_RECOVERY_TIMEOUT", "10")),
                ),
            }
        else:  # development
            # Development: More aggressive to catch issues early
            self.configs = {
                ServiceCriticality.CRITICAL: CircuitBreakerConfig(
                    failure_threshold=int(
                        os.getenv("CB_CRITICAL_FAILURE_THRESHOLD", "3"),
                    ),
                    recovery_timeout=int(
                        os.getenv("CB_CRITICAL_RECOVERY_TIMEOUT", "30"),
                    ),
                ),
                ServiceCriticality.HIGH: CircuitBreakerConfig(
                    failure_threshold=int(os.getenv("CB_HIGH_FAILURE_THRESHOLD", "3")),
                    recovery_timeout=int(os.getenv("CB_HIGH_RECOVERY_TIMEOUT", "20")),
                ),
                ServiceCriticality.MEDIUM: CircuitBreakerConfig(
                    failure_threshold=int(
                        os.getenv("CB_MEDIUM_FAILURE_THRESHOLD", "2"),
                    ),
                    recovery_timeout=int(os.getenv("CB_MEDIUM_RECOVERY_TIMEOUT", "15")),
                ),
                ServiceCriticality.LOW: CircuitBreakerConfig(
                    failure_threshold=int(os.getenv("CB_LOW_FAILURE_THRESHOLD", "2")),
                    recovery_timeout=int(os.getenv("CB_LOW_RECOVERY_TIMEOUT", "5")),
                ),
            }

    def get_config(self, criticality: ServiceCriticality) -> CircuitBreakerConfig:
        """Get circuit breaker configuration for specified criticality level."""
        return self.configs[criticality]

    def create_circuit_breaker(
        self,
        criticality: ServiceCriticality,
        name: str | None = None,
        expected_exception: type | None = None,
    ):
        """Create a circuit breaker decorator with appropriate configuration.

        Args:
            criticality: Service criticality level
            name: Optional name for the circuit breaker (for monitoring)
            expected_exception: Exception type to trigger circuit breaker

        Returns:
            Circuit breaker decorator
        """
        config = self.get_config(criticality)

        # Override exception type if provided
        exception_type = expected_exception or config.expected_exception

        return circuit(
            failure_threshold=config.failure_threshold,
            recovery_timeout=config.recovery_timeout,
            expected_exception=exception_type,
            name=name,
        )

    def get_all_configurations(self) -> dict[ServiceCriticality, CircuitBreakerConfig]:
        """Get all circuit breaker configurations for monitoring/debugging."""
        return self.configs.copy()


# Global circuit breaker factory instance
circuit_breaker_factory = ConfigurableCircuitBreaker()


def get_circuit_breaker(
    criticality: ServiceCriticality,
    name: str | None = None,
    expected_exception: type | None = None,
):
    """Convenience function to get a circuit breaker decorator.

    Args:
        criticality: Service criticality level
        name: Optional name for the circuit breaker
        expected_exception: Exception type to trigger circuit breaker

    Returns:
        Circuit breaker decorator

    Example:
        @get_circuit_breaker(ServiceCriticality.CRITICAL, "database_connection")
        async def connect_to_database():
            # Database connection logic
            pass
    """
    return circuit_breaker_factory.create_circuit_breaker(
        criticality=criticality,
        name=name,
        expected_exception=expected_exception,
    )


# Service-specific circuit breaker configurations
def DATABASE_CIRCUIT_BREAKER():
    return get_circuit_breaker(
        ServiceCriticality.CRITICAL,
        "database_operations",
    )


def KAFKA_CIRCUIT_BREAKER():
    return get_circuit_breaker(
        ServiceCriticality.HIGH,
        "kafka_operations",
    )


def WORKFLOW_CIRCUIT_BREAKER():
    return get_circuit_breaker(
        ServiceCriticality.HIGH,
        "workflow_operations",
    )


def MONITORING_CIRCUIT_BREAKER():
    return get_circuit_breaker(
        ServiceCriticality.MEDIUM,
        "monitoring_operations",
    )


def EXTERNAL_API_CIRCUIT_BREAKER():
    return get_circuit_breaker(
        ServiceCriticality.LOW,
        "external_api_operations",
    )


def VAULT_CIRCUIT_BREAKER():
    """Circuit breaker for HashiCorp Vault operations."""
    return get_circuit_breaker(
        ServiceCriticality.CRITICAL,
        "vault_secrets_operations",
    )
