"""Circuit Breaker Environment Configuration Model.

Environment-specific circuit breaker configuration model for PostgreSQL-RedPanda
event bus integration. Provides strongly typed configuration overrides for different
deployment environments (production, staging, development).

Following ONEX shared model architecture with contract-driven configuration.
"""

from enum import Enum

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from pydantic import BaseModel, Field, validator


class EnvironmentType(str, Enum):
    """Supported deployment environments."""

    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"


class ModelCircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration for a specific environment."""

    failure_threshold: int = Field(
        ...,
        ge=1,
        le=20,
        description="Number of failures before opening circuit",
    )
    recovery_timeout: int = Field(
        ...,
        ge=5,
        le=300,
        description="Seconds before transitioning to half-open",
    )
    success_threshold: int = Field(
        ...,
        ge=1,
        le=10,
        description="Successes needed in half-open to close circuit",
    )
    timeout_seconds: int = Field(
        ...,
        ge=5,
        le=120,
        description="Event publishing timeout in seconds",
    )
    max_queue_size: int = Field(
        ...,
        ge=10,
        le=10000,
        description="Maximum queued events when circuit is open",
    )
    dead_letter_enabled: bool = Field(
        ...,
        description="Enable dead letter queue for failed events",
    )
    graceful_degradation: bool = Field(
        ...,
        description="Allow operations to continue without events",
    )

    @validator("recovery_timeout")
    def validate_recovery_timeout(cls, v: int, values: dict) -> int:
        """Validate recovery timeout is reasonable for failure threshold."""
        failure_threshold = values.get("failure_threshold", 0)
        if failure_threshold > 0 and v < failure_threshold * 2:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Recovery timeout ({v}s) should be at least 2x failure threshold ({failure_threshold})",
            )
        return v

    @validator("success_threshold")
    def validate_success_threshold(cls, v: int, values: dict) -> int:
        """Validate success threshold is reasonable for failure threshold."""
        failure_threshold = values.get("failure_threshold", 0)
        if failure_threshold > 0 and v > failure_threshold:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Success threshold ({v}) should not exceed failure threshold ({failure_threshold})",
            )
        return v

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"
        schema_extra = {
            "example": {
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "success_threshold": 3,
                "timeout_seconds": 30,
                "max_queue_size": 1000,
                "dead_letter_enabled": True,
                "graceful_degradation": True,
            },
        }


class ModelCircuitBreakerEnvironmentConfig(BaseModel):
    """Environment-specific circuit breaker configuration model.

    Provides contract-driven environment configuration overrides for circuit breaker
    behavior in different deployment environments. Enables production-ready configuration
    management without hardcoded values.

    Usage:
        config = ModelCircuitBreakerEnvironmentConfig(
            production=ModelCircuitBreakerConfig(failure_threshold=5, ...),
            staging=ModelCircuitBreakerConfig(failure_threshold=3, ...),
            development=ModelCircuitBreakerConfig(failure_threshold=2, ...)
        )

        prod_config = config.get_config_for_environment("production")
    """

    production: ModelCircuitBreakerConfig = Field(
        ...,
        description="Circuit breaker configuration for production environment",
    )
    staging: ModelCircuitBreakerConfig = Field(
        ...,
        description="Circuit breaker configuration for staging environment",
    )
    development: ModelCircuitBreakerConfig = Field(
        ...,
        description="Circuit breaker configuration for development environment",
    )

    def get_config_for_environment(
        self,
        environment: str,
        default_environment: str | None = None,
    ) -> ModelCircuitBreakerConfig:
        """Get circuit breaker configuration for specified environment.

        Args:
            environment: Target environment name
            default_environment: Fallback environment if target not found

        Returns:
            Circuit breaker configuration for the environment

        Raises:
            OnexError: If environment not found and no default provided
        """
        try:
            env_type = EnvironmentType(environment.lower())
        except ValueError:
            if default_environment:
                try:
                    env_type = EnvironmentType(default_environment.lower())
                except ValueError:
                    raise OnexError(
                        code=CoreErrorCode.CONFIGURATION_ERROR,
                        message=f"Unknown environment '{environment}' and invalid default '{default_environment}'",
                    )
            else:
                raise OnexError(
                    code=CoreErrorCode.CONFIGURATION_ERROR,
                    message=f"Unknown environment '{environment}'. Supported: {list(EnvironmentType)}",
                )

        config_map = {
            EnvironmentType.PRODUCTION: self.production,
            EnvironmentType.STAGING: self.staging,
            EnvironmentType.DEVELOPMENT: self.development,
        }

        return config_map[env_type]

    def get_all_environments(self) -> dict[str, ModelCircuitBreakerConfig]:
        """Get all environment configurations as dictionary."""
        return {
            "production": self.production,
            "staging": self.staging,
            "development": self.development,
        }

    @classmethod
    def create_default_config(cls) -> "ModelCircuitBreakerEnvironmentConfig":
        """Create default environment configuration following production requirements."""
        return cls(
            production=ModelCircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60,
                success_threshold=3,
                timeout_seconds=30,
                max_queue_size=1000,
                dead_letter_enabled=True,
                graceful_degradation=True,
            ),
            staging=ModelCircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                success_threshold=2,
                timeout_seconds=20,
                max_queue_size=500,
                dead_letter_enabled=True,
                graceful_degradation=True,
            ),
            development=ModelCircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=15,
                success_threshold=1,
                timeout_seconds=10,
                max_queue_size=100,
                dead_letter_enabled=False,
                graceful_degradation=True,
            ),
        )

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"
        schema_extra = {
            "example": {
                "production": {
                    "failure_threshold": 5,
                    "recovery_timeout": 60,
                    "success_threshold": 3,
                    "timeout_seconds": 30,
                    "max_queue_size": 1000,
                    "dead_letter_enabled": True,
                    "graceful_degradation": True,
                },
                "staging": {
                    "failure_threshold": 3,
                    "recovery_timeout": 30,
                    "success_threshold": 2,
                    "timeout_seconds": 20,
                    "max_queue_size": 500,
                    "dead_letter_enabled": True,
                    "graceful_degradation": True,
                },
                "development": {
                    "failure_threshold": 2,
                    "recovery_timeout": 15,
                    "success_threshold": 1,
                    "timeout_seconds": 10,
                    "max_queue_size": 100,
                    "dead_letter_enabled": False,
                    "graceful_degradation": True,
                },
            },
        }
