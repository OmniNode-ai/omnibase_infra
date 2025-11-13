"""
Registry Configuration Module

Provides centralized configuration for the NodeBridgeRegistry component
with environment-specific settings and production-ready defaults.
"""

import os
from functools import lru_cache
from typing import Optional


class RegistryConfig:
    """
    Registry configuration with environment-specific defaults.

    Provides configurable settings for:
    - Offset tracking and memory management
    - Background task cleanup intervals
    - Circuit breaker thresholds
    - Connection timeouts and retry limits
    - Security settings
    """

    def __init__(self, environment: str = "development"):
        """
        Initialize registry configuration.

        Args:
            environment: Environment name (development, test, staging, production)
        """
        self.environment = environment.lower()

        # Load environment-specific configuration
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load configuration from environment variables with sensible defaults."""

        # === Offset Tracking Configuration ===
        # Memory management for processed message offsets
        self.offset_tracking_enabled = self._get_bool_env(
            "OFFSET_TRACKING_ENABLED", True
        )
        self.max_tracked_offsets = self._get_int_env(
            "MAX_TRACKED_OFFSETS", 10000 if self.environment == "production" else 5000
        )
        self.offset_cache_ttl_seconds = self._get_int_env(
            "OFFSET_CACHE_TTL_SECONDS",
            3600 if self.environment == "production" else 1800,  # 1 hour / 30 minutes
        )
        self.offset_cleanup_interval_seconds = self._get_int_env(
            "OFFSET_CLEANUP_INTERVAL_SECONDS",
            300 if self.environment == "production" else 600,  # 5 minutes / 10 minutes
        )

        # === Background Task Configuration ===
        # Cleanup intervals and task management
        self.node_ttl_hours = self._get_int_env(
            "NODE_TTL_HOURS", 24 if self.environment == "production" else 12
        )
        self.cleanup_interval_hours = self._get_float_env(
            "CLEANUP_INTERVAL_HOURS", 2.0 if self.environment == "production" else 6.0
        )
        self.cleanup_task_timeout_seconds = self._get_int_env(
            "CLEANUP_TASK_TIMEOUT_SECONDS",
            300 if self.environment == "production" else 600,
        )

        # === Circuit Breaker Configuration ===
        # Error recovery and retry logic
        self.circuit_breaker_enabled = self._get_bool_env(
            "CIRCUIT_BREAKER_ENABLED", True
        )
        self.max_retry_attempts = self._get_int_env(
            "MAX_RETRY_ATTEMPTS", 3 if self.environment == "production" else 5
        )
        self.retry_backoff_base_seconds = self._get_float_env(
            "RETRY_BACKOFF_BASE_SECONDS", 2.0
        )
        self.retry_backoff_max_seconds = self._get_float_env(
            "RETRY_BACKOFF_MAX_SECONDS", 60.0
        )
        self.circuit_breaker_threshold = self._get_int_env(
            "CIRCUIT_BREAKER_THRESHOLD", 5
        )
        self.circuit_breaker_timeout_seconds = self._get_int_env(
            "CIRCUIT_BREAKER_TIMEOUT_SECONDS",
            60 if self.environment == "production" else 30,
        )

        # === Atomic Registration Configuration ===
        # Transaction handling and data consistency
        self.atomic_registration_enabled = self._get_bool_env(
            "ATOMIC_REGISTRATION_ENABLED",
            self.environment == "production",
        )
        self.registration_timeout_seconds = self._get_int_env(
            "REGISTRATION_TIMEOUT_SECONDS",
            30 if self.environment == "production" else 60,
        )
        self.consul_timeout_seconds = self._get_int_env(
            "CONSUL_TIMEOUT_SECONDS", 10 if self.environment == "production" else 15
        )
        self.postgres_timeout_seconds = self._get_int_env(
            "POSTGRES_TIMEOUT_SECONDS", 15 if self.environment == "production" else 30
        )

        # === Database Retry Configuration ===
        # Connection retry logic for PostgreSQL
        self.database_max_retries = self._get_int_env(
            "DATABASE_MAX_RETRIES", 5 if self.environment == "production" else 3
        )
        self.database_retry_base_delay_seconds = self._get_float_env(
            "DATABASE_RETRY_BASE_DELAY_SECONDS", 1.0
        )
        self.database_retry_max_delay_seconds = self._get_float_env(
            "DATABASE_RETRY_MAX_DELAY_SECONDS", 30.0
        )

        # === Performance Configuration ===
        # Connection pools and batch sizes
        self.connection_pool_size = self._get_int_env(
            "CONNECTION_POOL_SIZE", 20 if self.environment == "production" else 10
        )
        self.max_concurrent_registrations = self._get_int_env(
            "MAX_CONCURRENT_REGISTRATIONS",
            100 if self.environment == "production" else 50,
        )
        self.memory_monitoring_interval_seconds = self._get_int_env(
            "MEMORY_MONITORING_INTERVAL_SECONDS",
            60 if self.environment == "production" else 120,
        )
        self.memory_warning_threshold_mb = self._get_float_env(
            "MEMORY_WARNING_THRESHOLD_MB",
            512.0 if self.environment == "production" else 256.0,
        )
        self.memory_critical_threshold_mb = self._get_float_env(
            "MEMORY_CRITICAL_THRESHOLD_MB",
            1024.0 if self.environment == "production" else 512.0,
        )

        # === Security Configuration ===
        # Logging and data sanitization
        self.sanitize_logs_in_production = self._get_bool_env(
            "SANITIZE_LOGS_IN_PRODUCTION",
            self.environment == "production",
        )
        self.log_sensitive_data = self._get_bool_env("LOG_SENSITIVE_DATA", False)
        self.enable_emoji_logs = self._get_bool_env(
            "ENABLE_EMOJI_LOGS", self.environment != "production"
        )
        self.security_mask_character = self._get_str_env("SECURITY_MASK_CHARACTER", "*")

        # === Kafka Configuration ===
        # Message processing and consumer settings
        self.kafka_consumer_timeout_ms = self._get_int_env(
            "KAFKA_CONSUMER_TIMEOUT_MS",
            5000 if self.environment == "production" else 10000,
        )
        self.kafka_max_poll_records = self._get_int_env(
            "KAFKA_MAX_POLL_RECORDS", 100 if self.environment == "production" else 50
        )
        self.kafka_auto_commit_enabled = self._get_bool_env(
            "KAFKA_AUTO_COMMIT_ENABLED",
            False,  # Always use manual commits for reliability
        )
        self.kafka_commit_timeout_ms = self._get_int_env(
            "KAFKA_COMMIT_TIMEOUT_MS",
            5000 if self.environment == "production" else 10000,
        )

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key, "").lower()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
        return default

    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer value from environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default

    def _get_float_env(self, key: str, default: float) -> float:
        """Get float value from environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default

    def _get_str_env(self, key: str, default: str) -> str:
        """Get string value from environment variable."""
        return os.getenv(key, default)

    def get_summary(self) -> dict:
        """
        Get a summary of current configuration.

        Returns:
            Dictionary containing current configuration values
        """
        return {
            "environment": self.environment,
            "offset_tracking": {
                "enabled": self.offset_tracking_enabled,
                "max_tracked_offsets": self.max_tracked_offsets,
                "cache_ttl_seconds": self.offset_cache_ttl_seconds,
                "cleanup_interval_seconds": self.offset_cleanup_interval_seconds,
            },
            "background_tasks": {
                "node_ttl_hours": self.node_ttl_hours,
                "cleanup_interval_hours": self.cleanup_interval_hours,
                "cleanup_timeout_seconds": self.cleanup_task_timeout_seconds,
            },
            "circuit_breaker": {
                "enabled": self.circuit_breaker_enabled,
                "max_retry_attempts": self.max_retry_attempts,
                "backoff_base_seconds": self.retry_backoff_base_seconds,
                "backoff_max_seconds": self.retry_backoff_max_seconds,
                "failure_threshold": self.circuit_breaker_threshold,
                "timeout_seconds": self.circuit_breaker_timeout_seconds,
            },
            "atomic_registration": {
                "enabled": self.atomic_registration_enabled,
                "timeout_seconds": self.registration_timeout_seconds,
                "consul_timeout_seconds": self.consul_timeout_seconds,
                "postgres_timeout_seconds": self.postgres_timeout_seconds,
            },
            "database_retry": {
                "max_retries": self.database_max_retries,
                "base_delay_seconds": self.database_retry_base_delay_seconds,
                "max_delay_seconds": self.database_retry_max_delay_seconds,
            },
            "performance": {
                "connection_pool_size": self.connection_pool_size,
                "max_concurrent_registrations": self.max_concurrent_registrations,
                "memory_monitoring_interval": self.memory_monitoring_interval_seconds,
                "memory_warning_threshold_mb": self.memory_warning_threshold_mb,
                "memory_critical_threshold_mb": self.memory_critical_threshold_mb,
            },
            "security": {
                "sanitize_logs": self.sanitize_logs_in_production,
                "log_sensitive_data": self.log_sensitive_data,
                "enable_emoji_logs": self.enable_emoji_logs,
                "mask_character": self.security_mask_character,
            },
            "kafka": {
                "consumer_timeout_ms": self.kafka_consumer_timeout_ms,
                "max_poll_records": self.kafka_max_poll_records,
                "auto_commit_enabled": self.kafka_auto_commit_enabled,
                "commit_timeout_ms": self.kafka_commit_timeout_ms,
            },
        }


# Global instance for backward compatibility
_global_registry_config: Optional[RegistryConfig] = None


@lru_cache(maxsize=1)
def get_registry_config(environment: str = "development") -> RegistryConfig:
    """
    Get global registry configuration instance.

    Args:
        environment: Environment name

    Returns:
        RegistryConfig instance
    """
    global _global_registry_config
    if (
        _global_registry_config is None
        or _global_registry_config.environment != environment
    ):
        _global_registry_config = RegistryConfig(environment=environment)
    return _global_registry_config


def set_registry_config(config: RegistryConfig) -> None:
    """
    Set global registry configuration instance.

    Args:
        config: RegistryConfig instance
    """
    global _global_registry_config
    _global_registry_config = config
