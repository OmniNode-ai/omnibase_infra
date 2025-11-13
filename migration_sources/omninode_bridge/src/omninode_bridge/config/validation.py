"""Configuration validation for OmniNode Bridge."""

import os
import socket
from pathlib import Path

from .environment_config import EnvironmentConfig


class ConfigValidationError(Exception):
    """Configuration validation error."""

    def __init__(self, message: str, field: str | None = None):
        self.message = message
        self.field = field
        super().__init__(message)


class ConfigValidator:
    """Validates configuration settings for different environments."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> tuple[bool, list[str], list[str]]:
        """Validate all configuration sections."""
        self.errors.clear()
        self.warnings.clear()

        # Validate each section
        self._validate_environment()
        self._validate_database()
        self._validate_kafka()
        self._validate_security()
        self._validate_services()
        self._validate_cache()
        self._validate_circuit_breaker()

        return len(self.errors) == 0, self.errors, self.warnings

    def _validate_environment(self):
        """Validate environment-specific settings."""
        if self.config.environment not in [
            "development",
            "staging",
            "production",
            "test",
        ]:
            self.errors.append(f"Invalid environment: {self.config.environment}")

        # Production-specific validations
        if self.config.is_production():
            if not self.config.database.password:
                self.errors.append("Database password is required in production")

            if self.config.security.api_key in [None, "omninode-bridge-api-key-2024"]:
                self.errors.append("Secure API key is required in production")

            if self.config.log_level == "debug":
                self.warnings.append("Debug logging enabled in production")

        # Development-specific warnings
        if self.config.is_development():
            if not self.config.database.password:
                self.warnings.append("Database password not set for development")

    def _validate_database(self):
        """Validate database configuration."""
        db_config = self.config.database

        # Connection validation
        try:
            socket.create_connection((db_config.host, db_config.port), timeout=5)
        except (TimeoutError, OSError):
            if self.config.is_production():
                self.errors.append(
                    f"Cannot connect to database at {db_config.host}:{db_config.port}",
                )
            else:
                self.warnings.append(
                    f"Database connection check failed: {db_config.host}:{db_config.port}",
                )

        # Pool size validation
        if db_config.pool_min_size > db_config.pool_max_size:
            self.errors.append(
                "Database pool_min_size cannot be greater than pool_max_size",
            )

        # SSL validation
        if db_config.ssl_enabled:
            if db_config.ssl_cert_path and not Path(db_config.ssl_cert_path).exists():
                self.errors.append(
                    f"SSL certificate file not found: {db_config.ssl_cert_path}",
                )

            if db_config.ssl_key_path and not Path(db_config.ssl_key_path).exists():
                self.errors.append(f"SSL key file not found: {db_config.ssl_key_path}")

            if db_config.ssl_ca_path and not Path(db_config.ssl_ca_path).exists():
                self.errors.append(f"SSL CA file not found: {db_config.ssl_ca_path}")

        # Performance settings validation
        if db_config.query_timeout_seconds < 5:
            self.warnings.append("Query timeout is very low (< 5 seconds)")

        if db_config.pool_max_size > 100:
            self.warnings.append("Very high database pool size may impact performance")

    def _validate_kafka(self):
        """Validate Kafka configuration."""
        kafka_config = self.config.kafka

        # Bootstrap servers validation
        servers = kafka_config.bootstrap_servers.split(",")
        for server in servers:
            if ":" not in server.strip():
                self.errors.append(f"Invalid Kafka bootstrap server format: {server}")
            else:
                host, port_str = server.strip().rsplit(":", 1)
                try:
                    port = int(port_str)
                    if not (1 <= port <= 65535):
                        self.errors.append(f"Invalid Kafka port: {port}")
                except ValueError:
                    self.errors.append(f"Invalid Kafka port format: {port_str}")

        # Topic name validation
        for topic_name in [kafka_config.workflow_topic, kafka_config.task_events_topic]:
            if not topic_name or len(topic_name) < 3:
                self.errors.append(f"Invalid topic name: {topic_name}")

        # Performance settings validation
        if kafka_config.batch_size > kafka_config.max_request_size:
            self.errors.append("Kafka batch_size cannot exceed max_request_size")

    def _validate_security(self):
        """Validate security configuration."""
        security_config = self.config.security

        # API key validation
        if security_config.api_key:
            if len(security_config.api_key) < 16:
                self.warnings.append(
                    "API key is shorter than recommended (16+ characters)",
                )

            # Check against known weak/default API keys
            weak_api_keys = [
                "omninode-bridge-api-key-2024",
                "omninode-bridge-secure-api-key-2024",
                "default-api-key",
                "test-api-key",
                "development-key",
            ]
            if security_config.api_key in weak_api_keys:
                if self.config.is_production():
                    self.errors.append(
                        "Weak or default API key detected - must be changed in production"
                    )
                else:
                    self.warnings.append(
                        "Using weak or default API key - should be changed"
                    )

        # CORS validation
        for origin in security_config.cors_allowed_origins:
            if origin == "*" and self.config.is_production():
                self.warnings.append(
                    "Wildcard CORS origin (*) not recommended in production",
                )

        # Rate limiting validation
        if security_config.rate_limit_requests_per_minute < 1:
            self.errors.append("Rate limit must be at least 1 request per minute")

        if (
            security_config.rate_limit_burst_size
            > security_config.rate_limit_requests_per_minute
        ):
            self.warnings.append("Rate limit burst size exceeds requests per minute")

    def _validate_services(self):
        """Validate service configuration."""
        services_config = self.config.services

        # Port validation
        ports = [
            services_config.hook_receiver_port,
            services_config.model_metrics_port,
            services_config.workflow_coordinator_port,
            services_config.ai_lab_ollama_port,
        ]

        # Check for port conflicts
        if len(set(ports)) != len(ports):
            self.errors.append("Service port conflicts detected")

        # URL validation
        urls = [services_config.hook_receiver_url, services_config.model_metrics_url]

        for url in urls:
            if not url.startswith(("http://", "https://")):
                self.errors.append(f"Invalid service URL format: {url}")

    def _validate_cache(self):
        """Validate cache configuration."""
        cache_config = self.config.cache

        # Memory validation
        if cache_config.workflow_memory_mb > cache_config.workflow_disk_mb:
            self.warnings.append("Workflow memory cache larger than disk cache")

        # Directory validation
        if cache_config.workflow_cache_dir:
            cache_dir = Path(cache_config.workflow_cache_dir)
            if not cache_dir.parent.exists():
                self.errors.append(
                    f"Cache directory parent does not exist: {cache_dir.parent}",
                )

        # Threshold validation
        if (
            cache_config.workflow_compression_threshold_kb
            > cache_config.workflow_disk_threshold_kb
        ):
            self.warnings.append("Compression threshold larger than disk threshold")

    def _validate_circuit_breaker(self):
        """Validate circuit breaker configuration."""
        cb_config = self.config.circuit_breaker

        # Threshold validation
        thresholds = [
            cb_config.critical_failure_threshold,
            cb_config.high_failure_threshold,
            cb_config.medium_failure_threshold,
            cb_config.low_failure_threshold,
        ]

        if not all(t >= 1 for t in thresholds):
            self.errors.append("All circuit breaker failure thresholds must be >= 1")

        # Timeout validation
        timeouts = [
            cb_config.critical_recovery_timeout,
            cb_config.high_recovery_timeout,
            cb_config.medium_recovery_timeout,
            cb_config.low_recovery_timeout,
        ]

        if not all(t >= 5 for t in timeouts):
            self.errors.append(
                "All circuit breaker recovery timeouts must be >= 5 seconds",
            )


def validate_config(
    config: EnvironmentConfig | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Validate configuration and return results."""
    if config is None:
        from . import get_config

        config = get_config()

    validator = ConfigValidator(config)
    return validator.validate_all()


def validate_startup_config() -> bool:
    """Validate configuration at startup and log results."""
    from omninode_bridge.config import get_config

    config = get_config()
    is_valid, errors, warnings = validate_config(config)

    # Log validation results
    import logging

    logger = logging.getLogger(__name__)

    if warnings:
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")

    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        logger.error(
            "Configuration validation failed - service may not function properly",
        )
        return False
    else:
        logger.info("Configuration validation passed")
        return True


def check_required_environment_variables() -> tuple[bool, list[str]]:
    """Check that required environment variables are set."""
    required_vars = []
    missing_vars = []

    # Environment-specific required variables
    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment == "production":
        required_vars.extend(["POSTGRES_PASSWORD", "SECURITY_API_KEY"])

    if environment in ["staging", "production"]:
        required_vars.extend(["KAFKA_BOOTSTRAP_SERVERS", "POSTGRES_HOST"])

    # Check for missing variables
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    return len(missing_vars) == 0, missing_vars
