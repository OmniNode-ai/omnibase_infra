"""Settings configuration for metadata stamping service."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="METADATA_STAMPING_",
        extra="ignore",  # Ignore extra environment variables
    )

    # Service configuration
    service_name: str = Field(
        default="metadata-stamping-service", description="Service name"
    )
    service_host: str = Field(default="0.0.0.0", description="Service host")
    service_port: int = Field(default=8053, description="Service port")
    service_workers: int = Field(default=1, description="Number of workers")
    service_reload: bool = Field(default=False, description="Enable hot reload")
    log_level: str = Field(default="INFO", description="Logging level")

    # Database configuration
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="metadata_stamping_dev", description="Database name")
    db_user: str = Field(default="postgres", description="Database user")
    db_password: str = Field(default="", description="Database password")
    db_pool_min_size: int = Field(default=5, description="Minimum connection pool size")
    db_pool_max_size: int = Field(
        default=20, description="Maximum connection pool size"
    )
    db_command_timeout: float = Field(
        default=30.0, description="Database command timeout"
    )
    db_enable_ssl: bool = Field(default=False, description="Enable SSL for database")

    # Performance tuning
    hash_generator_pool_size: int = Field(
        default=50, description="Hash generator pool size"
    )
    hash_generator_max_workers: int = Field(default=2, description="Max worker threads")
    enable_batch_operations: bool = Field(
        default=True, description="Enable batch operations"
    )
    enable_performance_metrics: bool = Field(
        default=True, description="Enable metrics collection"
    )
    enable_debug_endpoints: bool = Field(
        default=False, description="Enable debug endpoints"
    )

    # Feature flags
    enable_cors: bool = Field(default=False, description="Enable CORS")
    enable_prometheus_metrics: bool = Field(
        default=True, description="Enable Prometheus metrics"
    )
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")

    # File processing limits
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    max_batch_size: int = Field(default=50, description="Maximum batch size")
    supported_file_types: str = Field(
        default="jpg,jpeg,png,gif,pdf,txt,md,doc,docx,mp3,mp4,zip",
        description="Comma-separated list of supported file extensions",
    )

    # Event publishing configuration
    enable_events: bool = Field(
        default=False, description="Enable Kafka event publishing"
    )
    kafka_bootstrap_servers: str = Field(
        default="localhost:29092", description="Kafka bootstrap servers"
    )
    kafka_enable_dlq: bool = Field(
        default=True, description="Enable Kafka dead letter queue"
    )
    kafka_max_retry_attempts: int = Field(
        default=3, description="Maximum Kafka retry attempts"
    )
    kafka_timeout_seconds: int = Field(
        default=30, description="Kafka operation timeout"
    )
    event_secret_key: str = Field(
        default="", description="Secret key for event signing (HMAC-SHA256)"
    )
    event_key_id: str = Field(default="default", description="Key ID for event signing")
    event_actor_id: str = Field(
        default="system", description="Default actor ID for events"
    )

    # Registry configuration
    enable_registry: bool = Field(default=False, description="Enable Consul registry")
    consul_host: str = Field(default="localhost", description="Consul host")
    consul_port: int = Field(default=8500, description="Consul port")
    service_registration_enabled: bool = Field(
        default=True, description="Enable service registration"
    )
    local_ip: str = Field(
        default="localhost",
        description="Local machine IP for health check URLs (use when Consul is remote)",
    )

    # Security configuration
    enable_security: bool = Field(
        default=True, description="Enable O.N.E. security middleware"
    )
    trusted_public_keys: str = Field(
        default="", description="Comma-separated trusted public keys"
    )
    signature_validation_enabled: bool = Field(
        default=True, description="Enable signature validation"
    )

    def get_database_config(self) -> dict:
        """Get database configuration dictionary.

        Returns:
            Database configuration
        """
        return {
            "host": self.db_host,
            "port": self.db_port,
            "database": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
            "min_connections": self.db_pool_min_size,
            "max_connections": self.db_pool_max_size,
            "command_timeout": self.db_command_timeout,
            "ssl_enabled": self.db_enable_ssl,
        }

    def get_supported_file_types_list(self) -> list:
        """Get list of supported file types.

        Returns:
            List of supported file extensions
        """
        return [f".{ext}" for ext in self.supported_file_types.split(",")]

    def get_event_config(self) -> dict:
        """Get event publishing configuration dictionary.

        Returns:
            Event publishing configuration
        """
        return {
            "enable_events": self.enable_events,
            "kafka_bootstrap_servers": self.kafka_bootstrap_servers,
            "kafka_enable_dlq": self.kafka_enable_dlq,
            "kafka_max_retry_attempts": self.kafka_max_retry_attempts,
            "kafka_timeout_seconds": self.kafka_timeout_seconds,
            "event_secret_key": self.event_secret_key,
            "event_key_id": self.event_key_id,
            "event_actor_id": self.event_actor_id,
        }

    def get_registry_config(self) -> dict:
        """Get registry configuration dictionary.

        Returns:
            Registry configuration
        """
        return {
            "enable_registry": self.enable_registry,
            "consul_host": self.consul_host,
            "consul_port": self.consul_port,
            "service_registration_enabled": self.service_registration_enabled,
        }

    def get_security_config(self) -> dict:
        """Get security configuration dictionary.

        Returns:
            Security configuration
        """
        return {
            "enable_security": self.enable_security,
            "trusted_public_keys": (
                self.trusted_public_keys.split(",") if self.trusted_public_keys else []
            ),
            "signature_validation_enabled": self.signature_validation_enabled,
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance
    """
    return Settings()
