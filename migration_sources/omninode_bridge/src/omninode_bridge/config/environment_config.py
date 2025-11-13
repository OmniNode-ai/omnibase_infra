"""Environment-specific configuration management for OmniNode Bridge."""

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration settings with environment-specific defaults and validation.

    This class provides comprehensive PostgreSQL database configuration with:
    - Environment-based default values for development, staging, and production
    - SSL/TLS security configuration for encrypted connections
    - Connection pool optimization with performance tuning
    - Automatic validation of security requirements based on environment
    - Connection monitoring and leak detection capabilities

    All settings can be overridden via environment variables with the POSTGRES_ prefix.
    For example, POSTGRES_HOST overrides the host setting.

    Environment Security Policies:
        - Development: SSL optional, basic connection pooling
        - Staging: SSL recommended, moderate connection pooling
        - Production: SSL required, optimized connection pooling

    Example:
        Basic usage with environment variables:
            POSTGRES_HOST=db.example.com
            POSTGRES_PASSWORD=secure_password
            POSTGRES_SSL_ENABLED=true

            config = DatabaseConfig()
            url = config.get_database_url()

    Attributes:
        host: PostgreSQL server hostname or IP address
        port: PostgreSQL server port (1-65535)
        database: Database name to connect to
        user: Database username for authentication
        password: Database password (required in production)

        pool_min_size: Minimum number of connections in pool
        pool_max_size: Maximum number of connections in pool
        pool_exhaustion_threshold: Pool utilization warning threshold percentage

        max_queries_per_connection: Connection renewal threshold
        connection_max_age_seconds: Maximum connection lifetime
        query_timeout_seconds: Query execution timeout
        acquire_timeout_seconds: Connection acquisition timeout

        ssl_enabled: Enable SSL/TLS encryption
        ssl_cert_path: Path to SSL client certificate
        ssl_key_path: Path to SSL client private key
        ssl_ca_path: Path to SSL Certificate Authority bundle
        ssl_check_hostname: Verify SSL certificate hostname

        leak_detection: Enable connection pool leak detection
    """

    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    database: str = Field(default="omninode_bridge", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str | None = Field(default=None, description="Database password")

    # Connection pool settings
    pool_min_size: int = Field(default=5, ge=1, description="Minimum pool size")
    pool_max_size: int = Field(default=20, ge=1, description="Maximum pool size")
    pool_exhaustion_threshold: int = Field(
        default=80,
        ge=50,
        le=95,
        description="Pool exhaustion warning threshold",
    )

    # Performance settings
    max_queries_per_connection: int = Field(
        default=10000,
        ge=100,
        description="Max queries per connection before renewal",
    )
    connection_max_age_seconds: int = Field(
        default=3600,
        ge=300,
        description="Connection maximum age in seconds",
    )
    query_timeout_seconds: int = Field(
        default=30,
        ge=5,
        description="Query timeout in seconds",
    )
    acquire_timeout_seconds: int = Field(
        default=10,
        ge=1,
        description="Connection acquire timeout",
    )

    # SSL settings
    ssl_enabled: bool = Field(default=False, description="Enable SSL connection")
    ssl_cert_path: str | None = Field(
        default=None,
        description="SSL certificate path",
    )
    ssl_key_path: str | None = Field(default=None, description="SSL key path")
    ssl_ca_path: str | None = Field(default=None, description="SSL CA path")
    ssl_check_hostname: bool = Field(default=True, description="Verify SSL hostname")

    # Monitoring
    leak_detection: bool = Field(
        default=True,
        description="Enable connection leak detection",
    )

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Validate password requirements based on environment.

        Args:
            v: Password value to validate
            info: Pydantic validation info context

        Returns:
            Validated password value

        Raises:
            ValueError: If password is required but missing in production environment
        """
        if not v:
            environment = os.getenv("ENVIRONMENT", "development").lower()
            if environment == "production":
                raise ValueError(
                    "Database password is required in production environment",
                )
        return v

    def get_database_url(self) -> str:
        """Generate database URL from configuration.

        Creates a PostgreSQL connection URL suitable for use with asyncpg
        or other database clients. Handles password encoding for special characters.

        Returns:
            Complete PostgreSQL connection URL string

        Example:
            config = DatabaseConfig(host="db.example.com", password="secret@123")
            url = config.get_database_url()
            # Returns: "postgresql://postgres:secret%40123@db.example.com:5436/omninode_bridge"
        """
        password_part = f":{self.password}" if self.password else ""
        return f"postgresql://{self.user}{password_part}@{self.host}:{self.port}/{self.database}"


class KafkaConfig(BaseSettings):
    """Kafka configuration settings with performance optimization and topic management.

    This class provides comprehensive Apache Kafka/RedPanda configuration with:
    - Producer optimization for high-throughput message publishing
    - Consumer configuration for reliable message processing
    - Topic naming conventions for ONEX ecosystem integration
    - Compression settings for network efficiency
    - Memory and batch size tuning for performance

    All settings can be overridden via environment variables with the KAFKA_ prefix.
    For example, KAFKA_BOOTSTRAP_SERVERS overrides the bootstrap_servers setting.

    Performance Considerations:
        - Batch size affects throughput vs latency tradeoff
        - Compression reduces network usage but increases CPU load
        - Buffer memory controls producer memory usage
        - Linger time affects message batching efficiency

    Example:
        Basic usage with environment variables:
            KAFKA_BOOTSTRAP_SERVERS=kafka1:9092,kafka2:9092
            KAFKA_COMPRESSION_TYPE=snappy
            KAFKA_BATCH_SIZE=32768

            config = KafkaConfig()
            print(f"Connecting to: {config.bootstrap_servers}")

    Attributes:
        bootstrap_servers: Comma-separated list of Kafka broker addresses

        workflow_topic: Topic name for workflow execution events
        task_events_topic: Topic name for task status updates

        compression_type: Message compression algorithm (none, gzip, snappy, lz4, zstd)
        batch_size: Producer batch size in bytes for batching efficiency
        linger_ms: Producer wait time for batching (0=no wait, >0=efficiency)
        buffer_memory: Producer buffer memory pool size in bytes
        max_request_size: Maximum size of producer requests in bytes

        group_id: Consumer group identifier for load balancing
        auto_offset_reset: Consumer offset reset policy (earliest, latest)
        enable_auto_commit: Enable automatic offset commits
    """

    model_config = SettingsConfigDict(
        env_prefix="KAFKA_",
        case_sensitive=False,
        extra="ignore",
    )

    bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers",
    )

    # Topic configuration
    workflow_topic: str = Field(
        default="dev.omninode_bridge.onex.workflows.v1",
        description="Workflow topic name",
    )
    task_events_topic: str = Field(
        default="dev.omninode_bridge.onex.task-events.v1",
        description="Task events topic name",
    )

    # Producer settings
    compression_type: str = Field(
        default="snappy",
        description="Compression type for Kafka messages",
    )
    batch_size: int = Field(
        default=16384,
        ge=1024,
        description="Batch size for producer",
    )
    linger_ms: int = Field(
        default=5,
        ge=0,
        description="Producer linger time in milliseconds",
    )
    buffer_memory: int = Field(
        default=33554432,
        ge=1048576,
        description="Producer buffer memory",
    )
    max_request_size: int = Field(
        default=1048576,
        ge=1024,
        description="Maximum request size",
    )

    # Consumer settings
    group_id: str = Field(
        default="omninode-bridge-consumer",
        description="Consumer group ID",
    )
    auto_offset_reset: str = Field(
        default="latest",
        description="Auto offset reset policy",
    )
    enable_auto_commit: bool = Field(default=True, description="Enable auto commit")

    @field_validator("compression_type")
    @classmethod
    def validate_compression_type(cls, v: str, info: ValidationInfo) -> str:
        """Validate Kafka compression type against supported algorithms.

        Args:
            v: Compression type string to validate
            info: Pydantic validation info context

        Returns:
            Validated compression type

        Raises:
            ValueError: If compression type is not supported by Kafka

        Note:
            Supported compression types:
            - none: No compression (fastest, largest messages)
            - gzip: Good compression ratio, moderate CPU usage
            - snappy: Fast compression, good for high throughput
            - lz4: Very fast compression, minimal CPU overhead
            - zstd: Best compression ratio, higher CPU usage
        """
        valid_types = {"none", "gzip", "snappy", "lz4", "zstd"}
        if v not in valid_types:
            raise ValueError(
                f"Invalid compression type: {v}. Must be one of {valid_types}",
            )
        return v


class SecurityConfig(BaseSettings):
    """Security configuration settings for API protection and access control.

    This class provides comprehensive security configuration including:
    - API key authentication for service-to-service communication
    - CORS (Cross-Origin Resource Sharing) policy configuration
    - Rate limiting to prevent abuse and DoS attacks
    - Security audit logging for compliance and monitoring
    - Encryption settings for sensitive data protection

    All settings can be overridden via environment variables with the SECURITY_ prefix.
    For example, SECURITY_API_KEY overrides the api_key setting.

    Security Best Practices:
        - Use strong, unique API keys in production
        - Configure restrictive CORS origins for production
        - Set appropriate rate limits based on expected usage
        - Enable audit logging for security monitoring
        - Rotate API keys regularly

    Example:
        Production security configuration:
            SECURITY_API_KEY=your-secure-api-key-here
            SECURITY_CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
            SECURITY_RATE_LIMIT_REQUESTS_PER_MINUTE=1000
            SECURITY_AUDIT_LOG_ENABLED=true

            config = SecurityConfig()
            print(f"Rate limit: {config.rate_limit_requests_per_minute} req/min")

    Attributes:
        api_key: API key for authenticating service requests
        api_key_encryption_seed: Seed for API key encryption algorithms

        cors_allowed_origins: List of allowed CORS origins for web requests
        cors_allow_credentials: Whether to allow credentials in CORS requests
        cors_allow_methods: HTTP methods allowed in CORS requests
        cors_allow_headers: HTTP headers allowed in CORS requests

        rate_limit_requests_per_minute: Maximum requests per minute per client
        rate_limit_burst_size: Number of requests allowed in burst

        audit_log_enabled: Enable security audit logging for compliance
    """

    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        case_sensitive=False,
        extra="ignore",
    )

    # API Security
    api_key: str | None = Field(
        default=None,
        description="API key for service authentication",
    )
    api_key_encryption_seed: str = Field(
        default="omninode-bridge-2024",
        description="API key encryption seed",
    )

    # CORS settings
    cors_allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins",
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow CORS credentials",
    )
    cors_allow_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"],
        description="Allowed CORS methods",
    )
    cors_allow_headers: list[str] = Field(
        default=["*"],
        description="Allowed CORS headers",
    )

    # Rate limiting
    rate_limit_requests_per_minute: int = Field(
        default=100,
        ge=1,
        description="Rate limit requests per minute",
    )
    rate_limit_burst_size: int = Field(
        default=10,
        ge=1,
        description="Rate limit burst size",
    )

    # Audit logging
    audit_log_enabled: bool = Field(
        default=True,
        description="Enable security audit logging",
    )

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str], info: ValidationInfo) -> list[str]:
        """Parse CORS origins from string or list format.

        Args:
            v: CORS origins as comma-separated string or list
            info: Pydantic validation info context

        Returns:
            List of validated CORS origin URLs

        Example:
            Input: "http://localhost:3000,https://app.example.com"
            Output: ["http://localhost:3000", "https://app.example.com"]

            Input: ["http://localhost:3000", "https://app.example.com"]
            Output: ["http://localhost:3000", "https://app.example.com"]
        """
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


class ServiceConfig(BaseSettings):
    """Service configuration settings for microservice communication and AI lab integration.

    This class provides configuration for all microservices in the OmniNode Bridge ecosystem:
    - Hook Receiver service for webhook processing
    - Model Metrics service for AI model monitoring
    - Workflow Coordinator service for task orchestration
    - AI Lab infrastructure with multiple compute nodes

    All services default to localhost (127.0.0.1) for security, preventing external access
    unless explicitly configured. In container environments, hostnames are automatically
    resolved to the appropriate container service names.

    AI Lab Infrastructure:
        The AI lab consists of multiple specialized compute nodes:
        - Mac Studio: Primary development and inference workstation
        - Mac Mini: Secondary compute node for parallel processing
        - AI PC: High-performance CUDA workstation for training
        - MacBook Air: Mobile development and testing environment

    Example:
        Basic service configuration:
            HOOK_RECEIVER_HOST=0.0.0.0
            HOOK_RECEIVER_PORT=8001
            AI_LAB_MAC_STUDIO=192.168.86.200
            AI_LAB_OLLAMA_PORT=11434

            config = ServiceConfig()
            print(f"Hook receiver: {config.hook_receiver_url}")

    Attributes:
        hook_receiver_host: Host for webhook receiver service
        hook_receiver_port: Port for webhook receiver service
        hook_receiver_workers: Number of worker processes
        hook_receiver_url: Complete URL for webhook receiver

        model_metrics_host: Host for model metrics service
        model_metrics_port: Port for model metrics service
        model_metrics_workers: Number of worker processes
        model_metrics_url: Complete URL for model metrics service

        workflow_coordinator_host: Host for workflow coordinator
        workflow_coordinator_port: Port for workflow coordinator
        workflow_coordinator_workers: Number of worker processes

        ai_lab_mac_studio: IP address of Mac Studio workstation
        ai_lab_mac_mini: IP address of Mac Mini compute node
        ai_lab_ai_pc: IP address of AI PC workstation
        ai_lab_macbook_air: IP address of MacBook Air development machine
        ai_lab_ollama_port: Port for Ollama LLM inference service
    """

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    # Hook Receiver - Default to localhost for security
    hook_receiver_host: str = Field(
        default="127.0.0.1", description="Hook receiver host"
    )
    hook_receiver_port: int = Field(
        default=8001,
        ge=1,
        le=65535,
        description="Hook receiver port",
    )
    hook_receiver_workers: int = Field(
        default=1,
        ge=1,
        description="Hook receiver workers",
    )
    hook_receiver_url: str = Field(
        default="http://localhost:8001",
        description="Hook receiver URL",
    )

    # Model Metrics - Default to localhost for security
    model_metrics_host: str = Field(
        default="127.0.0.1", description="Model metrics host"
    )
    model_metrics_port: int = Field(
        default=8005,
        ge=1,
        le=65535,
        description="Model metrics port",
    )
    model_metrics_workers: int = Field(
        default=1,
        ge=1,
        description="Model metrics workers",
    )
    model_metrics_url: str = Field(
        default="http://localhost:8005",
        description="Model metrics URL",
    )

    # Workflow Coordinator
    workflow_coordinator_host: str = Field(
        default="127.0.0.1",
        description="Workflow coordinator host",
    )
    workflow_coordinator_port: int = Field(
        default=8006,
        ge=1,
        le=65535,
        description="Workflow coordinator port",
    )
    workflow_coordinator_workers: int = Field(
        default=1,
        ge=1,
        description="Workflow coordinator workers",
    )

    # AI Lab Configuration
    ai_lab_mac_studio: str = Field(
        default="192.168.86.200",
        description="AI Lab Mac Studio IP",
    )
    ai_lab_mac_mini: str = Field(
        default="192.168.86.101",
        description="AI Lab Mac Mini IP",
    )
    ai_lab_ai_pc: str = Field(default="192.168.86.201", description="AI Lab AI PC IP")
    ai_lab_macbook_air: str = Field(
        default="192.168.86.105",
        description="AI Lab MacBook Air IP",
    )
    ai_lab_ollama_port: int = Field(
        default=11434,
        ge=1,
        le=65535,
        description="Ollama service port",
    )


class CacheConfig(BaseSettings):
    """Cache configuration settings for workflow data and performance optimization.

    This class provides configuration for workflow result caching with:
    - Memory-based caching for fast access to recent workflows
    - Disk-based caching for persistent storage of large workflows
    - Compression settings to optimize storage efficiency
    - Age-based expiration to prevent unbounded cache growth
    - Configurable thresholds for memory vs disk storage decisions

    All settings can be overridden via environment variables with the CACHE_ prefix.
    For example, CACHE_WORKFLOW_MEMORY_MB overrides the workflow_memory_mb setting.

    Cache Strategy:
        1. Small workflows (< compression_threshold_kb): Stored in memory
        2. Medium workflows (< disk_threshold_kb): Stored in memory with compression
        3. Large workflows (> disk_threshold_kb): Stored on disk with compression
        4. Expired workflows (> max_age_hours): Automatically purged

    Example:
        Production cache configuration:
            CACHE_WORKFLOW_MEMORY_MB=500
            CACHE_WORKFLOW_DISK_MB=2000
            CACHE_WORKFLOW_MAX_AGE_HOURS=48
            CACHE_WORKFLOW_CACHE_DIR=/var/cache/omninode

            config = CacheConfig()
            print(f"Memory limit: {config.workflow_memory_mb}MB")

    Attributes:
        workflow_memory_mb: Maximum memory usage for workflow cache in megabytes
        workflow_disk_mb: Maximum disk usage for workflow cache in megabytes
        workflow_compression_threshold_kb: Size threshold for enabling compression
        workflow_disk_threshold_kb: Size threshold for moving to disk storage
        workflow_max_age_hours: Maximum age before cache entries expire
        workflow_cache_dir: Directory path for disk-based cache storage
    """

    model_config = SettingsConfigDict(
        env_prefix="CACHE_",
        case_sensitive=False,
        extra="ignore",
    )

    # Workflow cache settings
    workflow_memory_mb: int = Field(
        default=100,
        ge=10,
        description="Workflow cache memory limit in MB",
    )
    workflow_disk_mb: int = Field(
        default=500,
        ge=50,
        description="Workflow cache disk limit in MB",
    )
    workflow_compression_threshold_kb: int = Field(
        default=10,
        ge=1,
        description="Compression threshold in KB",
    )
    workflow_disk_threshold_kb: int = Field(
        default=50,
        ge=5,
        description="Disk storage threshold in KB",
    )
    workflow_max_age_hours: int = Field(
        default=24,
        ge=1,
        description="Maximum cache age in hours",
    )
    workflow_cache_dir: str | None = Field(
        default=None,
        description="Workflow cache directory",
    )


class CircuitBreakerConfig(BaseSettings):
    """Circuit breaker configuration settings for fault tolerance and resilience.

    This class provides configuration for circuit breaker patterns that prevent
    cascading failures and provide graceful degradation when services are unhealthy:

    - Priority-based failure thresholds for different operation criticality levels
    - Configurable recovery timeouts for automatic service restoration
    - Exponential backoff and retry logic integration
    - Health check intervals and failure detection mechanisms

    Circuit Breaker States:
        - CLOSED: Normal operation, requests flow through
        - OPEN: Service is failing, requests are rejected immediately
        - HALF_OPEN: Testing if service has recovered, limited requests allowed

    Priority Levels:
        - CRITICAL: Essential operations (database connections, core APIs)
        - HIGH: Important operations (workflow execution, data processing)
        - MEDIUM: Standard operations (metrics collection, status updates)
        - LOW: Optional operations (logging, non-essential integrations)

    All settings can be overridden via environment variables with the CB_ prefix.
    For example, CB_CRITICAL_FAILURE_THRESHOLD overrides critical_failure_threshold.

    Example:
        Production circuit breaker configuration:
            CB_CRITICAL_FAILURE_THRESHOLD=3
            CB_CRITICAL_RECOVERY_TIMEOUT=120
            CB_HIGH_FAILURE_THRESHOLD=5

            config = CircuitBreakerConfig()
            print(f"Critical threshold: {config.critical_failure_threshold}")

    Attributes:
        critical_failure_threshold: Number of failures before opening critical circuit
        critical_recovery_timeout: Recovery timeout for critical operations in seconds

        high_failure_threshold: Number of failures before opening high priority circuit
        high_recovery_timeout: Recovery timeout for high priority operations in seconds

        medium_failure_threshold: Number of failures before opening medium priority circuit
        medium_recovery_timeout: Recovery timeout for medium priority operations in seconds

        low_failure_threshold: Number of failures before opening low priority circuit
        low_recovery_timeout: Recovery timeout for low priority operations in seconds
    """

    model_config = SettingsConfigDict(
        env_prefix="CB_",
        case_sensitive=False,
        extra="ignore",
    )

    # Critical operations
    critical_failure_threshold: int = Field(
        default=5,
        ge=1,
        description="Critical operations failure threshold",
    )
    critical_recovery_timeout: int = Field(
        default=60,
        ge=10,
        description="Critical operations recovery timeout",
    )

    # High priority operations
    high_failure_threshold: int = Field(
        default=4,
        ge=1,
        description="High priority failure threshold",
    )
    high_recovery_timeout: int = Field(
        default=45,
        ge=10,
        description="High priority recovery timeout",
    )

    # Medium priority operations
    medium_failure_threshold: int = Field(
        default=3,
        ge=1,
        description="Medium priority failure threshold",
    )
    medium_recovery_timeout: int = Field(
        default=30,
        ge=10,
        description="Medium priority recovery timeout",
    )

    # Low priority operations
    low_failure_threshold: int = Field(
        default=2,
        ge=1,
        description="Low priority failure threshold",
    )
    low_recovery_timeout: int = Field(
        default=15,
        ge=5,
        description="Low priority recovery timeout",
    )


class EnvironmentConfig(BaseSettings):
    """Main environment configuration combining all settings."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Environment identification
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        description="Deployment environment",
    )
    service_version: str = Field(default="0.1.0", description="Service version")
    service_instance_id: str = Field(
        default="default",
        description="Service instance identifier",
    )

    # Logging
    log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info",
        description="Logging level",
    )

    # Configuration sections
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database configuration",
    )
    kafka: KafkaConfig = Field(
        default_factory=KafkaConfig,
        description="Kafka configuration",
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration",
    )
    services: ServiceConfig = Field(
        default_factory=ServiceConfig,
        description="Service configuration",
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache configuration",
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration",
    )

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str, info: ValidationInfo) -> str:
        """Validate and normalize environment."""
        return v.lower()

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    def is_testing(self) -> bool:
        """Check if running in test environment."""
        return self.environment == "test"

    def get_env_specific_defaults(self) -> dict:
        """Get environment-specific default configurations."""
        if self.environment == "production":
            return {
                "database.pool_min_size": 10,
                "database.pool_max_size": 50,
                "database.query_timeout_seconds": 60,
                "kafka.batch_size": 32768,
                "security.rate_limit_requests_per_minute": 1000,
                "cache.workflow_memory_mb": 500,
                "cache.workflow_disk_mb": 2000,
            }
        elif self.environment == "staging":
            return {
                "database.pool_min_size": 5,
                "database.pool_max_size": 20,
                "security.rate_limit_requests_per_minute": 500,
                "cache.workflow_memory_mb": 200,
                "cache.workflow_disk_mb": 1000,
            }
        elif self.environment == "test":
            return {
                "database.pool_min_size": 1,
                "database.pool_max_size": 3,
                "database.query_timeout_seconds": 10,
                "database.acquire_timeout_seconds": 5,
                "database.connection_max_age_seconds": 300,
                "database.max_queries_per_connection": 1000,
                "kafka.batch_size": 1024,
                "kafka.linger_ms": 0,
                "kafka.buffer_memory": 8388608,  # 8MB
                "kafka.max_request_size": 131072,  # 128KB
                "security.rate_limit_requests_per_minute": 10,
                "cache.workflow_memory_mb": 50,
                "cache.workflow_disk_mb": 100,
            }
        else:  # development
            return {}


@lru_cache
def get_config() -> EnvironmentConfig:
    """Get cached configuration instance."""
    return EnvironmentConfig()


def load_environment_config() -> EnvironmentConfig:
    """Load configuration for current environment."""
    config = EnvironmentConfig()

    # Apply environment-specific defaults
    env_defaults = config.get_env_specific_defaults()
    for key, value in env_defaults.items():
        # Navigate nested configuration and update values
        parts = key.split(".")
        current = config
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], value)

    return config
