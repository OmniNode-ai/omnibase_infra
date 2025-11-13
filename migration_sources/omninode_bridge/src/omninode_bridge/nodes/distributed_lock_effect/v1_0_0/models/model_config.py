#!/usr/bin/env python3
"""
Distributed Lock Configuration Model - ONEX v2.0 Compliant.

Configuration model for NodeDistributedLockEffect.
"""

from pydantic import BaseModel, ConfigDict, Field


class ModelDistributedLockConfig(BaseModel):
    """
    Configuration for NodeDistributedLockEffect.

    Contains PostgreSQL connection settings, lock behavior configuration,
    and performance tuning parameters.
    """

    # === POSTGRESQL CONFIGURATION ===

    postgres_host: str = Field(
        default="localhost",
        description="PostgreSQL host address",
    )

    postgres_port: int = Field(
        default=5432,
        description="PostgreSQL port",
        ge=1,
        le=65535,
    )

    postgres_database: str = Field(
        ...,
        description="PostgreSQL database name",
        min_length=1,
    )

    postgres_user: str = Field(
        default="postgres",
        description="PostgreSQL username",
    )

    postgres_password: str = Field(
        default="",
        description="PostgreSQL password",
    )

    postgres_min_connections: int = Field(
        default=5,
        description="Minimum connection pool size",
        ge=1,
    )

    postgres_max_connections: int = Field(
        default=20,
        description="Maximum connection pool size",
        ge=1,
    )

    postgres_connection_timeout: float = Field(
        default=10.0,
        description="PostgreSQL connection timeout in seconds",
        gt=0.0,
    )

    # === LOCK BEHAVIOR CONFIGURATION ===

    default_lease_duration: float = Field(
        default=30.0,
        description="Default lock lease duration in seconds",
        gt=0.0,
        le=3600.0,
    )

    max_lease_duration: float = Field(
        default=3600.0,
        description="Maximum allowed lock lease duration in seconds",
        gt=0.0,
    )

    cleanup_interval: float = Field(
        default=60.0,
        description="Interval for expired lock cleanup task in seconds",
        gt=0.0,
    )

    max_acquire_attempts: int = Field(
        default=3,
        description="Maximum lock acquisition attempts",
        ge=1,
        le=10,
    )

    acquire_retry_delay: float = Field(
        default=0.1,
        description="Delay between lock acquisition retries in seconds",
        ge=0.0,
        le=10.0,
    )

    # === PERFORMANCE TUNING ===

    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection",
    )

    enable_health_checks: bool = Field(
        default=True,
        description="Enable health check endpoints",
    )

    query_timeout: float = Field(
        default=30.0,
        description="Database query timeout in seconds",
        gt=0.0,
    )

    # === KAFKA EVENT PUBLISHING ===

    kafka_enabled: bool = Field(
        default=True,
        description="Enable Kafka event publishing",
    )

    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers",
    )

    kafka_topic_lock_acquired: str = Field(
        default="dev.omninode-bridge.lock.acquired.v1",
        description="Kafka topic for lock acquired events",
    )

    kafka_topic_lock_released: str = Field(
        default="dev.omninode-bridge.lock.released.v1",
        description="Kafka topic for lock released events",
    )

    kafka_topic_lock_expired: str = Field(
        default="dev.omninode-bridge.lock.expired.v1",
        description="Kafka topic for lock expired events",
    )

    # === LOGGING ===

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    log_queries: bool = Field(
        default=False,
        description="Log all database queries (for debugging)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "postgres_host": "192.168.86.200",
                "postgres_port": 5436,
                "postgres_database": "omninode_bridge",
                "postgres_user": "postgres",
                "postgres_password": "***",
                "postgres_min_connections": 5,
                "postgres_max_connections": 20,
                "postgres_connection_timeout": 10.0,
                "default_lease_duration": 30.0,
                "max_lease_duration": 3600.0,
                "cleanup_interval": 60.0,
                "max_acquire_attempts": 3,
                "acquire_retry_delay": 0.1,
                "enable_metrics": True,
                "enable_health_checks": True,
                "query_timeout": 30.0,
                "kafka_enabled": True,
                "kafka_bootstrap_servers": "192.168.86.200:9092",
                "kafka_topic_lock_acquired": "dev.omninode-bridge.lock.acquired.v1",
                "kafka_topic_lock_released": "dev.omninode-bridge.lock.released.v1",
                "kafka_topic_lock_expired": "dev.omninode-bridge.lock.expired.v1",
                "log_level": "INFO",
                "log_queries": False,
            }
        }
    )


__all__ = ["ModelDistributedLockConfig"]
