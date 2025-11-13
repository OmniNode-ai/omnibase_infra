"""
Batch Size Management Module

Provides centralized access to batch size configuration across all omninode bridge components.
This module ensures consistent batch sizing and allows for environment-specific optimization.
"""

import os
from functools import lru_cache
from typing import Optional

from ..config.settings import BatchSizeConfig, EnvironmentBatchSizeConfig


class BatchSizeManager:
    """
    Centralized batch size manager for all operations.

    Provides environment-aware batch size configuration with fallback
    to sensible defaults when configuration is not available.
    """

    def __init__(
        self, environment: str = "development", config: Optional[BatchSizeConfig] = None
    ):
        """
        Initialize batch size manager.

        Args:
            environment: Environment name (development, test, staging, production)
            config: Optional batch size configuration
        """
        self.environment = environment.lower()
        self._config = config

        # Load environment-specific defaults if no config provided
        if config is None:
            self._config = self._get_environment_defaults()

    def _get_environment_defaults(self) -> BatchSizeConfig:
        """Get environment-specific default batch sizes."""
        env_config = EnvironmentBatchSizeConfig()

        env_map = {
            "development": env_config.development,
            "test": env_config.test,
            "staging": env_config.staging,
            "production": env_config.production,
        }

        return env_map.get(self.environment, env_config.development)

    @property
    def config(self) -> BatchSizeConfig:
        """Get current batch size configuration."""
        return self._config

    # Database operations
    @property
    def database_batch_size(self) -> int:
        """Get database batch insert/update size."""
        return self._get_config_value(
            "database_batch_size",
            int(os.getenv("DATABASE_BATCH_SIZE", self._config.database_batch_size)),
        )

    @property
    def database_query_limit(self) -> int:
        """Get maximum rows per query."""
        return self._get_config_value(
            "database_query_limit",
            int(os.getenv("DATABASE_QUERY_LIMIT", self._config.database_query_limit)),
        )

    @property
    def database_statement_cache_size(self) -> int:
        """Get prepared statement cache size."""
        return self._get_config_value(
            "database_statement_cache_size",
            int(
                os.getenv(
                    "DATABASE_STATEMENT_CACHE_SIZE",
                    self._config.database_statement_cache_size,
                )
            ),
        )

    # Kafka operations
    @property
    def kafka_producer_batch_size(self) -> int:
        """Get Kafka producer batch size in bytes."""
        return self._get_config_value(
            "kafka_producer_batch_size",
            int(os.getenv("KAFKA_BATCH_SIZE", self._config.kafka_producer_batch_size)),
        )

    @property
    def kafka_consumer_max_poll_records(self) -> int:
        """Get maximum records per Kafka poll."""
        return self._get_config_value(
            "kafka_consumer_max_poll_records",
            int(
                os.getenv(
                    "KAFKA_MAX_POLL_RECORDS",
                    self._config.kafka_consumer_max_poll_records,
                )
            ),
        )

    @property
    def kafka_consumer_fetch_min_bytes(self) -> int:
        """Get minimum bytes per Kafka fetch."""
        return self._get_config_value(
            "kafka_consumer_fetch_min_bytes",
            int(
                os.getenv(
                    "KAFKA_FETCH_MIN_BYTES", self._config.kafka_consumer_fetch_min_bytes
                )
            ),
        )

    @property
    def kafka_consumer_fetch_max_bytes(self) -> int:
        """Get maximum bytes per Kafka fetch."""
        return self._get_config_value(
            "kafka_consumer_fetch_max_bytes",
            int(
                os.getenv(
                    "KAFKA_FETCH_MAX_BYTES", self._config.kafka_consumer_fetch_max_bytes
                )
            ),
        )

    # Redis operations
    @property
    def redis_batch_size(self) -> int:
        """Get Redis operation batch size."""
        return self._get_config_value(
            "redis_batch_size",
            int(os.getenv("REDIS_BATCH_SIZE", self._config.redis_batch_size)),
        )

    @property
    def redis_pipeline_size(self) -> int:
        """Get Redis pipeline size."""
        return self._get_config_value(
            "redis_pipeline_size",
            int(os.getenv("REDIS_PIPELINE_SIZE", self._config.redis_pipeline_size)),
        )

    # Node operations
    @property
    def orchestrator_batch_size(self) -> int:
        """Get orchestrator task batch size."""
        return self._get_config_value(
            "orchestrator_batch_size",
            int(
                os.getenv(
                    "ORCHESTRATOR_BATCH_SIZE", self._config.orchestrator_batch_size
                )
            ),
        )

    @property
    def reducer_batch_size(self) -> int:
        """Get reducer aggregation batch size."""
        return self._get_config_value(
            "reducer_batch_size",
            int(os.getenv("REDUCER_BATCH_SIZE", self._config.reducer_batch_size)),
        )

    @property
    def registry_batch_size(self) -> int:
        """Get registry operation batch size."""
        return self._get_config_value(
            "registry_batch_size",
            int(os.getenv("REGISTRY_BATCH_SIZE", self._config.registry_batch_size)),
        )

    # Performance optimization
    @property
    def performance_task_batch_size(self) -> int:
        """Get performance task batch size."""
        return self._get_config_value(
            "performance_task_batch_size",
            int(
                os.getenv(
                    "PERFORMANCE_TASK_BATCH_SIZE",
                    self._config.performance_task_batch_size,
                )
            ),
        )

    @property
    def processing_buffer_size(self) -> int:
        """Get event processing buffer size."""
        return self._get_config_value(
            "processing_buffer_size",
            int(
                os.getenv("PROCESSING_BUFFER_SIZE", self._config.processing_buffer_size)
            ),
        )

    # Cleanup operations
    @property
    def cleanup_batch_size(self) -> int:
        """Get cleanup operation batch size."""
        return self._get_config_value(
            "cleanup_batch_size",
            int(os.getenv("CLEANUP_BATCH_SIZE", self._config.cleanup_batch_size)),
        )

    @property
    def retention_cleanup_batch_size(self) -> int:
        """Get retention cleanup batch size."""
        return self._get_config_value(
            "retention_cleanup_batch_size",
            int(
                os.getenv(
                    "RETENTION_CLEANUP_BATCH_SIZE",
                    self._config.retention_cleanup_batch_size,
                )
            ),
        )

    # File operations
    @property
    def file_processing_batch_size(self) -> int:
        """Get file processing batch size."""
        return self._get_config_value(
            "file_processing_batch_size",
            int(
                os.getenv(
                    "FILE_PROCESSING_BATCH_SIZE",
                    self._config.file_processing_batch_size,
                )
            ),
        )

    @property
    def metadata_extraction_batch_size(self) -> int:
        """Get metadata extraction batch size."""
        return self._get_config_value(
            "metadata_extraction_batch_size",
            int(
                os.getenv(
                    "METADATA_EXTRACTION_BATCH_SIZE",
                    self._config.metadata_extraction_batch_size,
                )
            ),
        )

    def _get_config_value(self, attr_name: str, fallback_value: int) -> int:
        """
        Get configuration value with validation.

        Args:
            attr_name: Attribute name for validation
            fallback_value: Fallback value if attribute doesn't exist

        Returns:
            Validated configuration value
        """
        if hasattr(self._config, attr_name):
            value = getattr(self._config, attr_name)
            # Ensure value is within reasonable bounds
            if value <= 0:
                return fallback_value
            return value
        return fallback_value

    def update_config(self, new_config: BatchSizeConfig) -> None:
        """
        Update batch size configuration.

        Args:
            new_config: New batch size configuration
        """
        self._config = new_config

    def get_summary(self) -> dict:
        """
        Get a summary of current batch size configuration.

        Returns:
            Dictionary containing current batch sizes
        """
        return {
            "environment": self.environment,
            "database_operations": {
                "batch_size": self.database_batch_size,
                "query_limit": self.database_query_limit,
                "statement_cache_size": self.database_statement_cache_size,
            },
            "kafka_operations": {
                "producer_batch_size": self.kafka_producer_batch_size,
                "consumer_max_poll_records": self.kafka_consumer_max_poll_records,
                "fetch_min_bytes": self.kafka_consumer_fetch_min_bytes,
                "fetch_max_bytes": self.kafka_consumer_fetch_max_bytes,
            },
            "redis_operations": {
                "batch_size": self.redis_batch_size,
                "pipeline_size": self.redis_pipeline_size,
            },
            "node_operations": {
                "orchestrator": self.orchestrator_batch_size,
                "reducer": self.reducer_batch_size,
                "registry": self.registry_batch_size,
            },
            "performance": {
                "task_batch_size": self.performance_task_batch_size,
                "processing_buffer_size": self.processing_buffer_size,
            },
            "cleanup": {
                "batch_size": self.cleanup_batch_size,
                "retention_batch_size": self.retention_cleanup_batch_size,
            },
            "file_operations": {
                "processing_batch_size": self.file_processing_batch_size,
                "metadata_extraction_batch_size": self.metadata_extraction_batch_size,
            },
        }


# Global instance for backward compatibility
_global_batch_manager: Optional[BatchSizeManager] = None


@lru_cache(maxsize=1)
def get_batch_manager(
    environment: str = "development", config: Optional[BatchSizeConfig] = None
) -> BatchSizeManager:
    """
    Get global batch size manager instance.

    Args:
        environment: Environment name
        config: Optional batch size configuration

    Returns:
        BatchSizeManager instance
    """
    global _global_batch_manager
    if (
        _global_batch_manager is None
        or _global_batch_manager.environment != environment
    ):
        _global_batch_manager = BatchSizeManager(environment=environment, config=config)
    return _global_batch_manager


def set_batch_manager(manager: BatchSizeManager) -> None:
    """
    Set global batch size manager instance.

    Args:
        manager: BatchSizeManager instance
    """
    global _global_batch_manager
    _global_batch_manager = manager


# Convenience functions for backward compatibility
def get_database_batch_size(environment: str = "development") -> int:
    """Get database batch size for environment."""
    return get_batch_manager(environment).database_batch_size


def get_kafka_batch_size(environment: str = "development") -> int:
    """Get Kafka producer batch size for environment."""
    return get_batch_manager(environment).kafka_producer_batch_size


def get_redis_batch_size(environment: str = "development") -> int:
    """Get Redis batch size for environment."""
    return get_batch_manager(environment).redis_batch_size


def get_orchestrator_batch_size(environment: str = "development") -> int:
    """Get orchestrator batch size for environment."""
    return get_batch_manager(environment).orchestrator_batch_size


def get_reducer_batch_size(environment: str = "development") -> int:
    """Get reducer batch size for environment."""
    return get_batch_manager(environment).reducer_batch_size


def get_cleanup_batch_size(environment: str = "development") -> int:
    """Get cleanup batch size for environment."""
    return get_batch_manager(environment).cleanup_batch_size


def get_file_processing_batch_size(environment: str = "development") -> int:
    """Get file processing batch size for environment."""
    return get_batch_manager(environment).file_processing_batch_size
