"""
Configuration for ONEX node generation CLI.

Provides centralized configuration with environment variable support
and sensible defaults for CLI operations.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _get_kafka_bootstrap_servers() -> str:
    """Get Kafka bootstrap servers from environment."""
    # Default to remote infrastructure (resolves via /etc/hosts to 192.168.86.200:9092)
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092")


def _get_default_output_dir() -> str:
    """Get default output directory from environment."""
    return os.getenv("CODEGEN_OUTPUT_DIR", "./generated_nodes")


def _get_default_timeout_seconds() -> int:
    """Get default timeout seconds from environment."""
    return int(os.getenv("CODEGEN_TIMEOUT_SECONDS", "300"))


def _get_enable_verbose_logging() -> bool:
    """Get verbose logging setting from environment."""
    return os.getenv("CODEGEN_VERBOSE", "false").lower() in ("true", "1", "yes")


@dataclass
class CodegenCLIConfig:
    """
    Configuration for ONEX node generation CLI.

    Attributes:
        kafka_bootstrap_servers: Kafka broker addresses
        default_output_dir: Default output directory for generated nodes
        default_timeout_seconds: Default timeout for generation operations
        enable_verbose_logging: Enable verbose logging output
        kafka_consumer_group_prefix: Prefix for Kafka consumer group IDs
    """

    kafka_bootstrap_servers: str = field(default_factory=_get_kafka_bootstrap_servers)
    default_output_dir: str = field(default_factory=_get_default_output_dir)
    default_timeout_seconds: int = field(default_factory=_get_default_timeout_seconds)
    enable_verbose_logging: bool = field(default_factory=_get_enable_verbose_logging)
    kafka_consumer_group_prefix: str = "cli"

    @classmethod
    def from_env(cls) -> "CodegenCLIConfig":
        """
        Create configuration from environment variables.

        Returns:
            CodegenCLIConfig instance with values from environment
        """
        return cls()

    def with_overrides(
        self,
        kafka_bootstrap_servers: Optional[str] = None,
        default_output_dir: Optional[str] = None,
        default_timeout_seconds: Optional[int] = None,
        enable_verbose_logging: Optional[bool] = None,
    ) -> "CodegenCLIConfig":
        """
        Create a new config with specific overrides.

        Args:
            kafka_bootstrap_servers: Override Kafka bootstrap servers
            default_output_dir: Override default output directory
            default_timeout_seconds: Override default timeout
            enable_verbose_logging: Override verbose logging setting

        Returns:
            New CodegenCLIConfig instance with overrides applied
        """
        return CodegenCLIConfig(
            kafka_bootstrap_servers=kafka_bootstrap_servers
            or self.kafka_bootstrap_servers,
            default_output_dir=default_output_dir or self.default_output_dir,
            default_timeout_seconds=default_timeout_seconds
            or self.default_timeout_seconds,
            enable_verbose_logging=(
                enable_verbose_logging
                if enable_verbose_logging is not None
                else self.enable_verbose_logging
            ),
            kafka_consumer_group_prefix=self.kafka_consumer_group_prefix,
        )
