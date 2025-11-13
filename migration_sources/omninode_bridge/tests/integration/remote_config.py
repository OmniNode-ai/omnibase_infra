#!/usr/bin/env python3
"""
Remote Test Configuration for OmniNode Bridge Integration Tests.

This module provides centralized configuration for integration tests that can run
against both local (Docker Compose) and remote (distributed system) deployments.

Environment Variables:
    TEST_MODE: 'local' or 'remote' (default: 'local')
    KAFKA_BOOTSTRAP_SERVERS: Kafka broker address
    POSTGRES_HOST: PostgreSQL host
    POSTGRES_PORT: PostgreSQL port
    POSTGRES_DATABASE: Database name
    POSTGRES_USER: Database user
    POSTGRES_PASSWORD: Database password
    CONSUL_HOST: Consul host
    CONSUL_PORT: Consul port
    METADATA_STAMPING_URL: Metadata stamping service URL
    ONEXTREE_URL: OnexTree service URL

Usage:
    # Local testing (default)
    from tests.integration.remote_config import get_test_config
    config = get_test_config()

    # Remote testing (via environment variable)
    export TEST_MODE=remote
    config = get_test_config()  # Uses remote.env configuration
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class TestConfiguration:
    """
    Test configuration for integration tests.

    Supports both local (Docker Compose) and remote (distributed system) deployments.
    """

    # Test mode: 'local' or 'remote'
    test_mode: str

    # Kafka configuration
    kafka_bootstrap_servers: str

    # PostgreSQL configuration
    postgres_host: str
    postgres_port: int
    postgres_database: str
    postgres_user: str
    postgres_password: str

    # Consul configuration
    consul_host: str
    consul_port: int

    # Service URLs
    metadata_stamping_url: str
    onextree_url: str
    hook_receiver_url: str
    orchestrator_url: str
    reducer_url: str

    # Optional fields with defaults
    kafka_timeout_ms: int = 5000
    postgres_connection_timeout: int = 10
    test_timeout_seconds: int = 30
    service_startup_timeout_seconds: int = 60
    test_namespaces: list[str] = None

    def __post_init__(self):
        """Initialize default values for list fields."""
        if self.test_namespaces is None:
            self.test_namespaces = [
                "test.kafka.integration",
                "test.orchestrator.integration",
                "test.reducer.integration",
                "omninode.services.metadata",
            ]

    @property
    def postgres_connection_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )

    @property
    def is_remote_mode(self) -> bool:
        """Check if running in remote mode."""
        return self.test_mode == "remote"

    @property
    def is_local_mode(self) -> bool:
        """Check if running in local mode."""
        return self.test_mode == "local"

    def get_kafka_config(self) -> dict:
        """Get Kafka consumer/producer configuration."""
        return {
            "bootstrap_servers": self.kafka_bootstrap_servers,
            "request_timeout_ms": self.kafka_timeout_ms,
            "metadata_max_age_ms": 5000,
        }

    def get_postgres_config(self) -> dict:
        """Get PostgreSQL connection configuration."""
        return {
            "host": self.postgres_host,
            "port": self.postgres_port,
            "database": self.postgres_database,
            "user": self.postgres_user,
            "password": self.postgres_password,
            "timeout": self.postgres_connection_timeout,
        }


def load_test_environment() -> str:
    """
    Load test environment configuration from .env files.

    Returns:
        Test mode ('local' or 'remote')
    """
    # Get test mode from environment (default: local)
    test_mode = os.getenv("TEST_MODE", "local").lower()

    # Find project root (where .env files are located)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent

    # Load appropriate .env file based on test mode
    if test_mode == "remote":
        env_file = project_root / "remote.env"
        if not env_file.exists():
            raise FileNotFoundError(
                f"Remote environment file not found: {env_file}\n"
                "Please create remote.env with remote system configuration."
            )
        load_dotenv(env_file, override=True)
        print(f"✅ Loaded remote configuration from: {env_file}")
    else:
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
            print(f"✅ Loaded local configuration from: {env_file}")
        else:
            print("ℹ  No .env file found, using environment variables")  # noqa: RUF001

    return test_mode


def get_test_config(
    test_mode: Optional[str] = None,
    override_kafka_servers: Optional[str] = None,
    override_postgres_host: Optional[str] = None,
) -> TestConfiguration:
    """
    Get test configuration for integration tests.

    Args:
        test_mode: Override test mode ('local' or 'remote')
        override_kafka_servers: Override Kafka bootstrap servers
        override_postgres_host: Override PostgreSQL host

    Returns:
        TestConfiguration instance

    Environment Variables:
        TEST_MODE: 'local' or 'remote' (default: 'local')
        KAFKA_BOOTSTRAP_SERVERS: Kafka broker address
        POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DATABASE: PostgreSQL configuration
        CONSUL_HOST, CONSUL_PORT: Consul configuration

    Example:
        # Local testing
        config = get_test_config()
        assert config.is_local_mode

        # Remote testing
        os.environ['TEST_MODE'] = 'remote'
        config = get_test_config()
        assert config.is_remote_mode
    """
    # Load environment configuration
    detected_test_mode = load_test_environment()
    test_mode = test_mode or detected_test_mode

    # Kafka configuration
    if test_mode == "remote":
        # Remote mode: Use external advertised listeners
        kafka_servers = (
            os.getenv("KAFKA_ADVERTISED_HOST", "192.168.86.200")
            + ":"
            + os.getenv("KAFKA_ADVERTISED_PORT", "29102")
        )
    else:
        # Local mode: Use Docker network or localhost
        kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")

    kafka_servers = override_kafka_servers or kafka_servers

    # PostgreSQL configuration
    postgres_host = override_postgres_host or os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = int(os.getenv("POSTGRES_PORT", "5436"))

    # For remote mode, use remote host if POSTGRES_HOST is Docker network name
    if test_mode == "remote" and postgres_host == "omninode-bridge-postgres":
        postgres_host = "192.168.86.200"

    # Consul configuration
    consul_host = os.getenv("CONSUL_HOST", "localhost")
    if test_mode == "remote" and consul_host == "omninode-bridge-consul":
        consul_host = "192.168.86.200"

    consul_port = int(os.getenv("CONSUL_PORT", "28500"))

    # Service URLs
    base_url = "http://192.168.86.200" if test_mode == "remote" else "http://localhost"

    metadata_stamping_url = os.getenv(
        "METADATA_STAMPING_URL",
        f"{base_url}:{os.getenv('METADATA_STAMPING_PORT', '8057')}",
    )
    onextree_url = os.getenv(
        "ONEXTREE_URL", f"{base_url}:{os.getenv('ONEXTREE_PORT', '8058')}"
    )
    hook_receiver_url = os.getenv(
        "HOOK_RECEIVER_URL",
        f"{base_url}:{os.getenv('HOOK_RECEIVER_PORT', '8001')}",
    )
    orchestrator_url = os.getenv(
        "ORCHESTRATOR_URL", f"{base_url}:{os.getenv('ORCHESTRATOR_PORT', '8060')}"
    )
    reducer_url = os.getenv(
        "REDUCER_URL", f"{base_url}:{os.getenv('REDUCER_PORT', '8061')}"
    )

    return TestConfiguration(
        test_mode=test_mode,
        kafka_bootstrap_servers=kafka_servers,
        postgres_host=postgres_host,
        postgres_port=postgres_port,
        postgres_database=os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
        postgres_user=os.getenv("POSTGRES_USER", "postgres"),
        postgres_password=os.getenv(
            "POSTGRES_PASSWORD", "omninode-bridge-postgres-dev-2024"
        ),
        consul_host=consul_host,
        consul_port=consul_port,
        metadata_stamping_url=metadata_stamping_url,
        onextree_url=onextree_url,
        hook_receiver_url=hook_receiver_url,
        orchestrator_url=orchestrator_url,
        reducer_url=reducer_url,
    )


# Global configuration instance (lazy loaded)
_test_config: Optional[TestConfiguration] = None


def get_global_test_config() -> TestConfiguration:
    """
    Get global test configuration instance.

    This is cached for performance - call once per test session.

    Returns:
        TestConfiguration instance
    """
    global _test_config
    if _test_config is None:
        _test_config = get_test_config()
    return _test_config


def reset_global_config():
    """Reset global configuration (for testing)."""
    global _test_config
    _test_config = None


# Convenience functions for common use cases
def get_kafka_bootstrap_servers() -> str:
    """Get Kafka bootstrap servers for current test mode."""
    return get_global_test_config().kafka_bootstrap_servers


def get_postgres_connection_url() -> str:
    """Get PostgreSQL connection URL for current test mode."""
    return get_global_test_config().postgres_connection_url


def is_remote_testing() -> bool:
    """Check if running in remote testing mode."""
    return get_global_test_config().is_remote_mode


def is_local_testing() -> bool:
    """Check if running in local testing mode."""
    return get_global_test_config().is_local_mode


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration loading...")
    print(f"Test mode: {os.getenv('TEST_MODE', 'local')}")

    config = get_test_config()
    print("\n📋 Test Configuration:")
    print(f"  Mode: {config.test_mode}")
    print(f"  Kafka: {config.kafka_bootstrap_servers}")
    print(f"  PostgreSQL: {config.postgres_host}:{config.postgres_port}")
    print(f"  Consul: {config.consul_host}:{config.consul_port}")
    print(f"  Metadata Stamping: {config.metadata_stamping_url}")
    print(f"  OnexTree: {config.onextree_url}")
    print(f"  Connection URL: {config.postgres_connection_url}")
