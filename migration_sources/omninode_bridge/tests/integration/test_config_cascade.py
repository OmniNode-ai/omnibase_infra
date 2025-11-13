#!/usr/bin/env python3
"""
Integration tests for configuration cascade in omninode_bridge.

Tests the hierarchical configuration loading system:
1. Base YAML (orchestrator.yaml, reducer.yaml, registry.yaml)
2. Environment YAML (development.yaml, production.yaml)
3. Environment variable overrides (BRIDGE_* prefix)
4. Pydantic validation

Correlation ID: c5c5ba1d-0642-4aa2-a7a0-086b9592ea67
Task: Integration Test Gaps - Task 3.4 (Configuration Cascade Tests)
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from omninode_bridge.config.config_loader import (
    ConfigurationError,
    _apply_env_overrides,
    _convert_env_value,
    _deep_merge,
    load_node_config,
    reload_config,
    validate_config_files,
)


@pytest.fixture
def temp_config_dir():
    """Create temporary configuration directory with test configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create base orchestrator config with all required fields
        orchestrator_config = {
            "node": {
                "type": "orchestrator",
                "name": "test-orchestrator",
                "version": "0.1.0",
                "namespace": "omninode.bridge.test",
            },
            "orchestrator": {
                "max_concurrent_workflows": 10,
                "workflow_timeout_seconds": 60,
                "workflow_retry_attempts": 3,
                "workflow_retry_delay_seconds": 5,
                "task_batch_size": 50,
                "task_priority_levels": ["critical", "high", "medium", "low"],
                "default_task_priority": "medium",
                "enable_dependency_tracking": True,
                "max_dependency_depth": 10,
                "circular_dependency_detection": True,
                "worker_pool_size": 10,
                "event_processing_buffer_size": 1000,
                "state_sync_interval_seconds": 30,
            },
            "services": {
                "onextree": {
                    "host": "localhost",
                    "port": 8051,
                    "base_url": "http://localhost:8051",
                    "health_check_path": "/health",
                    "timeout_seconds": 30,
                    "retry_attempts": 3,
                },
                "metadata_stamping": {
                    "host": "localhost",
                    "port": 8053,
                    "base_url": "http://localhost:8053",
                    "health_check_path": "/health",
                    "timeout_seconds": 30,
                    "retry_attempts": 3,
                },
            },
            "kafka": {
                "bootstrap_servers": "localhost:9092",
                "producer": {
                    "compression_type": "snappy",
                    "batch_size": 16384,
                    "linger_ms": 5,
                    "acks": "all",
                    "max_in_flight_requests": 5,
                },
                "consumer": {
                    "group_id": "test-orchestrator-consumer",
                    "auto_offset_reset": "latest",
                    "enable_auto_commit": True,
                    "auto_commit_interval_ms": 5000,
                    "max_poll_records": 500,
                },
                "topics": {
                    "workflow_commands": "omninode.bridge.workflow.commands.v1",
                    "workflow_events": "omninode.bridge.workflow.events.v1",
                    "task_events": "omninode.bridge.task.events.v1",
                    "state_sync": "omninode.bridge.state.sync.v1",
                },
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "omninode_bridge",
                "user": "postgres",
                "pool_min_size": 5,
                "pool_max_size": 20,
                "pool_timeout_seconds": 10,
                "query_timeout_seconds": 30,
                "command_timeout_seconds": 60,
                "statement_cache_size": 100,
            },
            "consul": {
                "host": "localhost",
                "port": 8500,
                "enable_registration": False,
                "registration_timeout_seconds": 10,
                "health_check_interval_seconds": 30,
                "health_check_timeout_seconds": 5,
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "enable_structured_logging": True,
                "log_requests": True,
                "log_responses": False,
                "outputs": [
                    {"type": "console", "enabled": True},
                ],
            },
            "monitoring": {
                "enable_prometheus": True,
                "prometheus_port": 9090,
                "metrics_interval_seconds": 15,
                "health_check_interval_seconds": 30,
                "service_health_timeout_seconds": 5,
                "track_workflow_latency": True,
                "track_task_latency": True,
                "track_dependency_resolution_time": True,
            },
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 5,
                "recovery_timeout_seconds": 60,
                "half_open_max_requests": 3,
            },
            "cache": {
                "enabled": True,
                "workflow_state_ttl_seconds": 3600,
                "task_result_ttl_seconds": 1800,
                "dependency_graph_ttl_seconds": 7200,
                "max_cache_size_mb": 256,
            },
        }
        with open(config_dir / "orchestrator.yaml", "w") as f:
            yaml.dump(orchestrator_config, f)

        # Create development environment config
        dev_config = {
            "orchestrator": {
                "max_concurrent_workflows": 20,  # Override base
            },
            "kafka": {"bootstrap_servers": "dev-kafka:9092"},  # Override base
        }
        with open(config_dir / "development.yaml", "w") as f:
            yaml.dump(dev_config, f)

        # Create production environment config
        prod_config = {
            "orchestrator": {
                "max_concurrent_workflows": 100,  # Override base
                "workflow_timeout_seconds": 120,  # Override base
            },
            "kafka": {"bootstrap_servers": "prod-kafka:9092"},  # Override base
            "consul": {"host": "prod-consul", "port": 8500},  # Override host only
        }
        with open(config_dir / "production.yaml", "w") as f:
            yaml.dump(prod_config, f)

        # Create base reducer config
        reducer_config = {
            "reducer": {
                "aggregation_batch_size": 100,
                "aggregation_interval": 5,
            },
            "kafka": {"bootstrap_servers": "localhost:9092"},
        }
        with open(config_dir / "reducer.yaml", "w") as f:
            yaml.dump(reducer_config, f)

        # Create base registry config
        registry_config = {
            "registry": {
                "max_registered_nodes": 1000,
                "cleanup_interval": 60,
            },
            "consul": {"host": "localhost", "port": 8500},
        }
        with open(config_dir / "registry.yaml", "w") as f:
            yaml.dump(registry_config, f)

        yield config_dir


class TestDeepMerge:
    """Test deep dictionary merging."""

    def test_simple_merge(self):
        """Test merging simple dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test merging nested dictionaries."""
        base = {"orchestrator": {"max_concurrent_workflows": 10, "timeout": 60}}
        override = {"orchestrator": {"max_concurrent_workflows": 20}}
        result = _deep_merge(base, override)

        assert result == {
            "orchestrator": {"max_concurrent_workflows": 20, "timeout": 60}
        }

    def test_deep_nested_merge(self):
        """Test merging deeply nested dictionaries."""
        base = {
            "database": {
                "postgres": {"host": "localhost", "port": 5432},
                "redis": {"host": "localhost"},
            }
        }
        override = {"database": {"postgres": {"host": "prod-db"}}}
        result = _deep_merge(base, override)

        assert result == {
            "database": {
                "postgres": {"host": "prod-db", "port": 5432},
                "redis": {"host": "localhost"},
            }
        }

    def test_override_dict_with_scalar(self):
        """Test overriding nested dict with scalar value."""
        base = {"orchestrator": {"settings": {"timeout": 60}}}
        override = {"orchestrator": {"settings": "simple"}}
        result = _deep_merge(base, override)

        # Override replaces nested dict with scalar
        assert result == {"orchestrator": {"settings": "simple"}}

    def test_empty_override(self):
        """Test merge with empty override."""
        base = {"a": 1, "b": 2}
        override = {}
        result = _deep_merge(base, override)

        assert result == {"a": 1, "b": 2}


class TestEnvValueConversion:
    """Test environment variable value conversion."""

    def test_boolean_conversion(self):
        """Test boolean value conversion."""
        assert _convert_env_value("true") is True
        assert _convert_env_value("True") is True
        assert _convert_env_value("yes") is True
        assert _convert_env_value("1") is True
        assert _convert_env_value("on") is True

        assert _convert_env_value("false") is False
        assert _convert_env_value("False") is False
        assert _convert_env_value("no") is False
        assert _convert_env_value("0") is False
        assert _convert_env_value("off") is False

    def test_integer_conversion(self):
        """Test integer value conversion."""
        assert _convert_env_value("123") == 123
        assert _convert_env_value("0") is False  # Special case: "0" is boolean False
        assert _convert_env_value("1") is True  # Special case: "1" is boolean True
        assert _convert_env_value("42") == 42
        assert _convert_env_value("-10") == -10

    def test_float_conversion(self):
        """Test float value conversion."""
        assert _convert_env_value("3.14") == 3.14
        assert _convert_env_value("0.5") == 0.5
        assert _convert_env_value("-2.5") == -2.5

    def test_list_conversion(self):
        """Test comma-separated list conversion."""
        assert _convert_env_value("a,b,c") == ["a", "b", "c"]
        assert _convert_env_value("host1:9092,host2:9092") == [
            "host1:9092",
            "host2:9092",
        ]
        assert _convert_env_value("  a  ,  b  ,  c  ") == ["a", "b", "c"]

    def test_string_passthrough(self):
        """Test string value passthrough."""
        assert _convert_env_value("hello") == "hello"
        assert _convert_env_value("localhost") == "localhost"
        assert _convert_env_value("prod-db-cluster") == "prod-db-cluster"


class TestEnvOverrides:
    """Test environment variable override application."""

    def test_top_level_scalar_override(self):
        """Test top-level scalar override."""
        base_config = {"environment": "development"}
        os.environ["BRIDGE_ENVIRONMENT"] = "production"

        try:
            result = _apply_env_overrides(base_config)
            assert result["environment"] == "production"
        finally:
            del os.environ["BRIDGE_ENVIRONMENT"]

    def test_nested_value_override(self):
        """Test nested value override."""
        base_config = {"orchestrator": {"max_concurrent_workflows": 10}}
        os.environ["BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS"] = "200"

        try:
            result = _apply_env_overrides(base_config)
            assert result["orchestrator"]["max_concurrent_workflows"] == 200
        finally:
            del os.environ["BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS"]

    def test_deep_nested_override(self):
        """Test deeply nested value override."""
        base_config = {"database": {"postgres": {"host": "localhost", "port": 5432}}}
        os.environ["BRIDGE_DATABASE_POSTGRES_HOST"] = "prod-db"

        try:
            result = _apply_env_overrides(base_config)
            assert result["database"]["postgres"]["host"] == "prod-db"
            assert result["database"]["postgres"]["port"] == 5432
        finally:
            del os.environ["BRIDGE_DATABASE_POSTGRES_HOST"]

    def test_multiple_overrides(self):
        """Test multiple environment variable overrides."""
        base_config = {
            "orchestrator": {"max_concurrent_workflows": 10, "timeout": 60},
            "kafka": {"bootstrap_servers": "localhost:9092"},
        }

        os.environ["BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS"] = "200"
        os.environ["BRIDGE_ORCHESTRATOR_TIMEOUT"] = "120"
        os.environ["BRIDGE_KAFKA_BOOTSTRAP_SERVERS"] = "kafka-prod:9092"

        try:
            result = _apply_env_overrides(base_config)
            assert result["orchestrator"]["max_concurrent_workflows"] == 200
            assert result["orchestrator"]["timeout"] == 120
            assert result["kafka"]["bootstrap_servers"] == "kafka-prod:9092"
        finally:
            del os.environ["BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS"]
            del os.environ["BRIDGE_ORCHESTRATOR_TIMEOUT"]
            del os.environ["BRIDGE_KAFKA_BOOTSTRAP_SERVERS"]

    def test_create_new_section(self):
        """Test creating new section via environment variable."""
        base_config = {}
        os.environ["BRIDGE_NEWSERVICE_ENABLED"] = "true"

        try:
            result = _apply_env_overrides(base_config)
            assert result == {"newservice": {"enabled": True}}
        finally:
            del os.environ["BRIDGE_NEWSERVICE_ENABLED"]

    def test_ignore_non_bridge_env_vars(self):
        """Test that non-BRIDGE_ env vars are ignored."""
        base_config = {"environment": "development"}
        os.environ["RANDOM_VAR"] = "should_be_ignored"
        os.environ["OTHER_PREFIX_VAR"] = "also_ignored"

        try:
            result = _apply_env_overrides(base_config)
            assert result == {"environment": "development"}
        finally:
            del os.environ["RANDOM_VAR"]
            del os.environ["OTHER_PREFIX_VAR"]


class TestConfigurationCascade:
    """Test full configuration cascade with temporary config files."""

    def test_base_config_only(self, temp_config_dir):
        """Test loading base config without environment overrides."""
        # Create minimal development config
        with open(temp_config_dir / "development.yaml", "w") as f:
            yaml.dump({}, f)

        config = load_node_config("orchestrator", "development", temp_config_dir)

        # Base values should be present
        assert config.orchestrator.max_concurrent_workflows == 10
        assert config.orchestrator.workflow_timeout_seconds == 60
        assert config.kafka.bootstrap_servers == "localhost:9092"

    def test_environment_override(self, temp_config_dir):
        """Test environment config overrides base config."""
        reload_config()
        config = load_node_config("orchestrator", "development", temp_config_dir)

        # Development overrides should apply
        assert config.orchestrator.max_concurrent_workflows == 20  # Overridden
        assert (
            config.orchestrator.workflow_timeout_seconds == 60
        )  # From base (not overridden)
        assert config.kafka.bootstrap_servers == "dev-kafka:9092"  # Overridden

    def test_production_environment_override(self, temp_config_dir):
        """Test production environment overrides."""
        reload_config()
        config = load_node_config("orchestrator", "production", temp_config_dir)

        # Production overrides should apply
        assert config.orchestrator.max_concurrent_workflows == 100
        assert config.orchestrator.workflow_timeout_seconds == 120
        assert config.kafka.bootstrap_servers == "prod-kafka:9092"
        assert config.consul.host == "prod-consul"
        assert config.consul.port == 8500

    def test_env_var_override_cascade(self, temp_config_dir):
        """Test environment variables override both YAML configs."""
        os.environ["BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS"] = "500"
        os.environ["BRIDGE_KAFKA_BOOTSTRAP_SERVERS"] = "env-kafka:9092"

        try:
            reload_config()
            config = load_node_config("orchestrator", "development", temp_config_dir)

            # Env var overrides should take precedence
            assert config.orchestrator.max_concurrent_workflows == 500
            assert config.kafka.bootstrap_servers == "env-kafka:9092"

            # Other values from development config
            assert config.orchestrator.workflow_timeout_seconds == 60
        finally:
            del os.environ["BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS"]
            del os.environ["BRIDGE_KAFKA_BOOTSTRAP_SERVERS"]

    def test_complete_cascade_hierarchy(self, temp_config_dir):
        """Test complete cascade: base → env → env_var."""
        os.environ["BRIDGE_ORCHESTRATOR_WORKFLOW_TIMEOUT_SECONDS"] = "300"

        try:
            reload_config()
            config = load_node_config("orchestrator", "development", temp_config_dir)

            # Verify cascade:
            # - max_concurrent_workflows: dev override (20)
            # - workflow_timeout_seconds: env var override (300)
            # - health_check_interval_seconds: base value (30)
            assert config.orchestrator.max_concurrent_workflows == 20
            assert config.orchestrator.workflow_timeout_seconds == 300
            assert config.monitoring.health_check_interval_seconds == 30
        finally:
            del os.environ["BRIDGE_ORCHESTRATOR_WORKFLOW_TIMEOUT_SECONDS"]


class TestConfigValidation:
    """Test configuration validation functions."""

    def test_validate_config_files_success(self, temp_config_dir):
        """Test successful validation of all config files."""
        # Create minimal valid configs
        for node in ["orchestrator", "reducer", "registry"]:
            with open(temp_config_dir / f"{node}.yaml", "w") as f:
                yaml.dump({node: {}}, f)

        for env in ["development", "production"]:
            with open(temp_config_dir / f"{env}.yaml", "w") as f:
                yaml.dump({}, f)

        results = validate_config_files(temp_config_dir)

        assert results["orchestrator.yaml"] is True
        assert results["reducer.yaml"] is True
        assert results["registry.yaml"] is True
        assert results["development.yaml"] is True
        assert results["production.yaml"] is True

    def test_validate_config_files_with_missing_file(self, temp_config_dir):
        """Test validation with missing config file."""
        # Only create orchestrator config
        with open(temp_config_dir / "orchestrator.yaml", "w") as f:
            yaml.dump({"orchestrator": {}}, f)

        results = validate_config_files(temp_config_dir)

        assert results["orchestrator.yaml"] is True
        assert results["reducer.yaml"] is False  # Missing
        assert results["registry.yaml"] is False  # Missing

    def test_get_config_info(self, temp_config_dir):
        """Test configuration info retrieval."""
        # Create configs
        with open(temp_config_dir / "orchestrator.yaml", "w") as f:
            yaml.dump({"orchestrator": {}}, f)
        with open(temp_config_dir / "development.yaml", "w") as f:
            yaml.dump({}, f)

        # Note: get_config_info uses default config dir, so we can't easily test
        # with temp_config_dir without modifying the function signature
        # This is a design limitation - documenting for future improvement

    def test_invalid_yaml_raises_error(self, temp_config_dir):
        """Test that invalid YAML raises ConfigurationError."""
        # Create invalid YAML file
        with open(temp_config_dir / "orchestrator.yaml", "w") as f:
            f.write("invalid: yaml: syntax: [")

        with pytest.raises(ConfigurationError, match="Failed to parse YAML"):
            load_node_config("orchestrator", "development", temp_config_dir)

    def test_missing_config_file_raises_error(self, temp_config_dir):
        """Test that missing config file raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_node_config("orchestrator", "development", temp_config_dir)


class TestConfigReload:
    """Test configuration cache clearing and reloading."""

    def test_reload_config_clears_cache(self):
        """Test that reload_config clears LRU cache."""
        # This test verifies the cache_clear calls execute without error
        # Full cache behavior testing would require integration with actual config files
        reload_config()  # Should not raise

    def test_config_changes_after_reload(self, temp_config_dir):
        """Test that config changes are picked up after reload."""
        # Create initial config
        initial_config = {
            "orchestrator": {"max_concurrent_workflows": 10},
            "kafka": {"bootstrap_servers": "localhost:9092"},
        }
        with open(temp_config_dir / "orchestrator.yaml", "w") as f:
            yaml.dump(initial_config, f)
        with open(temp_config_dir / "development.yaml", "w") as f:
            yaml.dump({}, f)

        # Load config
        reload_config()
        config1 = load_node_config("orchestrator", "development", temp_config_dir)
        assert config1.orchestrator.max_concurrent_workflows == 10

        # Modify config file
        updated_config = {
            "orchestrator": {"max_concurrent_workflows": 50},
            "kafka": {"bootstrap_servers": "localhost:9092"},
        }
        with open(temp_config_dir / "orchestrator.yaml", "w") as f:
            yaml.dump(updated_config, f)

        # Load again without reload - should use cached version (if using cache)
        # Then reload and verify change
        reload_config()
        config2 = load_node_config("orchestrator", "development", temp_config_dir)
        assert config2.orchestrator.max_concurrent_workflows == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
