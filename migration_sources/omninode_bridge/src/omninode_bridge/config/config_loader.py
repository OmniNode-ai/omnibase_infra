"""Configuration loader for bridge nodes.

This module provides YAML-based configuration loading with:
- Hierarchical configuration merging (base + environment overrides)
- Environment variable override support
- Pydantic validation
- Type-safe configuration access
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from .settings import OrchestratorSettings, ReducerSettings, RegistrySettings


class ConfigurationError(Exception):
    """Configuration loading or validation error."""

    pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary with values to merge

    Returns:
        Merged dictionary with override values taking precedence

    Example:
        >>> base = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> override = {"a": {"c": 4, "e": 5}, "f": 6}
        >>> _deep_merge(base, override)
        {"a": {"b": 1, "c": 4, "e": 5}, "d": 3, "f": 6}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = _deep_merge(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


def _load_yaml_file(file_path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed YAML content as dictionary

    Raises:
        ConfigurationError: If file doesn't exist or YAML is invalid
    """
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            content = yaml.safe_load(f)
            if content is None:
                return {}
            if not isinstance(content, dict):
                raise ConfigurationError(
                    f"Invalid YAML format in {file_path}: expected dictionary, got {type(content)}"
                )
            return content
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML file {file_path}: {e}") from e
    except OSError as e:
        raise ConfigurationError(f"Failed to read file {file_path}: {e}") from e


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to configuration.

    Environment variables follow the pattern:
    - BRIDGE_<KEY>=value (for top-level scalars)
    - BRIDGE_<SECTION>_<KEY>=value (for nested values)
    - BRIDGE_<SECTION>_<SUBSECTION>_<KEY>=value (for deeply nested values)

    Examples:
        - BRIDGE_ENVIRONMENT=production (top-level scalar override)
        - BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS=200
        - BRIDGE_DATABASE_HOST=postgres-prod
        - BRIDGE_KAFKA_BOOTSTRAP_SERVERS=kafka:9092

    Args:
        config: Base configuration dictionary

    Returns:
        Configuration with environment variable overrides applied
    """
    import copy

    result = copy.deepcopy(config)

    # Look for environment variables with BRIDGE_ prefix
    for env_key, env_value in os.environ.items():
        if not env_key.startswith("BRIDGE_"):
            continue

        # Parse environment variable key
        # Format: BRIDGE_<KEY> for top-level scalars
        # Format: BRIDGE_<SECTION>_<REMAINDER> for nested values
        # where <REMAINDER> can contain underscores as part of the key name
        parts = env_key[7:].split("_", 1)  # Remove "BRIDGE_" prefix, split once

        # Handle top-level scalar override (e.g., BRIDGE_ENVIRONMENT=production)
        if len(parts) == 1:
            key = parts[0].lower()
            result[key] = _convert_env_value(env_value)
            continue

        section = parts[0].lower()
        remainder = parts[1].lower()

        # Try to find the right level to set the value
        if section in result and isinstance(result[section], dict):
            # Smart navigation: try to match nested dict structure
            remainder_parts = remainder.split("_")
            current = result[section]
            nav_path = []

            # Navigate through nested dicts as far as possible
            for i, part in enumerate(remainder_parts):
                if part in current and isinstance(current[part], dict):
                    # Found a nested dict, navigate into it
                    current = current[part]
                    nav_path.append(part)
                else:
                    # No more nested dicts, treat rest as key name with underscores
                    final_key = "_".join(remainder_parts[i:])
                    current[final_key] = _convert_env_value(env_value)
                    break
            else:
                # We navigated through all parts without finding a key
                # This shouldn't happen if config structure is correct
                # Just set the last part as key
                if remainder_parts:
                    current[remainder_parts[-1]] = _convert_env_value(env_value)
        elif section in result and not isinstance(result[section], dict):
            # Section exists but is a scalar - need to convert to dict for nested override
            # This handles the edge case where a scalar needs to become a nested structure
            result[section] = {remainder: _convert_env_value(env_value)}
        elif section not in result:
            # Section doesn't exist, create it
            result[section] = {remainder: _convert_env_value(env_value)}

    return result


def _convert_env_value(value: str) -> Any:
    """Convert environment variable string to appropriate Python type.

    Args:
        value: Environment variable value as string

    Returns:
        Converted value (bool, int, float, list, or str)

    Examples:
        - "true" -> True
        - "false" -> False
        - "123" -> 123
        - "3.14" -> 3.14
        - "a,b,c" -> ["a", "b", "c"]
        - "hello" -> "hello"
    """
    # Boolean conversion
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False

    # Numeric conversion
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # List conversion (comma-separated)
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]

    # String value
    return value


def load_node_config(
    node_type: Literal["orchestrator", "reducer", "registry"],
    environment: str | None = None,
    config_dir: Path | str | None = None,
) -> OrchestratorSettings | ReducerSettings | RegistrySettings:
    """Load and validate node configuration.

    Configuration loading hierarchy:
    1. Load base node config (orchestrator.yaml or reducer.yaml)
    2. Load environment config (development.yaml or production.yaml)
    3. Merge configurations (environment overrides base)
    4. Apply environment variable overrides
    5. Validate with Pydantic models

    Args:
        node_type: Type of node ("orchestrator" or "reducer")
        environment: Environment name (default: from ENVIRONMENT env var or "development")
        config_dir: Configuration directory (default: project_root/config)

    Returns:
        Validated configuration settings (OrchestratorSettings or ReducerSettings)

    Raises:
        ConfigurationError: If configuration loading or validation fails

    Examples:
        >>> # Load orchestrator config for development
        >>> config = load_node_config("orchestrator", environment="development")
        >>> print(config.orchestrator.max_concurrent_workflows)
        10

        >>> # Load reducer config for production
        >>> config = load_node_config("reducer", environment="production")
        >>> print(config.reducer.aggregation_batch_size)
        500

        >>> # Load with environment variable override
        >>> os.environ["BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS"] = "200"
        >>> config = load_node_config("orchestrator")
        >>> print(config.orchestrator.max_concurrent_workflows)
        200
    """
    # Determine environment
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    # Determine config directory
    if config_dir is None:
        # Default to project_root/config
        project_root = Path(__file__).parent.parent.parent.parent
        config_dir = project_root / "config"
    else:
        config_dir = Path(config_dir)

    if not config_dir.exists():
        raise ConfigurationError(f"Configuration directory not found: {config_dir}")

    # Load base node configuration
    base_config_file = config_dir / f"{node_type}.yaml"
    base_config = _load_yaml_file(base_config_file)

    # Load environment configuration
    env_config_file = config_dir / f"{environment}.yaml"
    env_config = _load_yaml_file(env_config_file)

    # Merge configurations
    merged_config = _deep_merge(base_config, env_config)

    # Apply environment variable overrides
    final_config = _apply_env_overrides(merged_config)

    # Validate with Pydantic
    try:
        if node_type == "orchestrator":
            return OrchestratorSettings(**final_config)
        elif node_type == "reducer":
            return ReducerSettings(**final_config)
        else:  # registry
            return RegistrySettings(**final_config)
    except ValidationError as e:
        raise ConfigurationError(
            f"Configuration validation failed for {node_type} node: {e}"
        ) from e


@lru_cache(maxsize=2)
def get_orchestrator_config(
    environment: str | None = None,
) -> OrchestratorSettings:
    """Get cached orchestrator configuration.

    Args:
        environment: Environment name (default: from ENVIRONMENT env var)

    Returns:
        Validated orchestrator configuration

    Raises:
        ConfigurationError: If configuration loading fails

    Example:
        >>> config = get_orchestrator_config()
        >>> print(config.orchestrator.max_concurrent_workflows)
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    return load_node_config("orchestrator", environment)


@lru_cache(maxsize=2)
def get_reducer_config(
    environment: str | None = None,
) -> ReducerSettings:
    """Get cached reducer configuration.

    Args:
        environment: Environment name (default: from ENVIRONMENT env var)

    Returns:
        Validated reducer configuration

    Raises:
        ConfigurationError: If configuration loading fails

    Example:
        >>> config = get_reducer_config()
        >>> print(config.reducer.aggregation_batch_size)
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    return load_node_config("reducer", environment)


@lru_cache(maxsize=2)
def get_registry_config(
    environment: str | None = None,
) -> RegistrySettings:
    """Get cached registry configuration.

    Args:
        environment: Environment name (default: from ENVIRONMENT env var)

    Returns:
        Validated registry configuration

    Raises:
        ConfigurationError: If configuration loading fails

    Example:
        >>> config = get_registry_config()
        >>> print(config.registry.max_registered_nodes)
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    return load_node_config("registry", environment)


def reload_config() -> None:
    """Clear configuration cache to force reload.

    Call this when configuration files change or environment variables are updated.

    Example:
        >>> # Update configuration
        >>> os.environ["BRIDGE_ORCHESTRATOR_MAX_CONCURRENT_WORKFLOWS"] = "500"
        >>> reload_config()
        >>> config = get_orchestrator_config()  # Loads fresh config
    """
    get_orchestrator_config.cache_clear()
    get_reducer_config.cache_clear()
    get_registry_config.cache_clear()


def validate_config_files(config_dir: Path | str | None = None) -> dict[str, bool]:
    """Validate all configuration files without loading them.

    Useful for pre-deployment validation and CI/CD checks.

    Args:
        config_dir: Configuration directory (default: project_root/config)

    Returns:
        Dictionary mapping config file names to validation status (True = valid)

    Example:
        >>> results = validate_config_files()
        >>> print(results)
        {
            "orchestrator.yaml": True,
            "reducer.yaml": True,
            "development.yaml": True,
            "production.yaml": True
        }
    """
    if config_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent
        config_dir = project_root / "config"
    else:
        config_dir = Path(config_dir)

    results = {}

    # Validate node configs
    for node_type in ["orchestrator", "reducer", "registry"]:
        config_file = f"{node_type}.yaml"
        try:
            _load_yaml_file(config_dir / config_file)
            results[config_file] = True
        except ConfigurationError:
            results[config_file] = False

    # Validate environment configs
    for environment in ["development", "production"]:
        config_file = f"{environment}.yaml"
        try:
            _load_yaml_file(config_dir / config_file)
            results[config_file] = True
        except ConfigurationError:
            results[config_file] = False

    return results


def get_config_info(
    node_type: Literal["orchestrator", "reducer", "registry"],
    environment: str | None = None,
) -> dict[str, Any]:
    """Get configuration information without loading full config.

    Useful for debugging and configuration inspection.

    Args:
        node_type: Type of node
        environment: Environment name

    Returns:
        Dictionary with configuration metadata

    Example:
        >>> info = get_config_info("orchestrator", "development")
        >>> print(info["config_files"])
        ["orchestrator.yaml", "development.yaml"]
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    project_root = Path(__file__).parent.parent.parent.parent
    config_dir = project_root / "config"

    base_config_file = config_dir / f"{node_type}.yaml"
    env_config_file = config_dir / f"{environment}.yaml"

    return {
        "node_type": node_type,
        "environment": environment,
        "config_dir": str(config_dir),
        "config_files": [
            f"{node_type}.yaml",
            f"{environment}.yaml",
        ],
        "files_exist": {
            f"{node_type}.yaml": base_config_file.exists(),
            f"{environment}.yaml": env_config_file.exists(),
        },
        "env_overrides_active": any(key.startswith("BRIDGE_") for key in os.environ),
    }
