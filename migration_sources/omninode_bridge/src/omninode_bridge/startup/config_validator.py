"""Startup configuration validation for OmniNode Bridge services."""

import logging
import sys

from omninode_bridge.config import get_config
from omninode_bridge.config.validation import (
    check_required_environment_variables,
    validate_config,
)

logger = logging.getLogger(__name__)


class StartupValidationError(Exception):
    """Raised when startup validation fails."""

    def __init__(self, message: str, errors: list[str]):
        self.message = message
        self.errors = errors
        super().__init__(message)


def startup_validation(service_name: str = "unknown") -> bool:
    """
    Perform comprehensive startup validation.

    Args:
        service_name: Name of the service being validated

    Returns:
        True if validation passes

    Raises:
        StartupValidationError: If critical validation fails
    """
    logger.info(f"Starting configuration validation for {service_name}")

    validation_results = {
        "environment_variables": False,
        "configuration": False,
        "external_dependencies": False,
        "security": False,
    }

    errors = []
    warnings = []

    try:
        # 1. Check required environment variables
        logger.info("Validating required environment variables...")
        env_vars_ok, missing_vars = check_required_environment_variables()
        if not env_vars_ok:
            errors.extend(
                [
                    f"Missing required environment variable: {var}"
                    for var in missing_vars
                ],
            )
        else:
            validation_results["environment_variables"] = True
            logger.info("✅ Environment variables validation passed")

        # 2. Load and validate configuration
        logger.info("Loading and validating configuration...")
        try:
            config = get_config()
            config_valid, config_errors, config_warnings = validate_config(config)

            if config_errors:
                errors.extend(config_errors)
            else:
                validation_results["configuration"] = True
                logger.info("✅ Configuration validation passed")

            if config_warnings:
                warnings.extend(config_warnings)

        except Exception as e:
            errors.append(f"Configuration loading failed: {e!s}")

        # 3. Validate external dependencies
        logger.info("Validating external dependencies...")
        dependency_errors = validate_external_dependencies()
        if dependency_errors:
            errors.extend(dependency_errors)
        else:
            validation_results["external_dependencies"] = True
            logger.info("✅ External dependencies validation passed")

        # 4. Security validation
        logger.info("Validating security configuration...")
        security_errors = validate_security_configuration()
        if security_errors:
            errors.extend(security_errors)
        else:
            validation_results["security"] = True
            logger.info("✅ Security configuration validation passed")

        # 5. Log warnings
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")

        # 6. Check if validation passed
        if errors:
            logger.error("Startup validation failed with errors:")
            for error in errors:
                logger.error(f"  - {error}")

            raise StartupValidationError(
                f"Startup validation failed for {service_name}",
                errors,
            )

        # 7. Log validation summary
        logger.info("🎉 Startup validation completed successfully")
        logger.info("Validation summary:")
        for check, passed in validation_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  - {check}: {status}")

        if warnings:
            logger.info(f"Total warnings: {len(warnings)}")

        return True

    except StartupValidationError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during startup validation: {e!s}")
        raise StartupValidationError(
            f"Unexpected validation error for {service_name}",
            [str(e)],
        )


def validate_external_dependencies() -> list[str]:
    """Validate external service dependencies."""
    errors = []

    try:
        config = get_config()

        # Database connectivity
        try:
            import asyncio

            import asyncpg

            async def check_db():
                try:
                    conn = await asyncpg.connect(config.database.get_database_url())
                    await conn.close()
                    return True
                except Exception as e:
                    return str(e)

            result = asyncio.run(check_db())
            if result is not True:
                if config.is_production():
                    errors.append(f"Database connection failed: {result}")
                else:
                    logger.warning(
                        f"Database connection failed (non-production): {result}",
                    )

        except Exception as e:
            errors.append(f"Database validation error: {e!s}")

        # Kafka connectivity (optional in development)
        try:
            # For now, just validate configuration format
            kafka_servers = config.kafka.bootstrap_servers.split(",")
            for server in kafka_servers:
                if ":" not in server.strip():
                    errors.append(f"Invalid Kafka server format: {server}")

        except Exception as e:
            errors.append(f"Kafka validation error: {e!s}")

    except Exception as e:
        errors.append(f"External dependencies validation error: {e!s}")

    return errors


def validate_security_configuration() -> list[str]:
    """Validate security-specific configuration."""
    errors = []

    try:
        config = get_config()

        # Production security requirements
        if config.is_production():
            # API key validation
            if (
                not config.security.api_key
                or config.security.api_key == "omninode-bridge-api-key-2024"
            ):
                errors.append("Secure API key required in production")

            if len(config.security.api_key or "") < 32:
                errors.append("API key must be at least 32 characters in production")

            # Database password validation
            if not config.database.password:
                errors.append("Database password required in production")

            # SSL validation
            if not config.database.ssl_enabled:
                errors.append("Database SSL must be enabled in production")

            # CORS validation
            for origin in config.security.cors_allowed_origins:
                if origin == "*":
                    errors.append("Wildcard CORS origins not allowed in production")
                if not origin.startswith("https://") and origin not in [
                    "http://localhost:3000",
                ]:
                    errors.append(f"Non-HTTPS CORS origin in production: {origin}")

        # Rate limiting validation
        if config.security.rate_limit_requests_per_minute < 1:
            errors.append("Rate limit must be at least 1 request per minute")

    except Exception as e:
        errors.append(f"Security validation error: {e!s}")

    return errors


def validate_service_specific_config(service_name: str) -> list[str]:
    """Validate service-specific configuration requirements."""
    errors = []

    try:
        config = get_config()

        if service_name == "hook-receiver":
            # Hook receiver specific validations
            if config.services.hook_receiver_port < 1024 and config.is_production():
                errors.append("Hook receiver port should be >= 1024 in production")

        elif service_name == "model-metrics":
            # Model metrics specific validations
            ai_lab_hosts = [
                config.services.ai_lab_mac_studio,
                config.services.ai_lab_mac_mini,
                config.services.ai_lab_ai_pc,
                config.services.ai_lab_macbook_air,
            ]

            if not any(ai_lab_hosts):
                errors.append("At least one AI lab host must be configured")

        elif service_name == "workflow-coordinator":
            # Workflow coordinator specific validations
            if config.cache.workflow_memory_mb < 50:
                errors.append("Workflow cache memory should be at least 50MB")

    except Exception as e:
        errors.append(f"Service-specific validation error: {e!s}")

    return errors


def log_startup_environment_info():
    """Log useful startup environment information."""
    try:
        config = get_config()

        logger.info("=== OmniNode Bridge Startup Information ===")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Service Version: {config.service_version}")
        logger.info(f"Service Instance: {config.service_instance_id}")
        logger.info(f"Log Level: {config.log_level}")
        logger.info(f"Database Host: {config.database.host}:{config.database.port}")
        logger.info(f"Database SSL: {config.database.ssl_enabled}")
        logger.info(f"Kafka Servers: {config.kafka.bootstrap_servers}")
        logger.info("Security Features: Rate limiting enabled")
        logger.info("==========================================")

    except Exception as e:
        logger.warning(f"Could not log startup environment info: {e!s}")


def emergency_shutdown_validation_failure(service_name: str, errors: list[str]):
    """Handle emergency shutdown when validation fails critically."""
    logger.critical(f"EMERGENCY SHUTDOWN: {service_name} failed startup validation")
    logger.critical("Critical errors:")
    for error in errors:
        logger.critical(f"  - {error}")

    logger.critical("Service cannot start safely. Exiting...")
    sys.exit(1)
