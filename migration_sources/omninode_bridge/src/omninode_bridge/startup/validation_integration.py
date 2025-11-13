"""Integration module for startup validation using Pydantic schemas.

This module demonstrates how to integrate the new Pydantic validation schemas
into the application startup process to ensure all external inputs are validated.
"""

import logging
import os

from pydantic import ValidationError

from ..validation.external_inputs import (
    EnvironmentVariablesSchema,
    validate_environment_variables,
)

logger = logging.getLogger(__name__)


def validate_startup_environment() -> (
    tuple[bool, EnvironmentVariablesSchema | None, list[str]]
):
    """
    Validate all environment variables at startup using Pydantic schemas.

    Returns:
        tuple: (is_valid, validated_env_schema, validation_errors)
    """
    validation_errors = []

    try:
        # Gather all environment variables
        env_vars = dict(os.environ)

        # Map environment variables to schema field names (lowercase)
        env_mapping = {}
        for key, value in env_vars.items():
            env_mapping[key.lower()] = value

        # Validate using Pydantic schema
        validated_env = validate_environment_variables(env_mapping)

        logger.info("✅ Environment variable validation passed")
        logger.info(f"Environment: {validated_env.environment}")
        logger.info(f"Service version: {validated_env.service_version}")
        logger.info(
            f"Database: {validated_env.postgres_host}:{validated_env.postgres_port}"
        )

        return True, validated_env, []

    except ValidationError as e:
        # Extract detailed validation errors
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            value = error.get("input", "N/A")

            error_msg = f"Field '{field}': {message} (value: {value})"
            validation_errors.append(error_msg)
            logger.error(f"❌ Environment validation error: {error_msg}")

        return False, None, validation_errors

    except Exception as e:
        error_msg = f"Unexpected error during environment validation: {e}"
        validation_errors.append(error_msg)
        logger.error(f"❌ {error_msg}")
        return False, None, validation_errors


def validate_required_environment_variables() -> tuple[bool, list[str]]:
    """
    Check that all required environment variables are present.

    Returns:
        tuple: (all_present, missing_variables)
    """
    # Define required variables by environment
    base_required = [
        "ENVIRONMENT",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DATABASE",
        "POSTGRES_USER",
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_WORKFLOW_TOPIC",
        "KAFKA_TASK_EVENTS_TOPIC",
    ]

    environment = os.getenv("ENVIRONMENT", "development").lower()

    # Add environment-specific required variables
    required_vars = base_required.copy()

    if environment == "production":
        required_vars.extend(
            ["POSTGRES_PASSWORD", "API_KEY", "JWT_SECRET", "WEBHOOK_SIGNING_SECRET"]
        )

    if environment in ["staging", "production"]:
        required_vars.extend(["SERVICE_VERSION", "LOG_LEVEL"])

    # Check for missing variables
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        return False, missing_vars
    else:
        logger.info(
            f"✅ All required environment variables present ({len(required_vars)} checked)"
        )
        return True, []


def log_environment_summary(validated_env: EnvironmentVariablesSchema):
    """Log a summary of the validated environment configuration."""
    logger.info("🔧 Environment Configuration Summary:")
    logger.info(f"  Environment: {validated_env.environment}")
    logger.info(f"  Service Version: {validated_env.service_version}")
    logger.info(f"  Log Level: {validated_env.log_level}")

    logger.info("🗄️ Database Configuration:")
    logger.info(f"  Host: {validated_env.postgres_host}")
    logger.info(f"  Port: {validated_env.postgres_port}")
    logger.info(f"  Database: {validated_env.postgres_database}")
    logger.info(f"  User: {validated_env.postgres_user}")
    logger.info(
        f"  Password: {'***SET***' if validated_env.postgres_password else 'NOT SET'}"
    )

    logger.info("📨 Kafka Configuration:")
    logger.info(f"  Bootstrap Servers: {validated_env.kafka_bootstrap_servers}")
    logger.info(f"  Workflow Topic: {validated_env.kafka_workflow_topic}")
    logger.info(f"  Task Events Topic: {validated_env.kafka_task_events_topic}")

    logger.info("🔐 Security Configuration:")
    logger.info(f"  API Key: {'***SET***' if validated_env.api_key else 'NOT SET'}")
    logger.info(
        f"  JWT Secret: {'***SET***' if validated_env.jwt_secret else 'NOT SET'}"
    )
    logger.info(
        f"  Webhook Secret: {'***SET***' if validated_env.webhook_signing_secret else 'NOT SET'}"
    )

    if validated_env.vault_enabled:
        logger.info("🏦 Vault Configuration:")
        logger.info(f"  Enabled: {validated_env.vault_enabled}")
        logger.info(f"  Address: {validated_env.vault_addr}")
        logger.info(
            f"  Token: {'***SET***' if validated_env.vault_token else 'NOT SET'}"
        )


def startup_validation_check() -> bool:
    """
    Comprehensive startup validation check.

    Returns:
        bool: True if all validations pass, False otherwise
    """
    logger.info("🚀 Starting comprehensive environment validation...")

    # Step 1: Check required variables are present
    required_check, missing_vars = validate_required_environment_variables()
    if not required_check:
        logger.error(
            f"❌ Startup validation failed: Missing required variables: {missing_vars}"
        )
        return False

    # Step 2: Validate environment variables with Pydantic schemas
    env_valid, validated_env, validation_errors = validate_startup_environment()
    if not env_valid:
        logger.error(
            "❌ Startup validation failed: Environment variable validation errors:"
        )
        for error in validation_errors:
            logger.error(f"  - {error}")
        return False

    # Step 3: Log configuration summary
    if validated_env:
        log_environment_summary(validated_env)

    # Step 4: Environment-specific validations
    environment = (
        validated_env.environment
        if validated_env
        else os.getenv("ENVIRONMENT", "development")
    )

    if environment == "production":
        # Additional production validations
        if (
            validated_env
            and validated_env.postgres_password
            and len(validated_env.postgres_password) < 12
        ):
            logger.error(
                "❌ Production validation failed: Database password too short (minimum 12 characters)"
            )
            return False

        if (
            validated_env
            and validated_env.api_key
            and "default" in validated_env.api_key.lower()
        ):
            logger.error(
                "❌ Production validation failed: Default API key detected in production"
            )
            return False

    logger.info("✅ All startup validation checks passed!")
    return True


# Example usage functions for different external input types
def example_webhook_validation():
    """Example of validating webhook payload."""
    from ..validation.external_inputs import validate_webhook_payload

    # Example GitHub webhook payload
    webhook_payload = {
        "event": "push",
        "action": "created",
        "timestamp": "2024-01-15T10:30:00Z",
        "signature": "sha256=abcdef123456",
        "repository": {"name": "omninode_bridge", "full_name": "org/omninode_bridge"},
        "sender": {"login": "developer"},
    }

    try:
        validated = validate_webhook_payload(webhook_payload)
        logger.info(f"✅ Webhook payload validation passed: {validated.event}")
        return True
    except ValidationError as e:
        logger.error(f"❌ Webhook validation failed: {e}")
        return False


def example_cli_validation():
    """Example of validating CLI input."""
    from ..validation.external_inputs import validate_cli_input

    # Example CLI arguments
    cli_args = {
        "workflow_file_path": "./workflows/example.json",
        "workflow_name": "example_workflow",
        "priority": 5,
        "service_name": "hook-receiver",
        "action": "start",
    }

    try:
        validated = validate_cli_input(cli_args)
        logger.info(f"✅ CLI input validation passed: {validated.workflow_name}")
        return True
    except ValidationError as e:
        logger.error(f"❌ CLI validation failed: {e}")
        return False
