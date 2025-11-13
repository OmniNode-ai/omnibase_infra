"""Validation module for external inputs and data schemas.

This module provides comprehensive Pydantic validation schemas for all external data
entering the OmniNode Bridge system to ensure security, type safety, and data integrity.
"""

from .external_inputs import (
    CLIInputSchema,
    ConfigurationFileSchema,
    EnvironmentVariablesSchema,
    FileUploadSchema,
    WebhookPayloadSchema,
    validate_cli_input,
    validate_config_file,
    validate_environment_variables,
    validate_file_upload,
    validate_webhook_payload,
)

__all__ = [
    "EnvironmentVariablesSchema",
    "CLIInputSchema",
    "WebhookPayloadSchema",
    "FileUploadSchema",
    "ConfigurationFileSchema",
    "validate_environment_variables",
    "validate_cli_input",
    "validate_webhook_payload",
    "validate_file_upload",
    "validate_config_file",
]
