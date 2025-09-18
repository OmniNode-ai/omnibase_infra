#!/usr/bin/env python3
"""
Configuration Subcontract Model - ONEX Infrastructure Standards Compliant.

Dedicated subcontract model for configuration functionality providing:
- Configuration source priority and validation
- Environment variable loading with prefix patterns
- Container service resolution with fallback
- Configuration validation and sanitization
- Sensitive data detection and masking
- Error handling and logging

This model is composed into infrastructure node contracts that require
configuration functionality, providing clean separation between node
logic and configuration management behavior.

ZERO TOLERANCE: No Any types allowed in implementation.
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ConfigurationSourceType(str, Enum):
    """Configuration source types in priority order."""

    CONTAINER = "container"
    ENVIRONMENT = "environment"
    DEFAULTS = "defaults"
    FILE = "file"


class ValidationRuleType(str, Enum):
    """Configuration validation rule types."""

    FORMAT = "format"
    RANGE = "range"
    ENUM = "enum"
    REQUIRED = "required"


class ModelConfigurationSource(BaseModel):
    """
    Configuration source with priority and validation.

    Defines where configuration values are loaded from
    and in what order, with validation capabilities.
    """

    source_type: ConfigurationSourceType = Field(
        ...,
        description="Type of configuration source",
    )

    priority: int = Field(
        ...,
        description="Priority for configuration loading (1-100)",
        ge=1,
        le=100,
    )

    validation_enabled: bool = Field(
        default=True,
        description="Whether validation is enabled for this source",
    )


class ModelEnvironmentConfiguration(BaseModel):
    """
    Environment-based configuration loading.

    Manages environment variable loading with proper
    prefixing, validation, and fallback values.
    """

    prefix: str = Field(
        ...,
        description="Environment variable prefix pattern",
        min_length=1,
        max_length=64,
    )

    required_variables: list[str] = Field(
        default_factory=list,
        description="Required environment variables",
    )

    optional_variables: list[str] = Field(
        default_factory=list,
        description="Optional environment variables",
    )

    fallback_values: dict[str, str] = Field(
        default_factory=dict,
        description="Fallback values for missing variables",
    )

    @field_validator("prefix")
    @classmethod
    def validate_prefix(cls, v: str) -> str:
        """Validate environment prefix follows ONEX patterns."""
        if not v.endswith("_"):
            v = f"{v}_"
        if not v.isupper():
            v = v.upper()
        if (
            not v.replace("_", "")
            .replace("0", "")
            .replace("1", "")
            .replace("2", "")
            .replace("3", "")
            .replace("4", "")
            .replace("5", "")
            .replace("6", "")
            .replace("7", "")
            .replace("8", "")
            .replace("9", "")
            .isalpha()
        ):
            raise ValueError(
                "Environment prefix must contain only letters, numbers, and underscores",
            )
        return v


class ModelValidationRule(BaseModel):
    """
    Individual validation rule for configuration values.

    Defines specific validation logic for configuration
    fields including format, range, and enum constraints.
    """

    field_name: str = Field(
        ...,
        description="Name of the field to validate",
        min_length=1,
    )

    rule_type: ValidationRuleType = Field(
        ...,
        description="Type of validation rule to apply",
    )

    pattern: str | None = Field(
        default=None,
        description="Regex pattern for format validation",
    )

    range_min: float | None = Field(
        default=None,
        description="Minimum value for range validation",
    )

    range_max: float | None = Field(
        default=None,
        description="Maximum value for range validation",
    )

    allowed_values: list[str] | None = Field(
        default=None,
        description="Allowed values for enum validation",
    )

    error_message: str | None = Field(
        default=None,
        description="Custom error message for validation failure",
    )

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str | None, info) -> str | None:
        """Validate regex pattern when rule_type is FORMAT."""
        if info.data.get("rule_type") == ValidationRuleType.FORMAT and not v:
            raise ValueError("Pattern is required when rule_type is 'format'")
        return v

    @field_validator("range_min", "range_max")
    @classmethod
    def validate_range_values(cls, v: float | None, info) -> float | None:
        """Validate range values when rule_type is RANGE."""
        if info.data.get("rule_type") == ValidationRuleType.RANGE:
            if info.field_name == "range_min" and v is None:
                raise ValueError("range_min is required when rule_type is 'range'")
            if info.field_name == "range_max" and v is None:
                raise ValueError("range_max is required when rule_type is 'range'")
        return v

    @field_validator("allowed_values")
    @classmethod
    def validate_allowed_values(cls, v: list[str] | None, info) -> list[str] | None:
        """Validate allowed values when rule_type is ENUM."""
        if info.data.get("rule_type") == ValidationRuleType.ENUM and not v:
            raise ValueError("allowed_values is required when rule_type is 'enum'")
        return v


class ModelConfigurationValidation(BaseModel):
    """
    Configuration validation rules and patterns.

    Manages validation rules, sensitive field detection,
    and required field enforcement for configuration.
    """

    validation_rules: list[ModelValidationRule] = Field(
        ...,
        description="List of validation rules to apply",
    )

    sensitive_field_patterns: list[str] = Field(
        default_factory=lambda: ["password", "secret", "key", "token", "credential"],
        description="Patterns to identify sensitive fields",
    )

    required_fields: list[str] = Field(
        default_factory=list,
        description="List of required configuration fields",
    )


class ModelConfigurationIntegration(BaseModel):
    """
    Configuration integration patterns.

    Defines how configuration integrates with container
    services, environment loading, and caching systems.
    """

    container_service_resolution_enabled: bool = Field(
        default=True,
        description="Enable container service resolution",
    )

    container_service_key: str = Field(
        default="configuration_service",
        description="Service key for container resolution",
    )

    environment_loading_enabled: bool = Field(
        default=True,
        description="Enable environment variable loading",
    )

    prefix_required: bool = Field(
        default=True,
        description="Require environment variable prefix",
    )

    fallback_enabled: bool = Field(
        default=True,
        description="Enable fallback to defaults",
    )

    caching_enabled: bool = Field(
        default=True,
        description="Enable configuration caching",
    )

    cache_duration_seconds: int = Field(
        default=300,
        description="Cache duration in seconds",
        ge=1,
        le=3600,
    )


class ModelConfigurationSecurity(BaseModel):
    """
    Configuration security settings.

    Manages sensitive data detection, sanitization,
    and secure logging for configuration values.
    """

    sanitize_logs: bool = Field(
        default=True,
        description="Sanitize sensitive values in logs",
    )

    mask_sensitive_values: bool = Field(
        default=True,
        description="Mask sensitive configuration values",
    )

    sensitive_patterns: list[str] = Field(
        default_factory=lambda: ["password", "secret", "key", "token", "credential"],
        description="Patterns that identify sensitive fields",
    )

    redaction_replacement: str = Field(
        default="[REDACTED]",
        description="Replacement text for sensitive values",
    )


class ModelConfigurationSubcontract(BaseModel):
    """
    Main configuration subcontract model.

    Comprehensive configuration management system that provides
    standardized loading, validation, and security patterns
    for ONEX infrastructure nodes.
    """

    subcontract_version: str = Field(
        default="1.0.0",
        description="Configuration subcontract version",
    )

    sources: list[ModelConfigurationSource] = Field(
        default_factory=lambda: [
            ModelConfigurationSource(
                source_type=ConfigurationSourceType.CONTAINER, priority=1,
            ),
            ModelConfigurationSource(
                source_type=ConfigurationSourceType.ENVIRONMENT, priority=2,
            ),
            ModelConfigurationSource(
                source_type=ConfigurationSourceType.DEFAULTS, priority=3,
            ),
        ],
        description="Configuration sources in priority order",
    )

    environment_config: ModelEnvironmentConfiguration | None = Field(
        default=None,
        description="Environment variable configuration",
    )

    validation_config: ModelConfigurationValidation | None = Field(
        default=None,
        description="Configuration validation settings",
    )

    integration_config: ModelConfigurationIntegration = Field(
        default_factory=ModelConfigurationIntegration,
        description="Integration pattern configuration",
    )

    security_config: ModelConfigurationSecurity = Field(
        default_factory=ModelConfigurationSecurity,
        description="Security and sanitization configuration",
    )

    fail_on_missing_required: bool = Field(
        default=True,
        description="Fail when required configuration is missing",
    )

    fail_on_invalid_format: bool = Field(
        default=True,
        description="Fail when configuration format is invalid",
    )

    log_configuration_errors: bool = Field(
        default=True,
        description="Log configuration loading errors",
    )

    provide_detailed_validation_messages: bool = Field(
        default=True,
        description="Provide detailed validation error messages",
    )
