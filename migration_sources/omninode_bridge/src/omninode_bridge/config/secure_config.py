"""
Secure Configuration Management with HashiCorp Vault Integration

This module provides secure configuration management that integrates with HashiCorp Vault
for secrets while maintaining backward compatibility with environment variables.

Features:
- Seamless Vault integration with fallback to environment variables
- SSL/TLS enforcement for production environments
- Rate limiting configuration
- Secret rotation handling
- Configuration validation and compliance checks
"""

import logging
import os
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from ..services.vault_secrets_manager import VaultSecretsManager, get_vault_manager

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SecurityMode(Enum):
    """Security enforcement modes."""

    PERMISSIVE = "permissive"  # Allow plaintext secrets (development only)
    STRICT = "strict"  # Require encrypted secrets and SSL
    COMPLIANCE = "compliance"  # Full compliance mode with audit requirements


class DatabaseConfig(BaseModel):
    """Secure database configuration with SSL/TLS enforcement."""

    host: str = Field(..., min_length=1, description="Database host address")
    port: int = Field(..., ge=1, le=65535, description="Database port")
    database: str = Field(..., min_length=1, description="Database name")
    user: str = Field(..., min_length=1, description="Database user")
    password: str = Field(..., min_length=8, description="Database password")

    # SSL/TLS Configuration
    ssl_enabled: bool = Field(..., description="Enable SSL/TLS connection")
    ssl_cert_path: str | None = Field(None, description="SSL certificate file path")
    ssl_key_path: str | None = Field(None, description="SSL key file path")
    ssl_ca_path: str | None = Field(None, description="SSL CA file path")
    ssl_check_hostname: bool = Field(True, description="Verify SSL hostname")
    ssl_mode: str = Field(
        ...,
        pattern=r"^(require|verify-ca|verify-full|prefer|disable)$",
        description="SSL mode",
    )

    # Connection Pool Configuration
    pool_min_size: int = Field(..., ge=1, le=100, description="Minimum pool size")
    pool_max_size: int = Field(..., ge=1, le=200, description="Maximum pool size")
    max_queries_per_connection: int = Field(
        ..., ge=100, le=100000, description="Max queries per connection"
    )
    connection_max_age_seconds: int = Field(
        ..., ge=300, le=86400, description="Connection maximum age"
    )
    query_timeout_seconds: int = Field(..., ge=5, le=300, description="Query timeout")
    acquire_timeout_seconds: int = Field(
        ..., ge=1, le=60, description="Connection acquire timeout"
    )
    pool_exhaustion_threshold: float = Field(
        ..., ge=50.0, le=95.0, description="Pool exhaustion threshold percentage"
    )
    leak_detection: bool = Field(True, description="Enable connection leak detection")

    @field_validator("pool_max_size")
    @classmethod
    def validate_pool_sizes(cls, v: int, info) -> int:
        """Validate pool max size is greater than min size."""
        if hasattr(info.data, "pool_min_size") and v <= info.data["pool_min_size"]:
            raise ValueError("pool_max_size must be greater than pool_min_size")
        return v


class APISecurityConfig(BaseModel):
    """API security configuration with rate limiting."""

    api_key: str = Field(..., min_length=16, description="API key for authentication")
    jwt_secret: str = Field(..., min_length=32, description="JWT signing secret")

    # Rate Limiting
    rate_limit_enabled: bool = Field(True, description="Enable API rate limiting")
    rate_limit_requests_per_minute: int = Field(
        ..., ge=1, le=10000, description="Requests per minute limit"
    )
    rate_limit_burst_size: int = Field(
        ..., ge=1, le=1000, description="Burst size for rate limiting"
    )
    rate_limit_strategy: str = Field(
        ...,
        pattern=r"^(fixed-window|sliding-window|token-bucket)$",
        description="Rate limiting strategy",
    )

    # API Key Management
    api_key_rotation_enabled: bool = Field(False, description="Enable API key rotation")
    api_key_rotation_interval_days: int = Field(
        ..., ge=1, le=365, description="API key rotation interval"
    )

    # Authentication
    jwt_expiration_hours: int = Field(
        ..., ge=1, le=168, description="JWT expiration in hours"
    )
    require_https: bool = Field(True, description="Require HTTPS for API calls")
    cors_origins: list[str] = Field(
        default_factory=list, description="Allowed CORS origins"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key_strength(cls, v: str) -> str:
        """Validate API key strength."""
        weak_keys = ["omninode-bridge-api-key-2024", "default-api-key", "test-api-key"]
        if v in weak_keys:
            raise ValueError("Weak or default API key detected")
        return v

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, v: list[str]) -> list[str]:
        """Validate CORS origins."""
        for origin in v:
            if not origin.startswith(("http://", "https://", "*")):
                raise ValueError(f"Invalid CORS origin format: {origin}")
        return v


class IntegrationsConfig(BaseModel):
    """Third-party integrations configuration."""

    github_token: str = Field(..., min_length=20, description="GitHub API token")
    slack_webhook_url: str = Field(
        ..., pattern=r"^https://hooks\.slack\.com/.*", description="Slack webhook URL"
    )

    # Webhook Security
    webhook_secret: str = Field(
        ..., min_length=16, description="Webhook signing secret"
    )
    webhook_timeout_seconds: int = Field(
        ..., ge=5, le=120, description="Webhook timeout"
    )
    webhook_retry_attempts: int = Field(
        ..., ge=0, le=10, description="Webhook retry attempts"
    )

    # Rate Limiting for Integrations
    github_rate_limit_per_hour: int = Field(
        ..., ge=100, le=10000, description="GitHub API rate limit"
    )
    slack_rate_limit_per_minute: int = Field(
        ..., ge=1, le=60, description="Slack API rate limit"
    )

    @field_validator("webhook_secret")
    @classmethod
    def validate_webhook_secret(cls, v: str) -> str:
        """Validate webhook secret strength."""
        if "default-webhook-secret" in v:
            raise ValueError("Default webhook secret not allowed")
        return v


class SecurityPolicyConfig(BaseModel):
    """Security policy configuration."""

    security_mode: SecurityMode = Field(..., description="Security enforcement mode")
    environment: Environment = Field(..., description="Deployment environment")

    # SSL/TLS Requirements
    require_ssl_database: bool = Field(
        True, description="Require SSL for database connections"
    )
    require_ssl_api: bool = Field(True, description="Require SSL for API endpoints")
    require_ssl_webhooks: bool = Field(
        True, description="Require SSL for webhook endpoints"
    )

    # Secret Management
    vault_enabled: bool = Field(False, description="Enable HashiCorp Vault integration")
    vault_url: str = Field(..., pattern=r"^https?://.*", description="Vault server URL")
    vault_mount_point: str = Field(..., min_length=1, description="Vault mount point")
    secret_rotation_enabled: bool = Field(
        False, description="Enable automatic secret rotation"
    )

    # Compliance Requirements
    audit_logging_enabled: bool = Field(
        True, description="Enable security audit logging"
    )
    security_headers_enabled: bool = Field(True, description="Enable security headers")
    input_validation_strict: bool = Field(
        True, description="Enable strict input validation"
    )

    # Password Policy
    min_password_length: int = Field(
        ..., ge=8, le=128, description="Minimum password length"
    )
    password_complexity_required: bool = Field(
        True, description="Require complex passwords"
    )


class SecureConfig(BaseModel):
    """Complete secure configuration for OmniNode Bridge."""

    database: DatabaseConfig = Field(..., description="Database configuration")
    api_security: APISecurityConfig = Field(
        ..., description="API security configuration"
    )
    integrations: IntegrationsConfig = Field(
        ..., description="Third-party integrations"
    )
    security_policy: SecurityPolicyConfig = Field(
        ..., description="Security policy settings"
    )

    # Service Ports
    hook_receiver_port: int = Field(
        ..., ge=1024, le=65535, description="Hook receiver service port"
    )
    tool_proxy_port: int = Field(
        ..., ge=1024, le=65535, description="Tool proxy service port"
    )
    service_registry_port: int = Field(
        ..., ge=1024, le=65535, description="Service registry port"
    )
    intelligence_engine_port: int = Field(
        ..., ge=1024, le=65535, description="Intelligence engine port"
    )
    model_metrics_port: int = Field(
        ..., ge=1024, le=65535, description="Model metrics service port"
    )
    workflow_coordinator_port: int = Field(
        ..., ge=1024, le=65535, description="Workflow coordinator port"
    )

    # Infrastructure
    consul_port: int = Field(..., ge=1024, le=65535, description="Consul service port")
    redpanda_port: int = Field(
        ..., ge=1024, le=65535, description="RedPanda Kafka port"
    )
    redpanda_ui_port: int = Field(
        ..., ge=1024, le=65535, description="RedPanda UI port"
    )

    @field_validator(
        "hook_receiver_port",
        "tool_proxy_port",
        "service_registry_port",
        "intelligence_engine_port",
        "model_metrics_port",
        "workflow_coordinator_port",
        "consul_port",
        "redpanda_port",
        "redpanda_ui_port",
    )
    @classmethod
    def validate_port_uniqueness(cls, v: int, info) -> int:
        """Validate port uniqueness across all services."""
        # Note: Full port uniqueness validation would require all ports to be available
        # This is a basic validation - comprehensive check happens during startup
        if v in [22, 80, 443, 3306, 5432]:  # Reserved/common ports
            raise ValueError(f"Port {v} is reserved or commonly used by other services")
        return v


class SecureConfigManager:
    """
    Secure configuration manager with Vault integration.

    This class manages configuration by:
    1. First trying to get secrets from HashiCorp Vault
    2. Falling back to environment variables if Vault is unavailable
    3. Enforcing security policies based on environment
    4. Validating configuration against compliance requirements
    """

    def __init__(self):
        self.vault_manager: Optional[VaultSecretsManager] = None
        self.environment = Environment(os.getenv("ENVIRONMENT", "development").lower())
        self.security_mode = SecurityMode(
            os.getenv("SECURITY_MODE", "permissive").lower()
        )
        self._config: Optional[SecureConfig] = None

    async def initialize(self) -> bool:
        """Initialize the secure configuration manager."""
        try:
            # Initialize Vault integration if enabled
            vault_enabled = os.getenv("VAULT_ENABLED", "false").lower() == "true"

            if vault_enabled or self.environment == Environment.PRODUCTION:
                try:
                    self.vault_manager = await get_vault_manager()
                    logger.info("Vault secrets manager initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Vault: {e}")
                    if self.environment == Environment.PRODUCTION:
                        logger.error("Vault required in production environment")
                        return False

            # Load and validate configuration
            self._config = await self._load_configuration()

            # Validate security compliance
            if not self._validate_security_compliance():
                logger.error("Configuration failed security compliance validation")
                return False

            logger.info(
                f"Secure configuration loaded for {self.environment.value} environment"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize secure configuration: {e}")
            return False

    async def _load_configuration(self) -> SecureConfig:
        """Load configuration from Vault and environment variables."""

        # Database Configuration
        database_config = DatabaseConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5436")),
            database=os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=await self._get_secret("postgres_password", "POSTGRES_PASSWORD"),
            # SSL/TLS Configuration - enforce in production
            ssl_enabled=self._get_bool_env(
                "POSTGRES_SSL_ENABLED", self.environment == Environment.PRODUCTION
            ),
            ssl_cert_path=os.getenv("POSTGRES_SSL_CERT"),
            ssl_key_path=os.getenv("POSTGRES_SSL_KEY"),
            ssl_ca_path=os.getenv("POSTGRES_SSL_CA"),
            ssl_check_hostname=self._get_bool_env("POSTGRES_SSL_CHECK_HOSTNAME", True),
            ssl_mode=os.getenv(
                "POSTGRES_SSL_MODE",
                "require" if self.environment == Environment.PRODUCTION else "prefer",
            ),
            # Connection Pool Configuration
            pool_min_size=int(os.getenv("POSTGRES_POOL_MIN_SIZE", "2")),
            pool_max_size=int(os.getenv("POSTGRES_POOL_MAX_SIZE", "20")),
            max_queries_per_connection=int(
                os.getenv("POSTGRES_MAX_QUERIES_PER_CONNECTION", "50000")
            ),
            connection_max_age_seconds=int(
                os.getenv("POSTGRES_CONNECTION_MAX_AGE", "3600")
            ),
            query_timeout_seconds=int(os.getenv("POSTGRES_QUERY_TIMEOUT", "30")),
            acquire_timeout_seconds=int(os.getenv("POSTGRES_ACQUIRE_TIMEOUT", "10")),
            pool_exhaustion_threshold=float(
                os.getenv("POSTGRES_POOL_EXHAUSTION_THRESHOLD", "80.0")
            ),
            leak_detection=self._get_bool_env("POSTGRES_LEAK_DETECTION", True),
        )

        # API Security Configuration
        api_security_config = APISecurityConfig(
            api_key=await self._get_secret("api_key", "API_KEY"),
            jwt_secret=await self._get_secret("jwt_secret", "JWT_SECRET"),
            # Rate Limiting Configuration
            rate_limit_enabled=self._get_bool_env("RATE_LIMIT_ENABLED", True),
            rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "100")),
            rate_limit_burst_size=int(os.getenv("RATE_LIMIT_BURST", "20")),
            rate_limit_strategy=os.getenv("RATE_LIMIT_STRATEGY", "sliding-window"),
            # API Key Management
            api_key_rotation_enabled=self._get_bool_env(
                "API_KEY_ROTATION_ENABLED", self.environment == Environment.PRODUCTION
            ),
            api_key_rotation_interval_days=int(
                os.getenv("API_KEY_ROTATION_DAYS", "30")
            ),
            # Authentication
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
            require_https=self._get_bool_env(
                "REQUIRE_HTTPS", self.environment != Environment.DEVELOPMENT
            ),
            cors_origins=(
                os.getenv("CORS_ORIGINS", "").split(",")
                if os.getenv("CORS_ORIGINS")
                else []
            ),
        )

        # Integrations Configuration
        # Prefer GH_PAT if provided, fallback to GITHUB_TOKEN
        integrations_config = IntegrationsConfig(
            github_token=(
                await self._get_secret("github_token", "GH_PAT")
                or os.getenv("GITHUB_TOKEN", "")
            ),
            slack_webhook_url=await self._get_secret(
                "slack_webhook_url", "SLACK_WEBHOOK_URL"
            ),
            # Webhook Security
            webhook_secret=await self._get_secret(
                "webhook_secret",
                "WEBHOOK_SIGNING_SECRET",
                default="default-webhook-secret-change-in-production",
            ),
            webhook_timeout_seconds=int(os.getenv("WEBHOOK_TIMEOUT", "30")),
            webhook_retry_attempts=int(os.getenv("WEBHOOK_RETRY_ATTEMPTS", "3")),
            # Rate Limiting for Integrations
            github_rate_limit_per_hour=int(os.getenv("GITHUB_RATE_LIMIT_HOUR", "5000")),
            slack_rate_limit_per_minute=int(os.getenv("SLACK_RATE_LIMIT_MINUTE", "1")),
        )

        # Security Policy Configuration
        security_policy_config = SecurityPolicyConfig(
            security_mode=self.security_mode,
            environment=self.environment,
            # SSL/TLS Requirements
            require_ssl_database=self.environment != Environment.DEVELOPMENT,
            require_ssl_api=self.environment == Environment.PRODUCTION,
            require_ssl_webhooks=self.environment != Environment.DEVELOPMENT,
            # Secret Management
            vault_enabled=self.vault_manager is not None,
            vault_url=os.getenv("VAULT_ADDR", "http://localhost:8200"),
            vault_mount_point=os.getenv("VAULT_MOUNT_POINT", "secret"),
            secret_rotation_enabled=self.environment == Environment.PRODUCTION,
            # Compliance Requirements
            audit_logging_enabled=self.environment != Environment.DEVELOPMENT,
            security_headers_enabled=True,
            input_validation_strict=self.security_mode != SecurityMode.PERMISSIVE,
            # Password Policy
            min_password_length=(
                16 if self.environment == Environment.PRODUCTION else 12
            ),
            password_complexity_required=self.environment != Environment.DEVELOPMENT,
        )

        # Complete Configuration
        return SecureConfig(
            database=database_config,
            api_security=api_security_config,
            integrations=integrations_config,
            security_policy=security_policy_config,
            # Service Ports
            hook_receiver_port=int(os.getenv("HOOK_RECEIVER_PORT", "8001")),
            tool_proxy_port=int(os.getenv("TOOL_PROXY_PORT", "8002")),
            service_registry_port=int(os.getenv("SERVICE_REGISTRY_PORT", "8003")),
            intelligence_engine_port=int(os.getenv("INTELLIGENCE_ENGINE_PORT", "8004")),
            model_metrics_port=int(os.getenv("MODEL_METRICS_PORT", "8005")),
            workflow_coordinator_port=int(
                os.getenv("WORKFLOW_COORDINATOR_PORT", "8006")
            ),
            # Infrastructure
            consul_port=int(os.getenv("CONSUL_PORT", "28500")),
            redpanda_port=int(os.getenv("REDPANDA_PORT", "29092")),
            redpanda_ui_port=int(os.getenv("REDPANDA_UI_PORT", "3839")),
        )

    async def _get_secret(
        self, vault_key: str, env_key: str, default: Optional[str] = None
    ) -> str:
        """Get secret from Vault or fallback to environment variable."""
        # Try Vault first if available
        if self.vault_manager:
            try:
                secret_value = await self.vault_manager.get_secret(vault_key)
                if secret_value:
                    return secret_value
                logger.debug(
                    f"Secret {vault_key} not found in Vault, falling back to environment"
                )
            except Exception as e:
                logger.warning(f"Failed to get secret {vault_key} from Vault: {e}")

        # Fallback to environment variable
        env_value = os.getenv(env_key, default)

        if not env_value and self.security_mode == SecurityMode.STRICT:
            raise ValueError(
                f"Secret {vault_key}/{env_key} is required in strict security mode"
            )

        return env_value or ""

    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    def _validate_security_compliance(self) -> bool:
        """Validate configuration against security compliance requirements."""
        if not self._config:
            return False

        validation_errors = []

        # Production environment validations
        if self.environment == Environment.PRODUCTION:
            # Database SSL/TLS required in production
            if not self._config.database.ssl_enabled:
                validation_errors.append(
                    "SSL/TLS must be enabled for database in production"
                )

            # API HTTPS required in production
            if not self._config.api_security.require_https:
                validation_errors.append("HTTPS must be required for API in production")

            # Strong passwords required in production
            if (
                len(self._config.database.password)
                < self._config.security_policy.min_password_length
            ):
                validation_errors.append(
                    f"Database password must be at least {self._config.security_policy.min_password_length} characters in production"
                )

            # API key rotation should be enabled in production
            if not self._config.api_security.api_key_rotation_enabled:
                validation_errors.append(
                    "API key rotation should be enabled in production"
                )

            # Default webhook secrets not allowed in production
            if "default-webhook-secret" in self._config.integrations.webhook_secret:
                validation_errors.append(
                    "Default webhook secret not allowed in production"
                )

        # Strict security mode validations
        if self.security_mode == SecurityMode.STRICT:
            # All secrets must be non-empty in strict mode
            required_secrets = [
                ("database.password", self._config.database.password),
                ("api_security.api_key", self._config.api_security.api_key),
                ("api_security.jwt_secret", self._config.api_security.jwt_secret),
            ]

            for name, value in required_secrets:
                if not value or len(value) < 8:
                    validation_errors.append(
                        f"Secret {name} is required and must be at least 8 characters in strict mode"
                    )

        # Rate limiting validations
        if self._config.api_security.rate_limit_enabled:
            if self._config.api_security.rate_limit_requests_per_minute < 1:
                validation_errors.append(
                    "Rate limit requests per minute must be at least 1"
                )

            if self._config.api_security.rate_limit_burst_size < 1:
                validation_errors.append("Rate limit burst size must be at least 1")

        # Log validation errors
        if validation_errors:
            for error in validation_errors:
                logger.error(f"Security compliance validation failed: {error}")
            return False

        logger.info("Security compliance validation passed")
        return True

    def get_config(self) -> SecureConfig:
        """Get the current secure configuration."""
        if not self._config:
            raise RuntimeError(
                "Configuration not initialized. Call initialize() first."
            )
        return self._config

    async def refresh_secrets(self) -> bool:
        """Refresh secrets from Vault."""
        if not self.vault_manager:
            logger.warning("Cannot refresh secrets - Vault manager not available")
            return False

        try:
            # Reload configuration to get fresh secrets
            self._config = await self._load_configuration()

            # Re-validate security compliance
            if not self._validate_security_compliance():
                logger.error("Configuration failed security compliance after refresh")
                return False

            logger.info("Secrets refreshed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to refresh secrets: {e}")
            return False

    async def validate_ssl_configuration(self) -> dict[str, Any]:
        """Validate SSL/TLS configuration and certificates."""
        config = self.get_config()
        validation_results = {}

        # Database SSL validation
        db_ssl_valid = True
        db_ssl_issues = []

        if config.database.ssl_enabled:
            if config.database.ssl_cert_path and not os.path.exists(
                config.database.ssl_cert_path
            ):
                db_ssl_issues.append(
                    f"SSL certificate file not found: {config.database.ssl_cert_path}"
                )
                db_ssl_valid = False

            if config.database.ssl_key_path and not os.path.exists(
                config.database.ssl_key_path
            ):
                db_ssl_issues.append(
                    f"SSL key file not found: {config.database.ssl_key_path}"
                )
                db_ssl_valid = False

            if config.database.ssl_ca_path and not os.path.exists(
                config.database.ssl_ca_path
            ):
                db_ssl_issues.append(
                    f"SSL CA file not found: {config.database.ssl_ca_path}"
                )
                db_ssl_valid = False

        validation_results["database_ssl"] = {
            "valid": db_ssl_valid,
            "enabled": config.database.ssl_enabled,
            "issues": db_ssl_issues,
            "ssl_mode": config.database.ssl_mode,
        }

        # API SSL validation
        api_ssl_valid = True
        api_ssl_issues = []

        if (
            config.security_policy.require_ssl_api
            and not config.api_security.require_https
        ):
            api_ssl_issues.append(
                "HTTPS should be required for API in current environment"
            )
            api_ssl_valid = False

        validation_results["api_ssl"] = {
            "valid": api_ssl_valid,
            "https_required": config.api_security.require_https,
            "issues": api_ssl_issues,
        }

        return validation_results

    async def get_security_status(self) -> dict[str, Any]:
        """Get comprehensive security status."""
        config = self.get_config()

        # SSL validation
        ssl_validation = await self.validate_ssl_configuration()

        # Vault status
        vault_status = {}
        if self.vault_manager:
            vault_health = await self.vault_manager.health_check()
            vault_status = {
                "enabled": True,
                "healthy": vault_health.get("status") == "healthy",
                "connected": vault_health.get("vault_connected", False),
                "authenticated": vault_health.get("authenticated", False),
                "secrets_configured": vault_health.get("secrets_configured", 0),
            }
        else:
            vault_status = {
                "enabled": False,
                "healthy": False,
                "connected": False,
                "authenticated": False,
                "secrets_configured": 0,
            }

        return {
            "environment": self.environment.value,
            "security_mode": self.security_mode.value,
            "ssl_validation": ssl_validation,
            "vault_status": vault_status,
            "rate_limiting": {
                "enabled": config.api_security.rate_limit_enabled,
                "requests_per_minute": config.api_security.rate_limit_requests_per_minute,
                "burst_size": config.api_security.rate_limit_burst_size,
                "strategy": config.api_security.rate_limit_strategy,
            },
            "compliance": {
                "audit_logging": config.security_policy.audit_logging_enabled,
                "security_headers": config.security_policy.security_headers_enabled,
                "input_validation_strict": config.security_policy.input_validation_strict,
                "secret_rotation": config.security_policy.secret_rotation_enabled,
            },
        }


# Global secure configuration manager instance
_secure_config_manager: Optional[SecureConfigManager] = None


async def get_secure_config_manager() -> SecureConfigManager:
    """Get or create the global secure configuration manager."""
    global _secure_config_manager

    if _secure_config_manager is None:
        _secure_config_manager = SecureConfigManager()
        await _secure_config_manager.initialize()

    return _secure_config_manager


async def get_config() -> SecureConfig:
    """Get the current secure configuration."""
    manager = await get_secure_config_manager()
    return manager.get_config()
