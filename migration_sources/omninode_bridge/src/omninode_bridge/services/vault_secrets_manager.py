"""
HashiCorp Vault Secrets Management Service for OmniNode Bridge

This module provides comprehensive secrets management using HashiCorp Vault,
including automatic secret rotation, lifecycle management, and secure retrieval.

Security Features:
- HashiCorp Vault integration with multiple auth methods
- Automatic secret rotation with configurable intervals
- Audit logging for all secret operations
- Circuit breaker pattern for Vault unavailability
- Encrypted secret caching with TTL
- Production-ready error handling and retry logic
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import aiohttp
import hvac
from cryptography.fernet import Fernet

from ..utils.circuit_breaker_config import VAULT_CIRCUIT_BREAKER

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Supported Vault authentication methods."""

    TOKEN = "token"
    KUBERNETES = "kubernetes"
    AWS_IAM = "aws-iam"
    APPROLE = "approle"
    USERPASS = "userpass"


class SecretType(Enum):
    """Types of secrets managed by the system."""

    DATABASE_PASSWORD = "database_password"
    API_KEY = "api_key"
    WEBHOOK_SECRET = "webhook_secret"
    GITHUB_TOKEN = "github_token"
    SLACK_WEBHOOK_URL = "slack_webhook_url"
    JWT_SECRET = "jwt_secret"
    ENCRYPTION_KEY = "encryption_key"


@dataclass
class SecretConfig:
    """Configuration for a managed secret."""

    name: str
    secret_type: SecretType
    vault_path: str
    rotation_interval_hours: int = 24 * 7  # Default: weekly rotation
    required: bool = True
    validation_regex: Optional[str] = None
    min_length: int = 12
    dependencies: list[str] = field(
        default_factory=list
    )  # Services that need restart after rotation


@dataclass
class SecretMetadata:
    """Metadata about a secret's lifecycle."""

    created_at: datetime
    last_rotated: datetime
    next_rotation: datetime
    version: int
    rotation_count: int = 0
    last_access: Optional[datetime] = None


class VaultSecretsManager:
    """
    Production-ready HashiCorp Vault secrets manager with comprehensive features.

    Features:
    - Multi-authentication method support
    - Automatic secret rotation with configurable policies
    - Encrypted local caching with TTL
    - Circuit breaker for Vault unavailability
    - Comprehensive audit logging
    - Service dependency management for rotations
    """

    def __init__(
        self,
        vault_url: str = None,
        vault_token: str = None,
        auth_method: AuthMethod = AuthMethod.TOKEN,
        mount_point: str = "secret",
        cache_ttl_seconds: int = 300,  # 5 minutes cache
        cache_encryption_key: str = None,
    ):
        """
        Initialize the Vault secrets manager.

        Args:
            vault_url: Vault server URL (default from VAULT_ADDR env)
            vault_token: Vault token for authentication (default from VAULT_TOKEN env)
            auth_method: Authentication method to use
            mount_point: Vault mount point for secrets
            cache_ttl_seconds: TTL for encrypted secret cache
            cache_encryption_key: Key for encrypting cached secrets
        """
        # Configuration
        self.vault_url = vault_url or os.getenv("VAULT_ADDR", "http://localhost:8200")
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self.auth_method = auth_method
        self.mount_point = mount_point
        self.cache_ttl = cache_ttl_seconds

        # Initialize Vault client
        self.vault_client = hvac.Client(url=self.vault_url, token=self.vault_token)

        # Encrypted caching
        self.cache_key = cache_encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.cache_key)
        self._secret_cache: dict[str, dict[str, Any]] = {}

        # Secret configurations
        self.secret_configs: dict[str, SecretConfig] = {}
        self.secret_metadata: dict[str, SecretMetadata] = {}

        # State tracking
        self._initialized = False
        self._rotation_tasks: dict[str, asyncio.Task] = {}
        self._audit_log: list[dict[str, Any]] = []

        # Performance metrics
        self.metrics = {
            "secrets_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "vault_errors": 0,
            "rotations_completed": 0,
            "rotations_failed": 0,
        }

        # Load default secret configurations
        self._load_default_secret_configs()

    def _load_default_secret_configs(self):
        """Load default secret configurations for OmniNode Bridge services."""
        default_configs = [
            SecretConfig(
                name="postgres_password",
                secret_type=SecretType.DATABASE_PASSWORD,
                vault_path="database/postgres/password",
                rotation_interval_hours=24 * 7,  # Weekly
                min_length=16,
                dependencies=[
                    "postgres_client",
                    "hook_receiver",
                    "workflow_coordinator",
                ],
            ),
            SecretConfig(
                name="api_key",
                secret_type=SecretType.API_KEY,
                vault_path="api/bridge/key",
                rotation_interval_hours=24 * 30,  # Monthly
                min_length=32,
                dependencies=["model_metrics_api", "workflow_coordinator"],
            ),
            SecretConfig(
                name="github_token",
                secret_type=SecretType.GITHUB_TOKEN,
                vault_path="integrations/github/token",
                rotation_interval_hours=24 * 60,  # Every 60 days
                min_length=40,
                validation_regex=r"^(ghp_|gho_|ghu_|ghs_|ghr_)[a-zA-Z0-9]{36}$",
            ),
            SecretConfig(
                name="slack_webhook_url",
                secret_type=SecretType.SLACK_WEBHOOK_URL,
                vault_path="integrations/slack/webhook",
                rotation_interval_hours=24 * 90,  # Every 90 days
                validation_regex=r"^https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[a-zA-Z0-9]+$",
            ),
            SecretConfig(
                name="jwt_secret",
                secret_type=SecretType.JWT_SECRET,
                vault_path="auth/jwt/secret",
                rotation_interval_hours=24 * 7,  # Weekly
                min_length=32,
                dependencies=["auth_service"],
            ),
        ]

        for config in default_configs:
            self.secret_configs[config.name] = config

    async def initialize(self) -> bool:
        """
        Initialize the secrets manager and authenticate with Vault.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Test Vault connectivity
            if not await self._test_vault_connection():
                logger.error("Failed to connect to Vault server")
                return False

            # Authenticate with Vault
            if not await self._authenticate():
                logger.error("Failed to authenticate with Vault")
                return False

            # Load existing secret metadata
            await self._load_secret_metadata()

            # Start rotation scheduling
            await self._start_rotation_scheduler()

            self._initialized = True
            logger.info("Vault secrets manager initialized successfully")

            await self._audit_log_event(
                "INITIALIZATION", "Secrets manager initialized", success=True
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Vault secrets manager: {e}")
            await self._audit_log_event(
                "INITIALIZATION", f"Initialization failed: {e}", success=False
            )
            return False

    @VAULT_CIRCUIT_BREAKER()
    async def _test_vault_connection(self) -> bool:
        """Test connection to Vault server."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.vault_url}/v1/sys/health", timeout=5
                ) as response:
                    return response.status in [
                        200,
                        429,
                        472,
                        473,
                    ]  # Vault health check status codes
        except Exception as e:
            logger.error(f"Vault connection test failed: {e}")
            return False

    async def _authenticate(self) -> bool:
        """Authenticate with Vault using configured auth method."""
        try:
            if self.auth_method == AuthMethod.TOKEN:
                if not self.vault_token:
                    logger.error("Vault token not provided")
                    return False
                self.vault_client.token = self.vault_token

            elif self.auth_method == AuthMethod.KUBERNETES:
                # Kubernetes authentication
                jwt_token = await self._get_kubernetes_jwt()
                auth_response = self.vault_client.auth.kubernetes.login(
                    role=os.getenv("VAULT_K8S_ROLE", "omninode-bridge"), jwt=jwt_token
                )
                self.vault_client.token = auth_response["auth"]["client_token"]

            elif self.auth_method == AuthMethod.APPROLE:
                # AppRole authentication
                role_id = os.getenv("VAULT_ROLE_ID")
                secret_id = os.getenv("VAULT_SECRET_ID")
                if not role_id or not secret_id:
                    logger.error("AppRole credentials not provided")
                    return False

                auth_response = self.vault_client.auth.approle.login(
                    role_id=role_id, secret_id=secret_id
                )
                self.vault_client.token = auth_response["auth"]["client_token"]

            else:
                logger.error(f"Unsupported auth method: {self.auth_method}")
                return False

            # Verify authentication worked
            if not self.vault_client.is_authenticated():
                logger.error("Vault authentication failed")
                return False

            logger.info(
                f"Successfully authenticated with Vault using {self.auth_method.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Vault authentication failed: {e}")
            return False

    async def _get_kubernetes_jwt(self) -> str:
        """Get Kubernetes JWT token for authentication."""
        token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        try:
            with open(token_path) as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read Kubernetes JWT token: {e}")
            raise

    @VAULT_CIRCUIT_BREAKER()
    async def get_secret(
        self, secret_name: str, force_refresh: bool = False
    ) -> Optional[str]:
        """
        Retrieve a secret with caching and audit logging.

        Args:
            secret_name: Name of the secret to retrieve
            force_refresh: Force refresh from Vault, bypassing cache

        Returns:
            Secret value or None if not found/error
        """
        if not self._initialized:
            logger.error("Secrets manager not initialized")
            return None

        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_value = self._get_from_cache(secret_name)
            if cached_value is not None:
                self.metrics["cache_hits"] += 1
                await self._audit_log_event(
                    "SECRET_ACCESS", f"Retrieved {secret_name} from cache"
                )
                return cached_value

        self.metrics["cache_misses"] += 1

        # Get secret configuration
        config = self.secret_configs.get(secret_name)
        if not config:
            logger.error(f"No configuration found for secret: {secret_name}")
            return None

        try:
            # Retrieve from Vault
            response = self.vault_client.secrets.kv.v2.read_secret_version(
                path=config.vault_path, mount_point=self.mount_point
            )

            secret_data = response["data"]["data"]
            secret_value = secret_data.get("value")

            if secret_value is None:
                logger.error(f"Secret value not found in Vault: {secret_name}")
                return None

            # Validate secret if validation rules exist
            if not self._validate_secret(secret_value, config):
                logger.error(f"Secret validation failed for: {secret_name}")
                return None

            # Cache the secret
            self._store_in_cache(secret_name, secret_value)

            # Update metadata
            await self._update_secret_access_metadata(secret_name)

            self.metrics["secrets_retrieved"] += 1
            await self._audit_log_event(
                "SECRET_ACCESS", f"Retrieved {secret_name} from Vault"
            )

            return secret_value

        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            self.metrics["vault_errors"] += 1
            await self._audit_log_event(
                "SECRET_ACCESS", f"Failed to retrieve {secret_name}: {e}", success=False
            )
            return None

    def _validate_secret(self, secret_value: str, config: SecretConfig) -> bool:
        """Validate secret against configuration rules."""
        # Check minimum length
        if len(secret_value) < config.min_length:
            logger.error(
                f"Secret {config.name} below minimum length {config.min_length}"
            )
            return False

        # Check regex pattern if provided
        if config.validation_regex:
            import re

            if not re.match(config.validation_regex, secret_value):
                logger.error(f"Secret {config.name} failed regex validation")
                return False

        return True

    def _get_from_cache(self, secret_name: str) -> Optional[str]:
        """Retrieve secret from encrypted cache if not expired."""
        cache_entry = self._secret_cache.get(secret_name)
        if not cache_entry:
            return None

        # Check if expired
        if time.time() > cache_entry["expires_at"]:
            del self._secret_cache[secret_name]
            return None

        # Decrypt and return
        try:
            encrypted_value = cache_entry["encrypted_value"]
            return self.cipher_suite.decrypt(encrypted_value).decode()
        except Exception as e:
            logger.warning(f"Failed to decrypt cached secret {secret_name}: {e}")
            # Remove invalid cache entry
            if secret_name in self._secret_cache:
                del self._secret_cache[secret_name]
            return None

    def _store_in_cache(self, secret_name: str, secret_value: str) -> None:
        """Store secret in encrypted cache with TTL."""
        try:
            encrypted_value = self.cipher_suite.encrypt(secret_value.encode())
            self._secret_cache[secret_name] = {
                "encrypted_value": encrypted_value,
                "expires_at": time.time() + self.cache_ttl,
                "cached_at": time.time(),
            }
        except Exception as e:
            logger.warning(f"Failed to cache secret {secret_name}: {e}")

    @VAULT_CIRCUIT_BREAKER()
    async def rotate_secret(self, secret_name: str) -> bool:
        """
        Rotate a secret and update all dependent services.

        Args:
            secret_name: Name of the secret to rotate

        Returns:
            True if rotation successful, False otherwise
        """
        if not self._initialized:
            logger.error("Secrets manager not initialized")
            return False

        config = self.secret_configs.get(secret_name)
        if not config:
            logger.error(f"No configuration found for secret: {secret_name}")
            return False

        try:
            logger.info(f"Starting rotation for secret: {secret_name}")

            # Generate new secret value
            new_value = self._generate_secret_value(config)

            # Store new secret in Vault
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=config.vault_path,
                secret={"value": new_value},
                mount_point=self.mount_point,
            )

            # Update cache
            self._store_in_cache(secret_name, new_value)

            # Update metadata
            now = datetime.now()
            if secret_name not in self.secret_metadata:
                self.secret_metadata[secret_name] = SecretMetadata(
                    created_at=now,
                    last_rotated=now,
                    next_rotation=now + timedelta(hours=config.rotation_interval_hours),
                    version=1,
                )
            else:
                metadata = self.secret_metadata[secret_name]
                metadata.last_rotated = now
                metadata.next_rotation = now + timedelta(
                    hours=config.rotation_interval_hours
                )
                metadata.version += 1
                metadata.rotation_count += 1

            # Notify dependent services
            await self._notify_services_of_rotation(
                config.dependencies, secret_name, new_value
            )

            self.metrics["rotations_completed"] += 1
            await self._audit_log_event(
                "SECRET_ROTATION", f"Rotated secret {secret_name}"
            )

            logger.info(f"Successfully rotated secret: {secret_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to rotate secret {secret_name}: {e}")
            self.metrics["rotations_failed"] += 1
            await self._audit_log_event(
                "SECRET_ROTATION", f"Failed to rotate {secret_name}: {e}", success=False
            )
            return False

    def _generate_secret_value(self, config: SecretConfig) -> str:
        """Generate a new secret value based on secret type."""
        import secrets
        import string

        if config.secret_type == SecretType.DATABASE_PASSWORD:
            # Generate strong database password
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            return "".join(
                secrets.choice(alphabet) for _ in range(max(config.min_length, 16))
            )

        elif config.secret_type == SecretType.API_KEY:
            # Generate API key with prefix
            alphabet = string.ascii_letters + string.digits
            key_part = "".join(
                secrets.choice(alphabet) for _ in range(max(config.min_length - 14, 20))
            )
            return f"omnibridge_{key_part}"

        elif config.secret_type == SecretType.JWT_SECRET:
            # Generate JWT secret (base64 encoded)
            import base64

            random_bytes = secrets.token_bytes(max(config.min_length, 32))
            return base64.urlsafe_b64encode(random_bytes).decode().rstrip("=")

        elif config.secret_type == SecretType.ENCRYPTION_KEY:
            # Generate Fernet encryption key
            return Fernet.generate_key().decode()

        else:
            # Default: random alphanumeric
            alphabet = string.ascii_letters + string.digits
            return "".join(secrets.choice(alphabet) for _ in range(config.min_length))

    async def _notify_services_of_rotation(
        self, dependencies: list[str], secret_name: str, new_value: str
    ) -> None:
        """Notify dependent services of secret rotation."""
        for service in dependencies:
            try:
                # In a production environment, this would trigger service restarts
                # or API calls to update secrets in running services
                logger.info(f"Notifying service {service} of {secret_name} rotation")

                # For now, we'll just log the event
                # In production, implement service-specific notification logic:
                # - Kubernetes rolling restarts
                # - Service API calls to update secrets
                # - Message queue notifications

            except Exception as e:
                logger.error(f"Failed to notify service {service} of rotation: {e}")

    async def _start_rotation_scheduler(self) -> None:
        """Start background tasks for automatic secret rotation."""
        for secret_name, config in self.secret_configs.items():
            task = asyncio.create_task(self._rotation_scheduler_task(secret_name))
            self._rotation_tasks[secret_name] = task

        logger.info(
            f"Started rotation scheduler for {len(self.secret_configs)} secrets"
        )

    async def _rotation_scheduler_task(self, secret_name: str) -> None:
        """Background task that schedules secret rotations."""
        config = self.secret_configs[secret_name]

        while True:
            try:
                # Calculate next rotation time
                metadata = self.secret_metadata.get(secret_name)
                if metadata and metadata.next_rotation:
                    next_rotation = metadata.next_rotation
                else:
                    # Schedule initial rotation
                    next_rotation = datetime.now() + timedelta(
                        hours=config.rotation_interval_hours
                    )

                # Wait until rotation time
                wait_seconds = (next_rotation - datetime.now()).total_seconds()
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)

                # Perform rotation
                await self.rotate_secret(secret_name)

            except asyncio.CancelledError:
                logger.info(f"Rotation scheduler for {secret_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in rotation scheduler for {secret_name}: {e}")
                # Wait before retrying
                await asyncio.sleep(3600)  # Wait 1 hour before retry

    async def _load_secret_metadata(self) -> None:
        """Load secret metadata from Vault or initialize defaults."""
        for secret_name in self.secret_configs:
            try:
                # Try to load metadata from Vault
                metadata_path = f"metadata/{secret_name}"
                response = self.vault_client.secrets.kv.v2.read_secret_version(
                    path=metadata_path, mount_point=self.mount_point
                )

                if response and "data" in response:
                    data = response["data"]["data"]
                    self.secret_metadata[secret_name] = SecretMetadata(
                        created_at=datetime.fromisoformat(data["created_at"]),
                        last_rotated=datetime.fromisoformat(data["last_rotated"]),
                        next_rotation=datetime.fromisoformat(data["next_rotation"]),
                        version=data["version"],
                        rotation_count=data.get("rotation_count", 0),
                    )

            except (KeyError, ValueError, TypeError, hvac.exceptions.InvalidPath) as e:
                # Metadata not found or invalid - initialize defaults
                logger.debug(
                    f"Could not load metadata for {secret_name}, using defaults: {e}"
                )
                config = self.secret_configs[secret_name]
                now = datetime.now()
                self.secret_metadata[secret_name] = SecretMetadata(
                    created_at=now,
                    last_rotated=now,
                    next_rotation=now + timedelta(hours=config.rotation_interval_hours),
                    version=1,
                )

    async def _update_secret_access_metadata(self, secret_name: str) -> None:
        """Update secret access metadata."""
        if secret_name in self.secret_metadata:
            self.secret_metadata[secret_name].last_access = datetime.now()

    async def _audit_log_event(
        self, event_type: str, description: str, success: bool = True
    ) -> None:
        """Log audit events for compliance and monitoring."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "description": description,
            "success": success,
            "vault_url": self.vault_url,
            "mount_point": self.mount_point,
        }

        self._audit_log.append(audit_entry)

        # Keep only last 1000 entries
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]

        # Log to standard logger
        log_level = logging.INFO if success else logging.ERROR
        logger.log(log_level, f"AUDIT: {event_type} - {description}")

    async def get_all_secrets(self) -> dict[str, str]:
        """
        Retrieve all configured secrets.
        WARNING: Use with caution in production.
        """
        secrets = {}
        for secret_name in self.secret_configs:
            secret_value = await self.get_secret(secret_name)
            if secret_value:
                secrets[secret_name] = secret_value
        return secrets

    async def health_check(self) -> dict[str, Any]:
        """Get health status and metrics."""
        try:
            vault_healthy = await self._test_vault_connection()
            authenticated = (
                self.vault_client.is_authenticated() if vault_healthy else False
            )

            return {
                "status": "healthy" if vault_healthy and authenticated else "unhealthy",
                "vault_connected": vault_healthy,
                "authenticated": authenticated,
                "initialized": self._initialized,
                "secrets_configured": len(self.secret_configs),
                "rotation_tasks_running": len(
                    [t for t in self._rotation_tasks.values() if not t.done()]
                ),
                "metrics": self.metrics.copy(),
                "cache_size": len(self._secret_cache),
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "vault_connected": False,
                "authenticated": False,
            }

    async def shutdown(self) -> None:
        """Gracefully shutdown the secrets manager."""
        logger.info("Shutting down Vault secrets manager")

        # Cancel rotation tasks
        for task in self._rotation_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._rotation_tasks:
            await asyncio.gather(*self._rotation_tasks.values(), return_exceptions=True)

        # Clear sensitive data
        self._secret_cache.clear()
        self._audit_log.clear()

        logger.info("Vault secrets manager shutdown complete")


# Global instance
_vault_manager: Optional[VaultSecretsManager] = None


async def get_vault_manager() -> VaultSecretsManager:
    """Get or create the global Vault secrets manager instance."""
    global _vault_manager

    if _vault_manager is None:
        _vault_manager = VaultSecretsManager()
        await _vault_manager.initialize()

    return _vault_manager


async def get_secret(secret_name: str) -> Optional[str]:
    """Convenience function to get a secret."""
    manager = await get_vault_manager()
    return await manager.get_secret(secret_name)


async def rotate_secret(secret_name: str) -> bool:
    """Convenience function to rotate a secret."""
    manager = await get_vault_manager()
    return await manager.rotate_secret(secret_name)
