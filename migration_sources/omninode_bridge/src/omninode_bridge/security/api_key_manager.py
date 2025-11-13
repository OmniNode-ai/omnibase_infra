"""API key management with rotation and security features."""

import asyncio
import hashlib
import hmac
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import aiofiles
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from pydantic import BaseModel

from .audit_logger import AuditEventType, AuditSeverity, get_audit_logger


class ApiKeyMetadata(BaseModel):
    """Metadata for API key tracking."""

    key_id: str
    created_at: datetime
    expires_at: datetime | None
    last_used: datetime | None = None
    usage_count: int = 0
    is_active: bool = True
    created_by: str = "system"
    description: str = ""


class ApiKeyManager:
    """Secure API key management with rotation capabilities."""

    def __init__(
        self,
        service_name: str,
        key_storage_path: str = "/app/secrets",
        rotation_interval_hours: int = 24 * 7,  # 7 days default
        max_keys_retained: int = 3,
    ):
        """Initialize API key manager.

        Args:
            service_name: Name of service for audit logging
            key_storage_path: Path to store encrypted key metadata
            rotation_interval_hours: Hours between automatic rotations
            max_keys_retained: Maximum number of old keys to retain
        """
        self.service_name = service_name
        self.key_storage_path = key_storage_path
        self.rotation_interval = timedelta(hours=rotation_interval_hours)
        self.max_keys_retained = max_keys_retained

        # Initialize audit logger
        self.audit_logger = get_audit_logger(service_name)

        # In-memory key cache for performance
        self._key_cache: dict[str, ApiKeyMetadata] = {}
        self._current_key_id: str | None = None

        # Encryption key for storing metadata (derived from environment)
        self._encryption_key = self._derive_encryption_key()

    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from environment variables."""
        # API_KEY_ENCRYPTION_SEED is required for secure key derivation
        encryption_seed = os.getenv("API_KEY_ENCRYPTION_SEED")
        if not encryption_seed:
            raise ValueError(
                "API_KEY_ENCRYPTION_SEED environment variable must be set. "
                "Generate a secure seed with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )

        # Use multiple environment variables for key derivation
        seed = (
            encryption_seed
            + os.getenv("SERVICE_INSTANCE_ID", "default")
            + self.service_name
        )

        # Use PBKDF2 for key derivation
        return hashlib.pbkdf2_hmac(
            "sha256",
            seed.encode("utf-8"),
            b"omninode-bridge-salt",
            100000,  # 100k iterations
            32,  # 256-bit key
        )

    async def initialize(self) -> None:
        """Initialize the key manager and load existing keys."""
        try:
            # Ensure storage directory exists
            os.makedirs(self.key_storage_path, mode=0o700, exist_ok=True)

            # Load existing keys
            await self._load_keys()

            # Create initial key if none exist
            if not self._key_cache:
                await self.generate_new_key("Initial API key")

            # Set current key
            self._update_current_key()

            self.audit_logger.log_event(
                event_type=AuditEventType.SERVICE_STARTUP,
                severity=AuditSeverity.LOW,
                additional_data={
                    "component": "api_key_manager",
                    "active_keys": len(
                        [k for k in self._key_cache.values() if k.is_active],
                    ),
                    "current_key_id": self._current_key_id,
                },
                message="API key manager initialized successfully",
            )

        except Exception as e:
            self.audit_logger.log_event(
                event_type=AuditEventType.SERVICE_STARTUP,
                severity=AuditSeverity.CRITICAL,
                additional_data={
                    "component": "api_key_manager",
                    "error": str(e),
                },
                message=f"Failed to initialize API key manager: {e}",
            )
            raise

    async def _load_keys(self) -> None:
        """Load existing keys from storage."""
        keys_file = os.path.join(self.key_storage_path, "api_keys.enc")

        if not os.path.exists(keys_file):
            return

        try:
            async with aiofiles.open(keys_file, "rb") as f:
                encrypted_data = await f.read()

            # Decrypt and load keys
            decrypted_data = self._decrypt_data(encrypted_data)
            import json

            keys_data = json.loads(decrypted_data.decode("utf-8"))

            for key_data in keys_data:
                metadata = ApiKeyMetadata(**key_data)
                self._key_cache[metadata.key_id] = metadata

        except Exception as e:
            self.audit_logger.log_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                severity=AuditSeverity.HIGH,
                additional_data={
                    "component": "api_key_manager",
                    "operation": "load_keys",
                    "error": str(e),
                },
                message=f"Failed to load API keys: {e}",
            )
            # Continue with empty cache - will generate new key

    async def _save_keys(self) -> None:
        """Save keys to encrypted storage."""
        keys_file = os.path.join(self.key_storage_path, "api_keys.enc")

        try:
            # Serialize key metadata
            import json

            keys_data = [metadata.dict() for metadata in self._key_cache.values()]
            data = json.dumps(keys_data, default=str).encode("utf-8")

            # Encrypt and save
            encrypted_data = self._encrypt_data(data)

            async with aiofiles.open(keys_file, "wb") as f:
                await f.write(encrypted_data)

            # Set secure permissions
            os.chmod(keys_file, 0o600)

        except Exception as e:
            self.audit_logger.log_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                severity=AuditSeverity.HIGH,
                additional_data={
                    "component": "api_key_manager",
                    "operation": "save_keys",
                    "error": str(e),
                },
                message=f"Failed to save API keys: {e}",
            )
            raise

    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using AES-256-GCM."""
        # Generate random IV
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM

        # Create cipher
        cipher = Cipher(algorithms.AES(self._encryption_key), modes.GCM(iv))
        encryptor = cipher.encryptor()

        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # Return IV + tag + ciphertext
        return iv + encryptor.tag + ciphertext

    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using AES-256-GCM."""
        # Extract components
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]

        # Create cipher
        cipher = Cipher(algorithms.AES(self._encryption_key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()

        # Decrypt data
        return decryptor.update(ciphertext) + decryptor.finalize()

    async def generate_new_key(self, description: str = "") -> str:
        """Generate a new API key and metadata.

        Args:
            description: Description of the key usage

        Returns:
            The new API key ID
        """
        key_id = str(uuid4())
        now = datetime.now(UTC)

        # Generate secure API key (64 bytes = 512 bits)
        api_key = secrets.token_urlsafe(64)

        # Store in environment variable for current use
        os.environ[f"API_KEY_{key_id}"] = api_key

        # Create metadata
        metadata = ApiKeyMetadata(
            key_id=key_id,
            created_at=now,
            expires_at=now + self.rotation_interval,
            last_used=None,
            description=description,
            is_active=True,
        )

        self._key_cache[key_id] = metadata
        await self._save_keys()

        self.audit_logger.log_event(
            event_type=AuditEventType.AUTHENTICATION_SUCCESS,
            severity=AuditSeverity.MEDIUM,
            additional_data={
                "component": "api_key_manager",
                "operation": "generate_key",
                "key_id": key_id,
                "description": description,
                "expires_at": metadata.expires_at.isoformat(),
            },
            message=f"New API key generated: {key_id}",
        )

        return key_id

    async def rotate_keys(self, force: bool = False) -> bool:
        """Rotate API keys if needed.

        Args:
            force: Force rotation even if not due

        Returns:
            True if rotation occurred
        """
        current_key = self._get_current_key_metadata()

        # Check if rotation is needed
        if not force and current_key:
            time_until_expiry = current_key.expires_at - datetime.now(UTC)
            if time_until_expiry > timedelta(hours=1):  # 1 hour grace period
                return False

        try:
            # Generate new key
            new_key_id = await self.generate_new_key("Automatic rotation")

            # Deactivate old keys beyond retention limit
            await self._cleanup_old_keys()

            # Update current key
            self._update_current_key()

            self.audit_logger.log_event(
                event_type=AuditEventType.AUTHENTICATION_SUCCESS,
                severity=AuditSeverity.MEDIUM,
                additional_data={
                    "component": "api_key_manager",
                    "operation": "rotate_keys",
                    "new_key_id": new_key_id,
                    "forced": force,
                    "active_keys": len(
                        [k for k in self._key_cache.values() if k.is_active],
                    ),
                },
                message=f"API keys rotated successfully, new key: {new_key_id}",
            )

            return True

        except Exception as e:
            self.audit_logger.log_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                severity=AuditSeverity.CRITICAL,
                additional_data={
                    "component": "api_key_manager",
                    "operation": "rotate_keys",
                    "error": str(e),
                },
                message=f"Failed to rotate API keys: {e}",
            )
            raise

    async def _cleanup_old_keys(self) -> None:
        """Remove old keys beyond retention limit."""
        active_keys = [
            (key_id, metadata)
            for key_id, metadata in self._key_cache.items()
            if metadata.is_active
        ]

        # Sort by creation date (newest first)
        active_keys.sort(key=lambda x: x[1].created_at, reverse=True)

        # Deactivate keys beyond retention limit
        for key_id, metadata in active_keys[self.max_keys_retained :]:
            metadata.is_active = False

            # Remove from environment
            env_key = f"API_KEY_{key_id}"
            if env_key in os.environ:
                del os.environ[env_key]

            self.audit_logger.log_event(
                event_type=AuditEventType.AUTHENTICATION_SUCCESS,
                severity=AuditSeverity.LOW,
                additional_data={
                    "component": "api_key_manager",
                    "operation": "cleanup_key",
                    "key_id": key_id,
                    "key_age_days": (datetime.now(UTC) - metadata.created_at).days,
                },
                message=f"Deactivated old API key: {key_id}",
            )

        await self._save_keys()

    def _update_current_key(self) -> None:
        """Update the current active key."""
        active_keys = [
            (key_id, metadata)
            for key_id, metadata in self._key_cache.items()
            if metadata.is_active
        ]

        if not active_keys:
            self._current_key_id = None
            return

        # Use newest active key
        active_keys.sort(key=lambda x: x[1].created_at, reverse=True)
        self._current_key_id = active_keys[0][0]

        # Update main environment variable
        current_key = self.get_current_api_key()
        if current_key:
            os.environ["API_KEY"] = current_key

    def _get_current_key_metadata(self) -> ApiKeyMetadata | None:
        """Get metadata for current key."""
        if not self._current_key_id:
            return None
        return self._key_cache.get(self._current_key_id)

    def get_current_api_key(self) -> str | None:
        """Get the current active API key."""
        if not self._current_key_id:
            return None

        env_key = f"API_KEY_{self._current_key_id}"
        return os.environ.get(env_key)

    async def validate_api_key(self, provided_key: str) -> tuple[bool, str | None]:
        """Validate provided API key against active keys.

        Args:
            provided_key: The API key to validate

        Returns:
            Tuple of (is_valid, key_id)
        """
        for key_id, metadata in self._key_cache.items():
            if not metadata.is_active:
                continue

            # Check if key has expired
            if metadata.expires_at and datetime.now(UTC) > metadata.expires_at:
                continue

            env_key = f"API_KEY_{key_id}"
            stored_key = os.environ.get(env_key)

            if stored_key and hmac.compare_digest(stored_key, provided_key):
                # Update usage statistics
                metadata.last_used = datetime.now(UTC)
                metadata.usage_count += 1
                await self._save_keys()

                return True, key_id

        return False, None

    async def get_key_status(self) -> dict[str, Any]:
        """Get status of all keys."""
        return {
            "current_key_id": self._current_key_id,
            "active_keys": len([k for k in self._key_cache.values() if k.is_active]),
            "total_keys": len(self._key_cache),
            "keys": [
                {
                    "key_id": key_id,
                    "created_at": metadata.created_at.isoformat(),
                    "expires_at": (
                        metadata.expires_at.isoformat() if metadata.expires_at else None
                    ),
                    "last_used": (
                        metadata.last_used.isoformat() if metadata.last_used else None
                    ),
                    "usage_count": metadata.usage_count,
                    "is_active": metadata.is_active,
                    "description": metadata.description,
                }
                for key_id, metadata in self._key_cache.items()
            ],
        }


# Global API key manager instance
_api_key_manager: ApiKeyManager | None = None


async def get_api_key_manager(service_name: str) -> ApiKeyManager:
    """Get or create the global API key manager."""
    global _api_key_manager

    if _api_key_manager is None:
        _api_key_manager = ApiKeyManager(service_name)
        await _api_key_manager.initialize()

    return _api_key_manager


async def setup_automatic_rotation(service_name: str, check_interval_hours: int = 1):
    """Setup automatic key rotation background task."""
    manager = await get_api_key_manager(service_name)

    async def rotation_task():
        while True:
            try:
                await asyncio.sleep(check_interval_hours * 3600)  # Convert to seconds
                await manager.rotate_keys(force=False)
            except Exception as e:
                # Log error but continue
                manager.audit_logger.log_event(
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    severity=AuditSeverity.HIGH,
                    additional_data={
                        "component": "api_key_rotation",
                        "error": str(e),
                    },
                    message=f"Automatic key rotation failed: {e}",
                )

    # Start background task
    asyncio.create_task(rotation_task())
