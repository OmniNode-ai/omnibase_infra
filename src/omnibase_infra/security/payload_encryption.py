"""
Payload Encryption for ONEX Infrastructure Security

Provides comprehensive payload encryption for sensitive event payloads
and data in transit/at rest protection.

Per ONEX security requirements:
- AES-256-GCM encryption for payload data
- Key rotation and management
- Envelope encryption with data encryption keys (DEK)
- Secure key derivation and storage
"""

import os
import json
import base64
import logging
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend

from omnibase_core.core.onex_error import OnexError, CoreErrorCode


@dataclass
class EncryptionMetadata:
    """Metadata for encrypted payload."""
    algorithm: str
    key_id: str
    iv: Optional[str] = None
    tag: Optional[str] = None
    version: str = "1.0"
    timestamp: Optional[str] = None


@dataclass
class EncryptedPayload:
    """Container for encrypted payload and metadata."""
    encrypted_data: str
    metadata: EncryptionMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "encrypted_data": self.encrypted_data,
            "metadata": {
                "algorithm": self.metadata.algorithm,
                "key_id": self.metadata.key_id,
                "iv": self.metadata.iv,
                "tag": self.metadata.tag,
                "version": self.metadata.version,
                "timestamp": self.metadata.timestamp
            }
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedPayload':
        """Create from dictionary."""
        metadata = EncryptionMetadata(**data["metadata"])
        return cls(encrypted_data=data["encrypted_data"], metadata=metadata)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EncryptedPayload':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class ONEXPayloadEncryption:
    """
    ONEX payload encryption service for sensitive data protection.
    
    Features:
    - AES-256-GCM encryption with authenticated encryption
    - Key rotation and management
    - Envelope encryption pattern
    - Secure random key generation
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        
        # Encryption configuration
        self._algorithm = "AES-256-GCM"
        self._key_size = 32  # 256 bits
        self._iv_size = 12   # 96 bits for GCM
        
        # Key management
        self._current_key_id = self._get_current_key_id()
        self._keys: Dict[str, bytes] = {}
        self._initialize_keys()
        
        self._logger.info(f"Payload encryption initialized with {self._algorithm}")
    
    def _get_current_key_id(self) -> str:
        """Get current encryption key ID from environment."""
        return os.getenv("ONEX_ENCRYPTION_KEY_ID", "default-key-v1")
    
    def _initialize_keys(self):
        """Initialize encryption keys."""
        # Try to load key from environment or key management service
        key_material = os.getenv("ONEX_ENCRYPTION_KEY")
        
        if key_material:
            # Use provided key material
            key_bytes = self._derive_key_from_material(key_material)
        else:
            # Generate new key for development
            key_bytes = self._generate_key()
            self._logger.warning("Using generated encryption key - not suitable for production")
        
        self._keys[self._current_key_id] = key_bytes
    
    def _derive_key_from_material(self, key_material: str) -> bytes:
        """
        Derive encryption key from key material using PBKDF2.
        
        Args:
            key_material: Base key material (password/passphrase)
            
        Returns:
            Derived encryption key
        """
        # Use fixed salt for deterministic key derivation
        # In production, use unique salt per key and store securely
        salt = b"onex-infrastructure-salt-2024"
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self._key_size,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        return kdf.derive(key_material.encode())
    
    def _generate_key(self) -> bytes:
        """Generate new encryption key."""
        return os.urandom(self._key_size)
    
    def encrypt_payload(self, payload: Union[Dict[str, Any], str], 
                       key_id: Optional[str] = None,
                       compress: bool = True) -> EncryptedPayload:
        """
        Encrypt payload data.
        
        Args:
            payload: Data to encrypt (dict or string)
            key_id: Encryption key ID (uses current if not specified)
            compress: Whether to compress payload before encryption
            
        Returns:
            EncryptedPayload with encrypted data and metadata
            
        Raises:
            OnexError: If encryption fails
        """
        try:
            # Use current key if not specified
            if key_id is None:
                key_id = self._current_key_id
            
            if key_id not in self._keys:
                raise OnexError(
                    f"Encryption key not found: {key_id}",
                    CoreErrorCode.CONFIGURATION_ERROR
                )
            
            # Serialize payload using duck typing
            if hasattr(payload, 'keys') and hasattr(payload, 'items'):
                # Dict-like object
                payload_data = json.dumps(payload, sort_keys=True).encode('utf-8')
            else:
                # String-like object
                payload_data = payload.encode('utf-8')
            
            # Compress if requested
            if compress:
                import gzip
                payload_data = gzip.compress(payload_data)
            
            # Generate random IV
            iv = os.urandom(self._iv_size)
            
            # Encrypt using AES-GCM
            cipher = AESGCM(self._keys[key_id])
            encrypted_data = cipher.encrypt(iv, payload_data, None)
            
            # Split encrypted data and authentication tag
            ciphertext = encrypted_data[:-16]  # All but last 16 bytes
            tag = encrypted_data[-16:]         # Last 16 bytes (GCM tag)
            
            # Create metadata
            metadata = EncryptionMetadata(
                algorithm=self._algorithm,
                key_id=key_id,
                iv=base64.b64encode(iv).decode('ascii'),
                tag=base64.b64encode(tag).decode('ascii'),
                timestamp=str(int(time.time()))
            )
            
            # Encode encrypted data
            encrypted_b64 = base64.b64encode(ciphertext).decode('ascii')
            
            self._logger.debug(f"Encrypted payload with key {key_id}")
            
            return EncryptedPayload(
                encrypted_data=encrypted_b64,
                metadata=metadata
            )
            
        except Exception as e:
            raise OnexError(
                f"Payload encryption failed: {str(e)}",
                CoreErrorCode.ENCRYPTION_ERROR
            ) from e
    
    def decrypt_payload(self, encrypted_payload: EncryptedPayload,
                       return_dict: bool = True) -> Union[Dict[str, Any], str]:
        """
        Decrypt payload data.
        
        Args:
            encrypted_payload: Encrypted payload container
            return_dict: Whether to return dict (True) or string (False)
            
        Returns:
            Decrypted payload data
            
        Raises:
            OnexError: If decryption fails
        """
        try:
            metadata = encrypted_payload.metadata
            
            # Validate algorithm
            if metadata.algorithm != self._algorithm:
                raise OnexError(
                    f"Unsupported encryption algorithm: {metadata.algorithm}",
                    CoreErrorCode.ENCRYPTION_ERROR
                )
            
            # Get encryption key
            if metadata.key_id not in self._keys:
                raise OnexError(
                    f"Decryption key not found: {metadata.key_id}",
                    CoreErrorCode.CONFIGURATION_ERROR
                )
            
            # Decode components
            ciphertext = base64.b64decode(encrypted_payload.encrypted_data.encode('ascii'))
            iv = base64.b64decode(metadata.iv.encode('ascii'))
            tag = base64.b64decode(metadata.tag.encode('ascii'))
            
            # Reconstruct encrypted data with tag
            encrypted_data = ciphertext + tag
            
            # Decrypt using AES-GCM
            cipher = AESGCM(self._keys[metadata.key_id])
            decrypted_data = cipher.decrypt(iv, encrypted_data, None)
            
            # Decompress if needed (detect gzip magic bytes)
            if decrypted_data.startswith(b'\x1f\x8b'):
                import gzip
                decrypted_data = gzip.decompress(decrypted_data)
            
            # Convert to string
            payload_str = decrypted_data.decode('utf-8')
            
            self._logger.debug(f"Decrypted payload with key {metadata.key_id}")
            
            # Return as dict or string based on parameter
            if return_dict:
                try:
                    return json.loads(payload_str)
                except json.JSONDecodeError:
                    # If not valid JSON, return as string
                    return payload_str
            else:
                return payload_str
                
        except Exception as e:
            raise OnexError(
                f"Payload decryption failed: {str(e)}",
                CoreErrorCode.DECRYPTION_ERROR
            ) from e
    
    def add_encryption_key(self, key_id: str, key_material: str):
        """
        Add new encryption key for key rotation.
        
        Args:
            key_id: Unique identifier for the key
            key_material: Key material to derive encryption key from
        """
        key_bytes = self._derive_key_from_material(key_material)
        self._keys[key_id] = key_bytes
        
        self._logger.info(f"Added encryption key: {key_id}")
    
    def rotate_key(self, new_key_id: str, new_key_material: str):
        """
        Rotate to new encryption key.
        
        Args:
            new_key_id: New key identifier
            new_key_material: New key material
        """
        self.add_encryption_key(new_key_id, new_key_material)
        self._current_key_id = new_key_id
        
        self._logger.info(f"Rotated to new encryption key: {new_key_id}")
    
    def remove_encryption_key(self, key_id: str):
        """
        Remove encryption key (for key cleanup after rotation).
        
        Args:
            key_id: Key identifier to remove
        """
        if key_id == self._current_key_id:
            raise OnexError(
                "Cannot remove current encryption key",
                CoreErrorCode.CONFIGURATION_ERROR
            )
        
        if key_id in self._keys:
            del self._keys[key_id]
            self._logger.info(f"Removed encryption key: {key_id}")
    
    def get_available_keys(self) -> List[str]:
        """
        Get list of available key IDs.
        
        Returns:
            List of available key identifiers
        """
        return list(self._keys.keys())
    
    def is_payload_encrypted(self, data: Union[str, Dict[str, Any]]) -> bool:
        """
        Check if data appears to be an encrypted payload.
        
        Args:
            data: Data to check
            
        Returns:
            True if data appears to be encrypted payload
        """
        try:
            # Check if data is string-like and parse if needed
            if hasattr(data, 'strip') and hasattr(data, 'replace'):
                data = json.loads(data)
            
            # Check if data is dict-like and has required keys
            return (hasattr(data, 'keys') and hasattr(data, 'get') and 
                   "encrypted_data" in data and 
                   "metadata" in data and
                   "algorithm" in data.get("metadata", {}))
        
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
    
    def encrypt_if_sensitive(self, data: Dict[str, Any], 
                           sensitive_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Conditionally encrypt sensitive fields in a dictionary.
        
        Args:
            data: Data dictionary to process
            sensitive_fields: List of field names to encrypt (if None, use defaults)
            
        Returns:
            Dictionary with sensitive fields encrypted
        """
        if sensitive_fields is None:
            # Default sensitive field patterns
            sensitive_fields = [
                'password', 'secret', 'token', 'key', 'credential',
                'ssn', 'credit_card', 'account_number', 'personal_data'
            ]
        
        result = {}
        
        for key, value in data.items():
            # Check if field should be encrypted
            should_encrypt = any(pattern in key.lower() for pattern in sensitive_fields)
            
            # Check if value should be encrypted using duck typing
            should_encrypt_value = should_encrypt and (
                (hasattr(value, 'strip') and hasattr(value, 'replace')) or  # String-like
                (hasattr(value, 'keys') and hasattr(value, 'items'))       # Dict-like
            )
            
            if should_encrypt_value:
                # Encrypt sensitive field
                encrypted = self.encrypt_payload(value)
                result[f"{key}_encrypted"] = encrypted.to_dict()
            else:
                result[key] = value
        
        return result


# Import time at module level for timestamp generation
import time


# Global payload encryption instance
_payload_encryption: Optional[ONEXPayloadEncryption] = None


def get_payload_encryption() -> ONEXPayloadEncryption:
    """
    Get global payload encryption instance.
    
    Returns:
        ONEXPayloadEncryption singleton instance
    """
    global _payload_encryption
    
    if _payload_encryption is None:
        _payload_encryption = ONEXPayloadEncryption()
    
    return _payload_encryption