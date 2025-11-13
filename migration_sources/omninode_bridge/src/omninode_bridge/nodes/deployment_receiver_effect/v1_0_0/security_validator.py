#!/usr/bin/env python3
"""
Security validation module for deployment receiver effect node.
Implements HMAC authentication, BLAKE3 checksum validation, and IP whitelisting.
"""

import hashlib
import hmac
import ipaddress
from pathlib import Path
from typing import Optional

try:
    import blake3
except ImportError:
    blake3 = None  # Will raise error at runtime if used

from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event

from .models.model_auth import (
    ModelAuthCredentials,
    ModelAuthValidationResult,
    ModelChecksumValidationResult,
    ModelIPWhitelistValidationResult,
)


class SecurityValidator:
    """
    Security validation for deployment receiver operations.

    Features:
    - HMAC signature validation
    - BLAKE3 checksum verification
    - IP whitelisting with CIDR support
    - Token expiry validation (if implemented)
    """

    def __init__(
        self, secret_key: str, allowed_ip_ranges: Optional[list[str]] = None
    ) -> None:
        """
        Initialize security validator.

        Args:
            secret_key: Secret key for HMAC signature validation
            allowed_ip_ranges: List of allowed IP ranges in CIDR notation
                              (e.g., ["192.168.86.0/24", "10.0.0.0/8"])
        """
        self.secret_key = secret_key.encode("utf-8")
        self.allowed_ip_ranges = allowed_ip_ranges or [
            "192.168.86.0/24",  # Local network
            "10.0.0.0/8",  # Private network
        ]

        # Parse allowed IP ranges
        self.parsed_ranges = []
        for ip_range in self.allowed_ip_ranges:
            try:
                self.parsed_ranges.append(ipaddress.ip_network(ip_range))
            except ValueError as e:
                emit_log_event(
                    LogLevel.WARNING,
                    f"Invalid IP range in whitelist: {ip_range}",
                    {"error": str(e)},
                )

        emit_log_event(
            LogLevel.INFO,
            "SecurityValidator initialized",
            {
                "allowed_ip_ranges": self.allowed_ip_ranges,
                "parsed_ranges_count": len(self.parsed_ranges),
            },
        )

    def validate_hmac_signature(
        self, credentials: ModelAuthCredentials, message: bytes
    ) -> ModelAuthValidationResult:
        """
        Validate HMAC signature for authentication.

        Args:
            credentials: Authentication credentials with signature
            message: Message bytes that were signed

        Returns:
            Authentication validation result
        """
        try:
            # Compute expected signature
            expected_signature = hmac.new(
                self.secret_key, message, hashlib.sha256
            ).hexdigest()

            # Compare signatures (constant-time comparison)
            is_valid = hmac.compare_digest(credentials.signature, expected_signature)

            if is_valid:
                emit_log_event(
                    LogLevel.INFO,
                    "HMAC authentication succeeded",
                    {"sender_id": str(credentials.sender_id)},
                )
                return ModelAuthValidationResult(
                    is_valid=True, sender_id=credentials.sender_id
                )
            else:
                emit_log_event(
                    LogLevel.WARNING,
                    "HMAC authentication failed: signature mismatch",
                    {
                        "sender_id": str(credentials.sender_id),
                        "sender_ip": credentials.sender_ip,
                    },
                )
                return ModelAuthValidationResult(
                    is_valid=False,
                    error_code="AUTHENTICATION_FAILED",
                    error_message="HMAC signature mismatch",
                )

        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                f"HMAC validation error: {e!s}",
                {"sender_id": str(credentials.sender_id), "error": str(e)},
            )
            return ModelAuthValidationResult(
                is_valid=False,
                error_code="AUTHENTICATION_ERROR",
                error_message=f"HMAC validation error: {e!s}",
            )

    def validate_checksum(
        self, file_path: str, expected_checksum: str
    ) -> ModelChecksumValidationResult:
        """
        Validate BLAKE3 checksum of file.

        Args:
            file_path: Path to file to validate
            expected_checksum: Expected BLAKE3 hash (64 hex chars)

        Returns:
            Checksum validation result
        """
        if blake3 is None:
            return ModelChecksumValidationResult(
                is_valid=False,
                expected_checksum=expected_checksum,
                actual_checksum="0" * 64,
                error_message="blake3 library not installed",
            )

        try:
            # Compute BLAKE3 hash
            hasher = blake3.blake3()
            path = Path(file_path)

            if not path.exists():
                return ModelChecksumValidationResult(
                    is_valid=False,
                    expected_checksum=expected_checksum,
                    actual_checksum="0" * 64,
                    error_message=f"File not found: {file_path}",
                )

            # Read file in chunks for memory efficiency
            chunk_size = 1024 * 1024  # 1MB chunks
            with path.open("rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)

            actual_checksum = hasher.hexdigest()

            # Compare checksums (constant-time comparison)
            is_valid = hmac.compare_digest(
                expected_checksum.lower(), actual_checksum.lower()
            )

            if is_valid:
                emit_log_event(
                    LogLevel.INFO,
                    "BLAKE3 checksum validation succeeded",
                    {"file_path": file_path},
                )
            else:
                emit_log_event(
                    LogLevel.WARNING,
                    "BLAKE3 checksum validation failed",
                    {
                        "file_path": file_path,
                        "expected": expected_checksum,
                        "actual": actual_checksum,
                    },
                )

            return ModelChecksumValidationResult(
                is_valid=is_valid,
                expected_checksum=expected_checksum,
                actual_checksum=actual_checksum,
                error_message=None if is_valid else "Checksum mismatch",
            )

        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                f"Checksum validation error: {e!s}",
                {"file_path": file_path, "error": str(e)},
            )
            return ModelChecksumValidationResult(
                is_valid=False,
                expected_checksum=expected_checksum,
                actual_checksum="0" * 64,
                error_message=f"Checksum validation error: {e!s}",
            )

    def validate_ip_whitelist(
        self, ip_address: str
    ) -> ModelIPWhitelistValidationResult:
        """
        Validate IP address against whitelist.

        Args:
            ip_address: IP address to validate

        Returns:
            IP whitelist validation result
        """
        try:
            # Parse IP address
            ip = ipaddress.ip_address(ip_address)

            # Check against all allowed ranges
            for ip_range in self.parsed_ranges:
                if ip in ip_range:
                    emit_log_event(
                        LogLevel.INFO,
                        "IP whitelist validation succeeded",
                        {"ip_address": ip_address, "matched_range": str(ip_range)},
                    )
                    return ModelIPWhitelistValidationResult(
                        is_allowed=True,
                        ip_address=ip_address,
                        matched_range=str(ip_range),
                    )

            # No match found
            emit_log_event(
                LogLevel.WARNING,
                "IP whitelist validation failed: not in allowed ranges",
                {"ip_address": ip_address, "allowed_ranges": self.allowed_ip_ranges},
            )
            return ModelIPWhitelistValidationResult(
                is_allowed=False,
                ip_address=ip_address,
                error_message=f"IP {ip_address} not in allowed ranges",
            )

        except ValueError as e:
            emit_log_event(
                LogLevel.ERROR,
                f"IP whitelist validation error: {e!s}",
                {"ip_address": ip_address, "error": str(e)},
            )
            return ModelIPWhitelistValidationResult(
                is_allowed=False,
                ip_address=ip_address,
                error_message=f"Invalid IP address: {e!s}",
            )

    def validate_all(
        self,
        credentials: ModelAuthCredentials,
        file_path: str,
        expected_checksum: str,
        message: bytes,
    ) -> tuple[
        ModelAuthValidationResult,
        ModelChecksumValidationResult,
        ModelIPWhitelistValidationResult,
    ]:
        """
        Perform all security validations.

        Args:
            credentials: Authentication credentials
            file_path: Path to file to validate
            expected_checksum: Expected BLAKE3 checksum
            message: Message bytes for HMAC validation

        Returns:
            Tuple of (auth_result, checksum_result, ip_result)
        """
        # Validate HMAC
        auth_result = self.validate_hmac_signature(credentials, message)

        # Validate checksum
        checksum_result = self.validate_checksum(file_path, expected_checksum)

        # Validate IP (if provided)
        if credentials.sender_ip:
            ip_result = self.validate_ip_whitelist(credentials.sender_ip)
        else:
            # No IP provided - allow for local testing
            ip_result = ModelIPWhitelistValidationResult(
                is_allowed=True,
                ip_address="unknown",
                error_message="No IP provided (local testing)",
            )

        return auth_result, checksum_result, ip_result


__all__ = ["SecurityValidator"]
