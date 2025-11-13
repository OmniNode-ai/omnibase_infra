#!/usr/bin/env python3
"""
Authentication models for deployment receiver effect node.
ONEX v2.0 compliant data models with HMAC and IP whitelisting support.
"""

import ipaddress
from typing import ClassVar, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ModelAuthCredentials(BaseModel):
    """
    Authentication credentials for deployment package.

    Attributes:
        sender_id: Unique identifier for the sender (UUID)
        auth_token: Authentication token (min 32 chars)
        signature: HMAC signature for package validation
        sender_ip: IP address of the sender (for whitelisting)
    """

    sender_id: UUID = Field(..., description="Unique identifier for the sender")

    auth_token: str = Field(
        ..., min_length=32, description="Authentication token for sender validation"
    )

    signature: str = Field(
        ..., description="HMAC signature for package integrity validation"
    )

    sender_ip: Optional[str] = Field(
        None, description="IP address of the sender (for whitelisting)"
    )

    @field_validator("sender_ip")
    @classmethod
    def validate_ip(cls, v: Optional[str]) -> Optional[str]:
        """Validate IP address format."""
        if v is not None:
            try:
                ipaddress.ip_address(v)
            except ValueError as e:
                raise ValueError(f"Invalid IP address format: {v}") from e
        return v

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "sender_id": "550e8400-e29b-41d4-a716-446655440000",
                "auth_token": "a" * 32,
                "signature": "b" * 64,
                "sender_ip": "192.168.86.101",
            }
        }


class ModelAuthValidationResult(BaseModel):
    """
    Result of authentication validation.

    Attributes:
        is_valid: Whether authentication passed
        error_code: Error code if validation failed
        error_message: Human-readable error message
        sender_id: Validated sender ID
    """

    is_valid: bool = Field(..., description="Whether authentication validation passed")

    error_code: Optional[str] = Field(
        None, description="Error code if validation failed"
    )

    error_message: Optional[str] = Field(
        None, description="Human-readable error message"
    )

    sender_id: Optional[UUID] = Field(None, description="Validated sender ID")

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "is_valid": True,
                "error_code": None,
                "error_message": None,
                "sender_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }


class ModelChecksumValidationResult(BaseModel):
    """
    Result of BLAKE3 checksum validation.

    Attributes:
        is_valid: Whether checksum matches
        expected_checksum: Expected BLAKE3 hash
        actual_checksum: Computed BLAKE3 hash
        error_message: Error message if validation failed
    """

    is_valid: bool = Field(..., description="Whether checksum validation passed")

    expected_checksum: str = Field(
        ...,
        pattern=r"^[a-f0-9]{64}$",
        description="Expected BLAKE3 hash (64 hex chars)",
    )

    actual_checksum: str = Field(
        ...,
        pattern=r"^[a-f0-9]{64}$",
        description="Computed BLAKE3 hash (64 hex chars)",
    )

    error_message: Optional[str] = Field(
        None, description="Error message if validation failed"
    )

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "is_valid": True,
                "expected_checksum": "a" * 64,
                "actual_checksum": "a" * 64,
                "error_message": None,
            }
        }


class ModelIPWhitelistValidationResult(BaseModel):
    """
    Result of IP whitelisting validation.

    Attributes:
        is_allowed: Whether IP is in whitelist
        ip_address: IP address being validated
        matched_range: CIDR range that matched (if any)
        error_message: Error message if validation failed
    """

    is_allowed: bool = Field(..., description="Whether IP address is in whitelist")

    ip_address: str = Field(..., description="IP address being validated")

    matched_range: Optional[str] = Field(None, description="CIDR range that matched")

    error_message: Optional[str] = Field(
        None, description="Error message if validation failed"
    )

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "is_allowed": True,
                "ip_address": "192.168.86.101",
                "matched_range": "192.168.86.0/24",
                "error_message": None,
            }
        }


__all__ = [
    "ModelAuthCredentials",
    "ModelAuthValidationResult",
    "ModelChecksumValidationResult",
    "ModelIPWhitelistValidationResult",
]
