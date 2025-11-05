"""
Credential type enum.

Strongly typed values for credential types.
"""

from enum import Enum


class EnumCredentialType(str, Enum):
    """Strongly typed credential type values."""

    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    TOKEN = "token"
    OAUTH_TOKEN = "oauth_token"
    JWT_TOKEN = "jwt_token"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"


# Export for use
__all__ = ["EnumCredentialType"]
