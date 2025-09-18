"""Credential type enumeration for credential cache configuration."""

from enum import Enum


class EnumCredentialType(str, Enum):
    """Enumeration for credential types."""

    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    TOKEN = "token"
    BASIC_AUTH = "basic_auth"
    OAUTH = "oauth"
    JWT = "jwt"
