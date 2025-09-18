"""
Deployment environment enum.

Strongly typed values for deployment environments.
"""

from enum import Enum


class EnumDeploymentEnvironment(str, Enum):
    """Strongly typed deployment environment values."""

    DEVELOPMENT = "development"
    DEV = "dev"
    TESTING = "testing"
    TEST = "test"
    STAGING = "staging"
    STAGE = "stage"
    PRODUCTION = "production"
    PROD = "prod"
    LOCAL = "local"
    SANDBOX = "sandbox"


# Export for use
__all__ = ["EnumDeploymentEnvironment"]