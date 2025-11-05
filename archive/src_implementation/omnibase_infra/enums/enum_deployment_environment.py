"""Deployment environment enumeration for infrastructure configuration."""

from enum import Enum


class EnumDeploymentEnvironment(str, Enum):
    """Enumeration for deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"
