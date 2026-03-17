# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Environment enumeration for deployment target classification.

This module defines the deployment environment types used for security policy
validation and configuration management. Environment-aware security policies
can enforce different constraints based on the deployment target.

Usage:
    The EnumEnvironment is used in security policy models to define
    environment-specific constraints and in configuration to specify
    the current deployment target.

See Also:
    - ModelEnvironmentPolicy: Uses this enum for environment-specific constraints
    - ModelHandlerSecurityPolicy: Handler-declared security requirements
"""

from enum import Enum


class EnumEnvironment(str, Enum):
    """Deployment environment classification.

    Defines the deployment environments for security policy enforcement
    and configuration management. Each environment has different security
    postures and permitted capabilities.

    Note:
        The canonical ONEX_ENVIRONMENT values are ``local``, ``staging``,
        ``production``, and ``ci``.  Legacy code that used ``"dev"`` as a
        default should be migrated to ``"local"`` (see OMN-5204).

    Attributes:
        DEVELOPMENT: Local development environment.
            Most permissive, allows debugging features and relaxed security.
            Legacy references to ``"dev"`` should use ``"local"`` instead.
        STAGING: Pre-production testing environment.
            Production-like but allows some testing features.
        PRODUCTION: Live production environment.
            Most restrictive, enforces all security constraints.
        CI: Continuous integration environment.
            Automated testing, may have elevated permissions for test execution.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CI = "ci"


__all__ = ["EnumEnvironment"]
