# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security models for handler policy validation.

This module exports security-related models for the two-layer handler
security validation system (OMN-1098). These models define:

1. Handler-declared security policies (what a handler needs)
2. Environment-level constraints (what an environment permits)
3. Validation results with errors and warnings (OMN-1137)

The security validation system uses these models to:
- Validate handler security requirements at registration time
- Enforce security constraints at invocation time
- Provide structured error reporting for violations

See Also:
    - EnumSecurityRuleId: Security validation rule identifiers
    - docs/design/HANDLER_SECURITY_VALIDATION.md: Full design documentation
"""

from omnibase_infra.models.security.classification_levels import (
    CLASSIFICATION_SECURITY_LEVELS,
    get_security_level,
)
from omnibase_infra.models.security.model_environment_policy import (
    ModelEnvironmentPolicy,
)
from omnibase_infra.models.security.model_handler_security_policy import (
    ModelHandlerSecurityPolicy,
)
from omnibase_infra.models.security.model_security_error import (
    ModelSecurityError,
)
from omnibase_infra.models.security.model_security_validation_result import (
    ModelSecurityValidationResult,
)
from omnibase_infra.models.security.model_security_warning import (
    ModelSecurityWarning,
)

__all__ = [
    "CLASSIFICATION_SECURITY_LEVELS",
    "ModelEnvironmentPolicy",
    "ModelHandlerSecurityPolicy",
    "ModelSecurityError",
    "ModelSecurityValidationResult",
    "ModelSecurityWarning",
    "get_security_level",
]
