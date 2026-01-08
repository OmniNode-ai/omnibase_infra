"""Shared utility functions for omnibase_infra.

IMPORTANT: Directory Structure Clarification
--------------------------------------------
This module (shared/utils) is a placeholder for future shared utilities.
The main utility implementations are in omnibase_infra.utils, NOT here.

Directory structure:
- omnibase_infra/utils/ - Main utility implementations (use this)
  - util_error_sanitization.py - Error sanitization functions
  - util_dsn_validation.py - DSN validation
  - util_env_parsing.py - Environment parsing
  - util_semver.py - Semantic versioning
  - correlation.py - Correlation ID management

- omnibase_infra/shared/utils/ - Reserved for cross-package shared utilities
  (currently empty, may be removed in future refactoring)

Usage:
    # Import from the main utils module
    from omnibase_infra.utils import (
        sanitize_backend_error,
        sanitize_error_message,
        sanitize_error_string,
    )
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'omnibase_infra.shared.utils' is deprecated. "
    "Use 'omnibase_infra.utils' instead. "
    "This module will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__: list[str] = []
