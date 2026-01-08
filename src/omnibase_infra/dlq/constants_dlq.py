# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""DLQ Constants and Shared Validation Patterns.

This module provides shared constants for the DLQ subsystem, ensuring
consistent validation patterns across models and services.

Defense-in-Depth Note:
    The PATTERN_TABLE_NAME constant is intentionally used in BOTH:
    1. Pydantic model validation (ModelDlqTrackingConfig.storage_table field)
    2. Runtime validation (ServiceDlqTracking._validate_storage_table method)

    This defense-in-depth approach ensures SQL injection prevention even if:
    - Direct attribute assignment bypasses Pydantic validation
    - Deserialization from untrusted sources bypasses model validation
    - Future code changes inadvertently bypass config validation

    DO NOT remove either validation layer. Both are intentional and required.

Related:
    - model_dlq_tracking_config.py - Pydantic field pattern validation
    - service_dlq_tracking.py - ServiceDlqTracking runtime defense-in-depth validation
    - OMN-1032 - PostgreSQL tracking integration ticket
"""

from __future__ import annotations

import re

# =============================================================================
# TABLE NAME VALIDATION
# =============================================================================

# Regex pattern string for valid PostgreSQL table names
# Must start with letter or underscore, followed by letters, digits, or underscores
# This matches PostgreSQL's identifier naming rules for unquoted identifiers
#
# Used in:
#   - ModelDlqTrackingConfig: Pydantic field pattern constraint
#   - ServiceDlqTracking: Runtime defense-in-depth validation
#
# Pattern explanation:
#   ^           - Start of string
#   [a-zA-Z_]   - First character must be letter (a-z, A-Z) or underscore
#   [a-zA-Z0-9_]* - Subsequent characters can be letters, digits, or underscores
#   $           - End of string
PATTERN_TABLE_NAME = r"^[a-zA-Z_][a-zA-Z0-9_]*$"

# Pre-compiled regex for runtime validation (avoids recompilation overhead)
# This is the compiled version of PATTERN_TABLE_NAME for use in service code
REGEX_TABLE_NAME = re.compile(PATTERN_TABLE_NAME)


__all__: list[str] = [
    "PATTERN_TABLE_NAME",
    "REGEX_TABLE_NAME",
]
