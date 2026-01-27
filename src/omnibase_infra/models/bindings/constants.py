# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared constants for binding expression parsing and validation.

This module contains guardrail constants and patterns used by both:
- :mod:`omnibase_infra.runtime.binding_resolver` (runtime resolution)
- :mod:`omnibase_infra.runtime.contract_loaders.operation_bindings_loader` (load-time validation)

These constants define the security and validation boundaries for binding expressions.

.. versionadded:: 0.2.6
    Created as part of OMN-1518 - Declarative operation bindings.
"""

from __future__ import annotations

import re
from typing import Final

# =============================================================================
# Guardrail Constants
# =============================================================================

MAX_EXPRESSION_LENGTH: Final[int] = 256
"""Maximum allowed length for binding expressions (characters).

Prevents denial-of-service via extremely long expressions that could
exhaust memory or CPU during regex matching.
"""

MAX_PATH_SEGMENTS: Final[int] = 20
"""Maximum allowed path depth (dot-separated segments).

Prevents deep nesting attacks and potential stack overflow during
path traversal. Also limits complexity of binding expressions.
"""

# =============================================================================
# JSON Recursion Depth Limits
# =============================================================================

DEFAULT_JSON_RECURSION_DEPTH: Final[int] = 100
"""Default maximum recursion depth for JSON compatibility validation.

This constant limits how deeply nested structures are validated in
``_is_json_compatible_recursive()``. It prevents stack overflow on
pathological inputs such as deeply nested dicts/lists or cyclic
references that somehow bypass Python's normal recursion limit.

The value of 100 is chosen to:
- Allow normal JSON structures (rarely exceed 10-20 levels)
- Prevent stack overflow on malicious or malformed inputs
- Align with common JSON parser depth limits

.. versionadded:: 0.2.6
"""

MIN_JSON_RECURSION_DEPTH: Final[int] = 10
"""Minimum configurable JSON recursion depth.

Values below this threshold would be too restrictive for practical use,
as even simple nested structures (like user -> address -> country)
can easily reach 5+ levels.

.. versionadded:: 0.2.6
"""

MAX_JSON_RECURSION_DEPTH: Final[int] = 1000
"""Maximum configurable JSON recursion depth.

Values above this threshold increase risk of stack overflow and would
exceed any reasonable JSON structure depth. Most JSON parsers use
similar limits (e.g., Python's json module has implicit limits).

.. versionadded:: 0.2.6
"""

# =============================================================================
# Valid Sources and Context Paths
# =============================================================================

VALID_SOURCES: Final[frozenset[str]] = frozenset({"payload", "envelope", "context"})
"""Valid source names for binding expressions.

Binding expressions must start with one of these sources:
- ``payload``: Access fields from the event payload
- ``envelope``: Access fields from the event envelope (correlation_id, etc.)
- ``context``: Access runtime context values (now_iso, dispatcher_id, etc.)
"""

VALID_CONTEXT_PATHS: Final[frozenset[str]] = frozenset(
    {
        "now_iso",
        "dispatcher_id",
        "correlation_id",
    }
)
"""Exhaustive allowlist of valid context paths.

Context paths are special runtime-provided values injected by the
dispatch infrastructure. Adding new context paths requires:

1. Update this allowlist
2. Update the dispatch context provider to supply the value
3. Document the new context path in relevant docstrings

Current paths:
- ``now_iso``: Current timestamp in ISO 8601 format
- ``dispatcher_id``: Unique identifier of the dispatcher instance
- ``correlation_id``: Request correlation ID for distributed tracing
"""

# =============================================================================
# Expression Pattern
# =============================================================================

EXPRESSION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\$\{([a-z]+)\.([a-zA-Z0-9_.]+)\}$"
)
"""Compiled regex for parsing binding expressions.

Pattern breakdown:
- ``^\\$\\{``: Literal ``${`` at start
- ``([a-z]+)``: Group 1 - source (lowercase letters only)
- ``\\.``: Literal dot separator
- ``([a-zA-Z0-9_.]+)``: Group 2 - path (letters, numbers, underscores, dots)
- ``\\}$``: Literal ``}`` at end

Examples of valid expressions:
- ``${payload.user.id}`` -> source="payload", path="user.id"
- ``${envelope.correlation_id}`` -> source="envelope", path="correlation_id"
- ``${context.now_iso}`` -> source="context", path="now_iso"
"""

__all__: list[str] = [
    "DEFAULT_JSON_RECURSION_DEPTH",
    "EXPRESSION_PATTERN",
    "MAX_EXPRESSION_LENGTH",
    "MAX_JSON_RECURSION_DEPTH",
    "MAX_PATH_SEGMENTS",
    "MIN_JSON_RECURSION_DEPTH",
    "VALID_CONTEXT_PATHS",
    "VALID_SOURCES",
]
