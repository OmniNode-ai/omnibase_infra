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
    "EXPRESSION_PATTERN",
    "MAX_EXPRESSION_LENGTH",
    "MAX_PATH_SEGMENTS",
    "VALID_CONTEXT_PATHS",
    "VALID_SOURCES",
]
