# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Type aliases for Registry Effect Node protocol definitions.

These type aliases define JSON-serializable value types and envelope/result
dictionary structures used by protocol methods.

Type Aliases:
    EnvelopeDict: Dictionary type for operation envelopes
    ResultDict: Dictionary type for operation results

DESIGN NOTE: Using `Any` for JSON container types is a documented ONEX exception.
JSON payloads in envelope/result dictionaries contain arbitrary serializable data
that cannot be precisely typed without schema-specific Pydantic models.

For strongly-typed data exchange, prefer explicit Pydantic models over raw JSON.
These type aliases are intentionally broad to support dynamic envelope patterns
while still providing meaningful type hints for dict containers.
"""

from __future__ import annotations

from typing import Any

# Envelope dictionary types for protocol methods.
# Values use `Any` because envelope payloads contain arbitrary JSON-serializable data.
# The structure is validated at runtime by the envelope executor implementations.
EnvelopeDict = dict[str, Any]
ResultDict = dict[str, Any]

__all__ = [
    "EnvelopeDict",
    "ResultDict",
]
