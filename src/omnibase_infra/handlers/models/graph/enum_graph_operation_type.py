# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Graph operation type enumeration for discriminated unions."""

from enum import Enum


class EnumGraphOperationType(str, Enum):
    """Graph operation types for handler routing.

    Attributes:
        QUERY: Execute a Cypher query that returns results
        EXECUTE: Execute a Cypher statement (write operation)
    """

    QUERY = "graph.query"
    EXECUTE = "graph.execute"


__all__: list[str] = ["EnumGraphOperationType"]
