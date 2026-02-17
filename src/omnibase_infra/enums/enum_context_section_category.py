# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Semantic categories for static context sections.

Used by the optional LLM augmentation pass (Pass 2) of the static context
token cost attribution service to classify deterministically parsed sections
into semantic categories.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

from enum import Enum


class EnumContextSectionCategory(str, Enum):
    """Semantic category for a static context section.

    Deterministic parser (Pass 1) assigns ``UNCATEGORIZED``.
    LLM augmentation (Pass 2) reclassifies into one of the semantic values.

    Attributes:
        UNCATEGORIZED: Default category assigned by deterministic parser.
        CONFIG: Configuration and environment variables.
        RULES: Development rules, standards, and invariants.
        TOPOLOGY: Infrastructure topology and network architecture.
        EXAMPLES: Code examples, usage patterns, and snippets.
        COMMANDS: CLI commands, database operations, health checks.
        ARCHITECTURE: System architecture, node types, data flow.
        DOCUMENTATION: Documentation references and guides.
        TESTING: Testing standards, markers, fixtures.
        ERROR_HANDLING: Error hierarchy, patterns, recovery.
    """

    UNCATEGORIZED = "uncategorized"
    CONFIG = "config"
    RULES = "rules"
    TOPOLOGY = "topology"
    EXAMPLES = "examples"
    COMMANDS = "commands"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    ERROR_HANDLING = "error_handling"


__all__ = ["EnumContextSectionCategory"]
