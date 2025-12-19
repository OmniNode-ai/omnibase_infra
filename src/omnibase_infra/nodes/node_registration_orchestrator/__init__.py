# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Node package.

This node orchestrates the registration workflow by coordinating between
the reducer (for intent generation) and effect node (for execution).

Node Type: ORCHESTRATOR
Purpose: Coordinate node lifecycle registration workflows by consuming
         introspection events, requesting intents from reducer, and
         dispatching execution to the effect node.
"""

from __future__ import annotations

__all__: list[str] = []
