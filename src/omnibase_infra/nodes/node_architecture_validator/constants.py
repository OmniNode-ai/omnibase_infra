# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Constants for the architecture validator node.

This module defines constants used by the architecture validator node and its
handlers. These constants are extracted from the node to enable handler-based
architecture while keeping the supported rules defined in one place.

Related:
    - OMN-1138: Architecture Validator for omnibase_infra
    - OMN-1726: Refactor to declarative pattern

.. versionadded:: 0.8.0
    Created as part of OMN-1138 Architecture Validator implementation.

.. versionchanged:: 0.9.0
    Extracted to separate module as part of OMN-1726 declarative refactoring.
"""

from __future__ import annotations

# Supported rule IDs from contract_architecture_validator.yaml
# These are the only rules that this validator node is designed to handle.
# Any rule not in this set indicates a misconfiguration or version mismatch.
SUPPORTED_RULE_IDS: frozenset[str] = frozenset(
    {
        "NO_HANDLER_PUBLISHING",
        "PURE_REDUCERS",
        "NO_FSM_IN_ORCHESTRATORS",
        "NO_WORKFLOW_IN_REDUCERS",
        "NO_DIRECT_HANDLER_DISPATCH",
        "NO_LOCAL_ONLY_PATHS",
    }
)

__all__ = ["SUPPORTED_RULE_IDS"]
