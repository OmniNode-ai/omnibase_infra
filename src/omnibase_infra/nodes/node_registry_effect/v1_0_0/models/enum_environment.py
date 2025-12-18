# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node registration environment enum.

NOTE: This is a temporary local enum for MVP. When omnibase_core >= 0.5.0
is released, use EnumEnvironment from omnibase_core.enums instead.
See Linear ticket OMN-901 for migration tracking.
"""

from __future__ import annotations

from enum import Enum


class EnumEnvironment(str, Enum):
    """Execution environment types for ONEX deployments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"
    INTEGRATION = "integration"
    PREVIEW = "preview"
    SANDBOX = "sandbox"
