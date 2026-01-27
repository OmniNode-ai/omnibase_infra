# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic pattern constants for event bus topic validation.

This module provides shared regex patterns for validating event bus topic names.
These patterns are used across multiple modules (e.g., MixinConsulTopicIndex,
MixinNodeIntrospection) to ensure consistent topic validation.

Note:
    This module is intentionally dependency-free (no imports from omnibase_infra)
    to avoid circular import issues. Keep it that way.
"""

from __future__ import annotations

import re
from typing import Final

# Topic name pattern: alphanumeric, underscores, hyphens, and periods only.
# This matches Kafka/Redpanda topic naming conventions and ensures safe
# interpolation into Consul KV paths (prevents path traversal via slashes).
# Pattern: ^[a-zA-Z0-9._-]+$
TOPIC_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9._-]+$")


__all__: list[str] = ["TOPIC_NAME_PATTERN"]
