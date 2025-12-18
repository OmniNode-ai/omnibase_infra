# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry Effect Node package.

This node provides dual registration capabilities to Consul (service discovery)
and PostgreSQL (persistent state) via the message bus bridge pattern.

Node Type: EFFECT
Purpose: Bridge message bus events to external infrastructure services for
         service discovery and persistent registry state management.
"""

from __future__ import annotations

__all__: list[str] = []
