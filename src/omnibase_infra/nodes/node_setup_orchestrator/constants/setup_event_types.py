# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Event type constants for the setup orchestrator.

Invariant I6: All 16 event type strings are defined once here.

Ticket: OMN-3491
"""

from __future__ import annotations

SETUP_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "setup.preflight.started",
        "setup.preflight.completed",
        "setup.preflight.failed",
        "setup.provision.started",
        "setup.provision.completed",
        "setup.provision.failed",
        "setup.infisical.started",
        "setup.infisical.skipped",
        "setup.infisical.completed",
        "setup.infisical.failed",
        "setup.validate.started",
        "setup.validate.completed",
        "setup.validate.failed",
        "setup.completed",
        "setup.cloud.unavailable",
        "setup.aborted",
    }
)

__all__: list[str] = ["SETUP_EVENT_TYPES"]
