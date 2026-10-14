# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Artifact update trigger types.

Every kind of change the artifact change detector emits. The detector, impact
analyzer, update plan reducer and reconciliation orchestrator all carry the value,
so it is declared here rather than imported from one node into another.

Tracking:
    - OMN-21053: carry the trigger context through the reconciliation chain
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from enum import StrEnum


class EnumUpdateTriggerType(StrEnum):
    """What caused an artifact update to be considered."""

    PR_OPENED = "pr_opened"
    PR_UPDATED = "pr_updated"
    PR_MERGED = "pr_merged"
    CONTRACT_CHANGED = "contract_changed"
    SCHEMA_CHANGED = "schema_changed"
    MANUAL_PLAN_REQUEST = "manual_plan_request"


__all__: list[str] = ["EnumUpdateTriggerType"]
