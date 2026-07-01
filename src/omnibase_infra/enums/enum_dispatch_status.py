# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""
Dispatch Status Enumeration — canonical definition lives in omnibase_core.

Re-exported from omnibase_core.enums for backwards-compatible infra imports.
Do NOT define EnumDispatchStatus here; the single source of truth is
omnibase_core.enums.enum_dispatch_status.EnumDispatchStatus (OMN-12545 S-1a).

The canonical core copy is the superset: it preserves the live infra members
NO_DISPATCHER and INTERNAL_ERROR alongside NO_HANDLER.
"""

from omnibase_core.enums.enum_dispatch_status import EnumDispatchStatus

__all__ = ["EnumDispatchStatus"]
