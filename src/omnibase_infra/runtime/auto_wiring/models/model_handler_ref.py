# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler class reference model — canonical definition lives in omnibase_core.

Re-exported from omnibase_core.models.dispatch for backwards-compatible infra imports.
Do NOT define ModelHandlerRef here; the single source of truth is
omnibase_core.models.dispatch.model_handler_ref.ModelHandlerRef (OMN-12546 S-1b).
"""

from omnibase_core.models.dispatch.model_handler_ref import ModelHandlerRef

__all__ = ["ModelHandlerRef"]
