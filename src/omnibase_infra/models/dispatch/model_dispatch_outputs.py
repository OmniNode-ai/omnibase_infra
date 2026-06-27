# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""
Dispatch Outputs Model — canonical definition lives in omnibase_core.

Re-exported from omnibase_core.models.dispatch for backwards-compatible infra imports.
Do NOT define ModelDispatchOutputs here; the single source of truth is
omnibase_core.models.dispatch.model_dispatch_outputs.ModelDispatchOutputs (OMN-12546 S-1b).
"""

from omnibase_core.models.dispatch.model_dispatch_outputs import ModelDispatchOutputs

__all__ = ["ModelDispatchOutputs"]
