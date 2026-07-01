# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""
Dispatch Metadata Model — canonical definition lives in omnibase_core.

Re-exported from omnibase_core.models.dispatch for backwards-compatible infra imports.
Do NOT define ModelDispatchMetadata here; the single source of truth is
omnibase_core.models.dispatch.model_dispatch_metadata.ModelDispatchMetadata (OMN-12546 S-1b).
"""

from omnibase_core.models.dispatch.model_dispatch_metadata import ModelDispatchMetadata

__all__ = ["ModelDispatchMetadata"]
