# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""
Dispatch Result Model — canonical definition lives in omnibase_core.

Re-exported from omnibase_core.models.dispatch for backwards-compatible infra imports.
Do NOT define ModelDispatchResult here; the single source of truth is
omnibase_core.models.dispatch.model_dispatch_result.ModelDispatchResult (OMN-12546 S-1b).

The promoted shape uses:
  - dispatcher_id (not handler_id)
  - ModelDispatchOutputs (not list[str])
  - ModelDispatchMetadata (not dict[str, str])
  - EnumCoreErrorCode (not str) for error_code
  - started_at required (no default_factory)
  - output_events, output_intents, projection_intents, dlq_topic
"""

from omnibase_core.models.dispatch.model_dispatch_result import ModelDispatchResult

__all__ = ["ModelDispatchResult"]
