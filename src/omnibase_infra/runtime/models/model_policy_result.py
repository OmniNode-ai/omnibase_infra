# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pydantic model for policy evaluation result.

This module provides the ModelPolicyResult Pydantic BaseModel for returning
structured results from policy evaluation methods.

Design Notes:
    - Uses ConfigDict(extra="allow") to support arbitrary policy-specific fields
    - Supports dict-like access via __getitem__ for flexible API usage
    - Can be instantiated from dicts using model_validate()
    - Follows ONEX naming convention: Model<Name>

This model replaces the former JsonType return type in ProtocolPolicy.evaluate()
and ProtocolPolicy.decide() method signatures, providing type safety while
maintaining flexibility for diverse policy results.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.mixins import MixinDictLikeAccessors


class ModelPolicyResult(MixinDictLikeAccessors, BaseModel):
    """Base Pydantic model for policy evaluation results.

    This model provides structured results from policy evaluation including
    decisions, computed values, and metadata.

    Common Fields:
        success: Whether the policy evaluation succeeded
        reason: Human-readable explanation of the decision
        metadata: Additional metadata for observability

    Configuration:
        - extra="allow": Accepts arbitrary additional fields for policy-specific results
        - frozen=False: Allows mutation
        - populate_by_name=True: Allows field access by alias

    Example:
        ```python
        # Return from orchestrator policy (retry decision)
        result = ModelPolicyResult(
            should_retry=True,
            delay_seconds=4.0,
            reason="Attempt 2 of 10, exponential backoff",
        )

        # Return from reducer policy (merge decision)
        result = ModelPolicyResult(
            merged_state={"count": 5, "sum": 100},
            conflicts=[],
            success=True,
        )

        # Access with get (dict-like, backwards compatible)
        should_retry = result.get("should_retry", False)

        # Access as attribute
        delay = result.delay_seconds
        ```

    Policy-Specific Fields:
        For orchestrator retry policies:
            - should_retry: Whether to retry the operation
            - delay_seconds: Delay before retry
            - max_retries_reached: Whether max retries exceeded

        For orchestrator routing policies:
            - target_handler: Handler to route to
            - priority: Routing priority
            - fallback_handlers: Alternative handlers if primary fails

        For reducer merge policies:
            - merged_state: Result of state merge
            - conflicts: List of detected conflicts
            - resolution_strategy: How conflicts were resolved

        For reducer projection policies:
            - projection: Projected view of state
            - projection_type: Type of projection applied

    Note:
        All fields are optional due to extra="allow". Specific policies should
        document which result fields they return.
    """

    model_config = ConfigDict(
        extra="allow",
        frozen=False,
        populate_by_name=True,
        from_attributes=True,
    )

    # Common result fields - all optional with defaults
    success: bool = True
    reason: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)


__all__: list[str] = ["ModelPolicyResult"]
