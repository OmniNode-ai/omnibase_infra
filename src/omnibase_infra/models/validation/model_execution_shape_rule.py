# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Execution Shape Rule Model.

Defines the validation rules for ONEX node archetype execution shapes.
Each rule specifies what node output types a node archetype is allowed
to return, whether it can publish directly, and other constraints.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from omnibase_infra.enums import EnumNodeArchetype, EnumNodeOutputType


class ModelExecutionShapeRule(BaseModel):
    """Execution shape rule for ONEX node archetype validation.

    Defines the constraints for a specific node archetype in the ONEX
    4-node architecture. These rules are used by the execution shape
    validator to detect violations during static analysis.

    Attributes:
        node_archetype: The node archetype this rule applies to.
        allowed_return_types: Node output types the node CAN return.
        forbidden_return_types: Node output types the node CANNOT return.
        can_publish_directly: Whether node can bypass event bus routing.
        can_access_system_time: Whether node can access non-deterministic time.

    Example:
        >>> from omnibase_infra.enums import EnumNodeArchetype, EnumNodeOutputType
        >>> from omnibase_infra.models.validation import ModelExecutionShapeRule
        >>>
        >>> # Reducer rule: can return projections, cannot return events
        >>> rule = ModelExecutionShapeRule(
        ...     node_archetype=EnumNodeArchetype.REDUCER,
        ...     allowed_return_types=[EnumNodeOutputType.PROJECTION],
        ...     forbidden_return_types=[EnumNodeOutputType.EVENT],
        ...     can_publish_directly=False,
        ...     can_access_system_time=False,
        ... )
        >>>
        >>> # Check if a return type is allowed
        >>> rule.is_return_type_allowed(EnumNodeOutputType.PROJECTION)  # True
        >>> rule.is_return_type_allowed(EnumNodeOutputType.EVENT)  # False
        >>> rule.is_return_type_allowed(EnumNodeOutputType.COMMAND)  # False (not in allowed)

    Note:
        **Interaction between allowed_return_types and forbidden_return_types:**

        The validation logic applies rules in the following priority order:

        1. **Forbidden check (highest priority)**: Output types in
           `forbidden_return_types` are ALWAYS rejected, regardless of
           whether they appear in `allowed_return_types`.

        2. **Allowed check**: If `allowed_return_types` is non-empty,
           the output type must be in that list to be allowed (explicit
           allow-list mode).

        3. **Permissive fallback**: If `allowed_return_types` is empty,
           all non-forbidden output types are implicitly allowed. This mode
           is not typically used in ONEX nodes.

        **Practical usage in ONEX:**

        Most node archetype rules explicitly list their allowed output types for
        clarity and type safety:

        - EFFECT: allowed=[EVENT, COMMAND], forbidden=[PROJECTION]
        - COMPUTE: allowed=[all 4 output types], forbidden=[] (fully permissive)
        - REDUCER: allowed=[PROJECTION], forbidden=[EVENT]
        - ORCHESTRATOR: allowed=[COMMAND, EVENT], forbidden=[INTENT, PROJECTION]
    """

    node_archetype: EnumNodeArchetype = Field(
        ...,
        description="The node archetype this rule applies to",
    )
    allowed_return_types: list[EnumNodeOutputType] = Field(
        default_factory=list,
        description=(
            "Node output types this node archetype is explicitly allowed to return. "
            "If non-empty, acts as an allow-list: only listed types pass validation. "
            "If empty, all non-forbidden types are implicitly allowed (permissive mode). "
            "Used by is_return_type_allowed() method for validation. "
            "Example: REDUCER sets [PROJECTION] to only allow projections, "
            "while COMPUTE sets all 4 types to be fully permissive."
        ),
    )
    forbidden_return_types: list[EnumNodeOutputType] = Field(
        default_factory=list,
        description="Node output types this node archetype is forbidden from returning",
    )
    can_publish_directly: bool = Field(
        default=False,
        description="Whether this node can publish messages directly (bypassing event bus)",
    )
    can_access_system_time: bool = Field(
        default=True,
        description="Whether this node can access system time (non-deterministic)",
    )

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        use_enum_values=False,  # Keep enum objects for type safety
    )

    @field_validator("allowed_return_types", "forbidden_return_types")
    @classmethod
    def validate_no_duplicates(
        cls, v: list[EnumNodeOutputType], info: ValidationInfo
    ) -> list[EnumNodeOutputType]:
        """Ensure no duplicate output types in return type lists."""
        if len(v) != len(set(v)):
            raise ValueError(f"Duplicate output types in {info.field_name}")
        return v

    def is_return_type_allowed(self, output_type: EnumNodeOutputType) -> bool:
        """Check if a node output type is allowed as a return type.

        The validation logic applies the following rules in order:

        1. If the output type is in `forbidden_return_types`, it is always forbidden.
        2. If `allowed_return_types` is non-empty, the output type must be in that list
           to be allowed (explicit allow-list mode).
        3. If `allowed_return_types` is empty, all non-forbidden output types are
           implicitly allowed (permissive mode for COMPUTE nodes).

        Args:
            output_type: The node output type to check.

        Returns:
            True if the output type is allowed, False if forbidden.

        Example:
            >>> # REDUCER: allowed=[PROJECTION], forbidden=[EVENT]
            >>> rule.is_return_type_allowed(EnumNodeOutputType.PROJECTION)  # True
            >>> rule.is_return_type_allowed(EnumNodeOutputType.EVENT)  # False
            >>> rule.is_return_type_allowed(EnumNodeOutputType.COMMAND)  # False (not in allowed)
            >>>
            >>> # COMPUTE: allowed=[all 4 output types], forbidden=[]
            >>> rule.is_return_type_allowed(EnumNodeOutputType.EVENT)  # True
        """
        # Rule 1: Forbidden output types are always rejected
        if output_type in self.forbidden_return_types:
            return False

        # Rule 2: If allowed list is specified, output type must be in it
        if self.allowed_return_types and output_type not in self.allowed_return_types:
            return False

        # Rule 3: All other output types are allowed
        return True


__all__ = ["ModelExecutionShapeRule"]
