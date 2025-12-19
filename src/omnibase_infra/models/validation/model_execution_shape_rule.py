# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Execution Shape Rule Model.

Defines the validation rules for ONEX handler execution shapes.
Each rule specifies what message categories a handler type is allowed
to return, whether it can publish directly, and other constraints.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_handler_type import EnumHandlerType
from omnibase_infra.enums.enum_message_category import EnumMessageCategory


class ModelExecutionShapeRule(BaseModel):
    """Execution shape rule for ONEX handler validation.

    Defines the constraints for a specific handler type in the ONEX
    4-node architecture. These rules are used by the execution shape
    validator to detect violations during static analysis.

    Attributes:
        handler_type: The handler type this rule applies to.
        allowed_return_types: Message categories the handler CAN return.
        forbidden_return_types: Message categories the handler CANNOT return.
        can_publish_directly: Whether handler can bypass event bus routing.
        can_access_system_time: Whether handler can access non-deterministic time.

    Example:
        >>> # Reducer rule: can return projections, cannot return events
        >>> rule = ModelExecutionShapeRule(
        ...     handler_type=EnumHandlerType.REDUCER,
        ...     allowed_return_types=[EnumMessageCategory.PROJECTION],
        ...     forbidden_return_types=[EnumMessageCategory.EVENT],
        ...     can_publish_directly=False,
        ...     can_access_system_time=False,
        ... )

    Note:
        The validation logic uses both `allowed_return_types` and
        `forbidden_return_types`:

        - Categories in `forbidden_return_types` are always rejected.
        - If `allowed_return_types` is non-empty, categories must be in
          that list to be allowed (allow-list mode).
        - If `allowed_return_types` is empty, all non-forbidden categories
          are implicitly allowed (permissive mode).

        This enables strict validation for most handlers while allowing
        COMPUTE handlers to remain fully permissive.
    """

    handler_type: EnumHandlerType = Field(
        ...,
        description="The handler type this rule applies to",
    )
    allowed_return_types: list[EnumMessageCategory] = Field(
        default_factory=list,
        description="Message categories this handler type is explicitly allowed to return",
    )
    forbidden_return_types: list[EnumMessageCategory] = Field(
        default_factory=list,
        description="Message categories this handler type is forbidden from returning",
    )
    can_publish_directly: bool = Field(
        default=False,
        description="Whether this handler can publish messages directly (bypassing event bus)",
    )
    can_access_system_time: bool = Field(
        default=True,
        description="Whether this handler can access system time (non-deterministic)",
    )

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        use_enum_values=False,  # Keep enum objects for type safety
    )

    def is_return_type_allowed(self, category: EnumMessageCategory) -> bool:
        """Check if a message category is allowed as a return type.

        The validation logic applies the following rules in order:

        1. If the category is in `forbidden_return_types`, it is always forbidden.
        2. If `allowed_return_types` is non-empty, the category must be in that list
           to be allowed (explicit allow-list mode).
        3. If `allowed_return_types` is empty, all non-forbidden categories are
           implicitly allowed (permissive mode for COMPUTE handlers).

        Args:
            category: The message category to check.

        Returns:
            True if the category is allowed, False if forbidden.

        Example:
            >>> # REDUCER: allowed=[PROJECTION], forbidden=[EVENT]
            >>> rule.is_return_type_allowed(EnumMessageCategory.PROJECTION)  # True
            >>> rule.is_return_type_allowed(EnumMessageCategory.EVENT)  # False
            >>> rule.is_return_type_allowed(EnumMessageCategory.COMMAND)  # False (not in allowed)
            >>>
            >>> # COMPUTE: allowed=[all 4 categories], forbidden=[]
            >>> rule.is_return_type_allowed(EnumMessageCategory.EVENT)  # True
        """
        # Rule 1: Forbidden categories are always rejected
        if category in self.forbidden_return_types:
            return False

        # Rule 2: If allowed list is specified, category must be in it
        if self.allowed_return_types and category not in self.allowed_return_types:
            return False

        # Rule 3: All other categories are allowed
        return True


__all__ = ["ModelExecutionShapeRule"]
