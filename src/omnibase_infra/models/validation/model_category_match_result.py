# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Category match result model for message type detection.

This model replaces the tuple pattern ``tuple[bool, MessageOutputCategory | None]``
that was used for category matching operations in the routing coverage validator.
By using a structured model, we provide:

- Clear semantics for match results (matched vs has_category)
- Type-safe access to the matched category
- Factory methods for common use cases

Usage in routing_coverage_validator.py:
    The ``_has_message_decorator`` function uses this model to return
    whether a class has a message type decorator and, if so, which
    category it represents.

Example:
    >>> # Successful match with category
    >>> result = ModelCategoryMatchResult.matched_with_category(
    ...     EnumMessageCategory.EVENT
    ... )
    >>> result.matched
    True
    >>> result.has_category
    True
    >>> result.category
    <EnumMessageCategory.EVENT: 'event'>

    >>> # Match without specific category (generic decorator)
    >>> result = ModelCategoryMatchResult.matched_without_category()
    >>> result.matched
    True
    >>> result.has_category
    False

    >>> # No match
    >>> result = ModelCategoryMatchResult.not_matched()
    >>> result.matched
    False

.. versionadded:: 0.6.1
    Created as part of Union Reduction Phase 2 (OMN-1007).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumMessageCategory, EnumNodeOutputType
from omnibase_infra.types import MessageOutputCategory


class ModelCategoryMatchResult(BaseModel):
    """Result of a category matching operation.

    This model represents the outcome of attempting to match a class name,
    decorator, or base class to a message category (EVENT, COMMAND, INTENT)
    or node output type (PROJECTION).

    The model distinguishes between:
    - **matched**: Whether the matching criteria was satisfied (e.g., decorator found)
    - **category**: The specific category identified, if any

    This distinction is important because some patterns (e.g., a generic
    ``@message_type`` decorator) indicate a match but don't specify the category.

    Attributes:
        matched: Whether the category matching criteria was satisfied.
        category: The matched category (EnumMessageCategory or EnumNodeOutputType),
            or None if no specific category was identified.

    Warning:
        **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to return
        ``True`` only when ``matched`` is True. This differs from typical Pydantic model
        behavior where ``bool(model)`` always returns ``True`` for any valid instance.

        This design enables idiomatic conditional checks for category matching::

            if result:
                # Category was matched - process it
                handle_category(result.category)
            else:
                # No match found - skip processing
                pass

        If you need to check model validity instead, use explicit attribute access::

            # Check for match (uses __bool__)
            if result:
                ...

            # Check model is valid (always True for constructed instance)
            if result is not None:
                ...

            # Explicit match check (preferred for clarity)
            if result.matched:
                ...

    Example:
        >>> # Event category match
        >>> result = ModelCategoryMatchResult.matched_with_category(
        ...     EnumMessageCategory.EVENT
        ... )
        >>> result.matched
        True
        >>> result.category
        <EnumMessageCategory.EVENT: 'event'>

        >>> # Projection (node output type) match
        >>> result = ModelCategoryMatchResult.matched_with_category(
        ...     EnumNodeOutputType.PROJECTION
        ... )
        >>> result.matched
        True
        >>> result.category
        <EnumNodeOutputType.PROJECTION: 'projection'>

        >>> # No match (e.g., class without message decorator)
        >>> result = ModelCategoryMatchResult.not_matched()
        >>> result.matched
        False
        >>> result.has_category
        False

    .. versionadded:: 0.6.1
        Created as part of OMN-1007 tuple-to-model conversion.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    matched: bool = Field(
        description="Whether the category matching criteria was satisfied.",
    )
    category: MessageOutputCategory | None = Field(
        default=None,
        description="The matched category (EnumMessageCategory or EnumNodeOutputType), "
        "or None if no specific category was identified.",
    )

    @property
    def has_category(self) -> bool:
        """Check if a specific category was identified.

        This property should be used to check if a category exists before
        accessing the ``category`` field to ensure meaningful results.

        Returns:
            True if category is not None, False otherwise.

        Example:
            >>> ModelCategoryMatchResult.matched_with_category(
            ...     EnumMessageCategory.EVENT
            ... ).has_category
            True
            >>> ModelCategoryMatchResult.matched_without_category().has_category
            False
            >>> ModelCategoryMatchResult.not_matched().has_category
            False

        .. versionadded:: 0.6.1
        """
        return self.category is not None

    @property
    def is_message_category(self) -> bool:
        """Check if the matched category is a message category (not PROJECTION).

        Message categories (EVENT, COMMAND, INTENT) are routable and can be
        used for message dispatching. PROJECTION is a node output type that
        is not routable.

        Returns:
            True if category is an EnumMessageCategory, False otherwise.

        Example:
            >>> ModelCategoryMatchResult.matched_with_category(
            ...     EnumMessageCategory.EVENT
            ... ).is_message_category
            True
            >>> ModelCategoryMatchResult.matched_with_category(
            ...     EnumNodeOutputType.PROJECTION
            ... ).is_message_category
            False

        .. versionadded:: 0.6.1
        """
        # Single type check is sufficient - EnumMessageCategory is the only message
        # category enum. EnumNodeOutputType contains PROJECTION which is NOT a
        # routable message category (it's for reducer state aggregation outputs).
        return isinstance(self.category, EnumMessageCategory)

    @property
    def is_projection(self) -> bool:
        """Check if the matched category is PROJECTION.

        PROJECTION is a node output type used for reducer state aggregation
        outputs. It is not a routable message category.

        Returns:
            True if category is EnumNodeOutputType.PROJECTION, False otherwise.

        Example:
            >>> ModelCategoryMatchResult.matched_with_category(
            ...     EnumNodeOutputType.PROJECTION
            ... ).is_projection
            True
            >>> ModelCategoryMatchResult.matched_with_category(
            ...     EnumMessageCategory.EVENT
            ... ).is_projection
            False

        .. versionadded:: 0.6.1
        """
        return self.category == EnumNodeOutputType.PROJECTION

    @classmethod
    def matched_with_category(
        cls, category: MessageOutputCategory
    ) -> ModelCategoryMatchResult:
        """Create a successful match result with a specific category.

        Use this factory when the matching operation found a pattern that
        indicates a specific message category or node output type.

        Args:
            category: The matched category (EnumMessageCategory or EnumNodeOutputType).

        Returns:
            ModelCategoryMatchResult with matched=True and the specified category.

        Example:
            >>> result = ModelCategoryMatchResult.matched_with_category(
            ...     EnumMessageCategory.COMMAND
            ... )
            >>> result.matched
            True
            >>> result.category
            <EnumMessageCategory.COMMAND: 'command'>

        .. versionadded:: 0.6.1
        """
        return cls(matched=True, category=category)

    @classmethod
    def matched_without_category(cls) -> ModelCategoryMatchResult:
        """Create a successful match result without a specific category.

        Use this factory when the matching operation found a pattern that
        indicates a message type, but the specific category cannot be
        determined. For example, a generic ``@message_type`` decorator
        without category-specific information.

        Returns:
            ModelCategoryMatchResult with matched=True but category=None.

        Example:
            >>> result = ModelCategoryMatchResult.matched_without_category()
            >>> result.matched
            True
            >>> result.has_category
            False

        .. versionadded:: 0.6.1
        """
        return cls(matched=True, category=None)

    @classmethod
    def not_matched(cls) -> ModelCategoryMatchResult:
        """Create a result indicating no match was found.

        Use this factory when the matching operation did not find any
        pattern indicating a message type.

        Returns:
            ModelCategoryMatchResult with matched=False and category=None.

        Example:
            >>> result = ModelCategoryMatchResult.not_matched()
            >>> result.matched
            False
            >>> result.has_category
            False

        .. versionadded:: 0.6.1
        """
        return cls(matched=False, category=None)

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Returns True if the matching operation was successful (matched=True),
        regardless of whether a specific category was identified.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when ``matched`` is True. This differs from typical
            Pydantic model behavior where ``bool(model)`` always returns ``True`` for
            any valid model instance.

            This design enables idiomatic conditional checks for match results::

                if result:
                    # Category was matched - process it
                    handle_category(result.category)
                else:
                    # No match found - skip processing
                    pass

            If you need to check model validity instead, use explicit attribute access::

                # Check for match (uses __bool__)
                if result:
                    ...

                # Check model is valid (always True for constructed instance)
                if result is not None:
                    ...

                # Explicit match check (preferred for clarity)
                if result.matched:
                    ...

        Returns:
            True if matched, False otherwise.

        Example:
            >>> if ModelCategoryMatchResult.matched_with_category(
            ...     EnumMessageCategory.EVENT
            ... ):
            ...     print("Found!")
            Found!

            >>> if not ModelCategoryMatchResult.not_matched():
            ...     print("Not found")
            Not found

        .. versionadded:: 0.6.1
        """
        return self.matched
