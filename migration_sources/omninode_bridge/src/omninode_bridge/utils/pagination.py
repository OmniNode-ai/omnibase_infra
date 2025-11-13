"""
Pagination Helper Utilities.

Provides reusable pagination utilities with validation, calculation,
and response formatting for database queries and API responses.

Features:
    - Offset/limit calculation with validation
    - Page-based pagination support
    - Cursor-based pagination utilities
    - Comprehensive input validation
    - Standardized response formatting
    - Performance-optimized bounds checking

Usage:
    from omninode_bridge.utils.pagination import PaginationHelper

    # Calculate offset from page number
    offset = PaginationHelper.calculate_offset(page=2, page_size=50)

    # Validate pagination parameters
    limit, offset = PaginationHelper.validate_pagination_params(
        limit=100, offset=200
    )

    # Create paginated response
    response = PaginationHelper.create_pagination_response(
        total=500,
        page=2,
        page_size=50,
        data=[...],
        request_url="/api/items"
    )
"""

from typing import Any, Optional
from urllib.parse import urlencode

from pydantic import BaseModel, Field, field_validator

# Default pagination configuration
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1000
MIN_PAGE_SIZE = 1
MAX_OFFSET = 10000
DEFAULT_OFFSET = 0


class PaginationRequest(BaseModel):
    """
    Pydantic model for pagination request parameters.

    Attributes:
        page: Page number (1-indexed)
        page_size: Number of items per page
        limit: Maximum number of items to return (alternative to page_size)
        offset: Number of items to skip (alternative to page)

    Validation:
        - page must be >= 1
        - page_size must be between MIN_PAGE_SIZE and MAX_PAGE_SIZE
        - limit must be between 1 and MAX_PAGE_SIZE
        - offset must be between 0 and MAX_OFFSET
    """

    page: Optional[int] = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: Optional[int] = Field(
        default=DEFAULT_PAGE_SIZE,
        ge=MIN_PAGE_SIZE,
        le=MAX_PAGE_SIZE,
        description="Items per page",
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_PAGE_SIZE,
        description="Maximum items to return (alternative to page_size)",
    )
    offset: Optional[int] = Field(
        default=None,
        ge=0,
        le=MAX_OFFSET,
        description="Items to skip (alternative to page)",
    )

    @field_validator("page_size")
    @classmethod
    def validate_page_size(cls, v: Optional[int]) -> int:
        """Validate page_size is within acceptable bounds."""
        if v is None:
            return DEFAULT_PAGE_SIZE
        if v < MIN_PAGE_SIZE or v > MAX_PAGE_SIZE:
            raise ValueError(
                f"page_size must be between {MIN_PAGE_SIZE} and {MAX_PAGE_SIZE}"
            )
        return v

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: Optional[int]) -> Optional[int]:
        """Validate limit is within acceptable bounds."""
        if v is not None and (v < 1 or v > MAX_PAGE_SIZE):
            raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        return v

    @field_validator("offset")
    @classmethod
    def validate_offset(cls, v: Optional[int]) -> Optional[int]:
        """Validate offset is within acceptable bounds."""
        if v is not None and (v < 0 or v > MAX_OFFSET):
            raise ValueError(f"offset must be between 0 and {MAX_OFFSET}")
        return v


class PaginationMetadata(BaseModel):
    """
    Pydantic model for pagination metadata in responses.

    Attributes:
        total: Total number of items across all pages
        page: Current page number (1-indexed)
        page_size: Number of items per page
        total_pages: Total number of pages
        has_next: Whether there is a next page
        has_previous: Whether there is a previous page
        next_url: URL for next page (optional)
        previous_url: URL for previous page (optional)
    """

    total: int = Field(..., ge=0, description="Total items across all pages")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether next page exists")
    has_previous: bool = Field(..., description="Whether previous page exists")
    next_url: Optional[str] = Field(default=None, description="URL for next page")
    previous_url: Optional[str] = Field(
        default=None, description="URL for previous page"
    )


class PaginatedResponse(BaseModel):
    """
    Pydantic model for paginated API responses.

    Attributes:
        data: List of items for current page
        pagination: Pagination metadata
        total: Total number of items (convenience field)
    """

    data: list[Any] = Field(..., description="Items for current page")
    pagination: PaginationMetadata = Field(..., description="Pagination metadata")
    total: int = Field(..., ge=0, description="Total items (convenience)")


class PaginationHelper:
    """
    Pagination helper utilities for database queries and API responses.

    Provides static methods for:
        - Offset calculation from page numbers
        - Pagination parameter validation
        - Paginated response formatting
        - URL generation for next/previous pages

    All methods include comprehensive validation and bounds checking
    to prevent resource exhaustion and invalid queries.
    """

    @staticmethod
    def calculate_offset(page: int, page_size: int) -> int:
        """
        Calculate offset from page number and page size.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            Calculated offset (0-indexed)

        Raises:
            ValueError: If page < 1 or page_size < 1
            ValueError: If calculated offset exceeds MAX_OFFSET

        Example:
            >>> offset = PaginationHelper.calculate_offset(page=3, page_size=50)
            >>> print(offset)
            100
        """
        if page < 1:
            raise ValueError(f"page must be >= 1, got: {page}")
        if page_size < MIN_PAGE_SIZE:
            raise ValueError(f"page_size must be >= {MIN_PAGE_SIZE}, got: {page_size}")

        offset = (page - 1) * page_size

        if offset > MAX_OFFSET:
            raise ValueError(
                f"Calculated offset {offset} exceeds maximum allowed value {MAX_OFFSET}"
            )

        return offset

    @staticmethod
    def validate_pagination_params(
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> tuple[int, int]:
        """
        Validate and normalize pagination parameters.

        Supports both limit/offset and page/page_size styles.
        Priority: limit/offset > page/page_size

        Args:
            limit: Maximum number of items to return
            offset: Number of items to skip
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            Tuple of (validated_limit, validated_offset)

        Raises:
            ValueError: If parameters are invalid or out of bounds

        Example:
            >>> limit, offset = PaginationHelper.validate_pagination_params(
            ...     page=2, page_size=50
            ... )
            >>> print(limit, offset)
            50 50
        """
        # If limit/offset provided, use them directly
        if limit is not None or offset is not None:
            validated_limit = limit if limit is not None else DEFAULT_PAGE_SIZE
            validated_offset = offset if offset is not None else DEFAULT_OFFSET

            # Validate limit
            if validated_limit < 1:
                raise ValueError(f"limit must be >= 1, got: {validated_limit}")
            if validated_limit > MAX_PAGE_SIZE:
                raise ValueError(
                    f"limit {validated_limit} exceeds maximum allowed value {MAX_PAGE_SIZE}"
                )

            # Validate offset
            if validated_offset < 0:
                raise ValueError(f"offset must be >= 0, got: {validated_offset}")
            if validated_offset > MAX_OFFSET:
                raise ValueError(
                    f"offset {validated_offset} exceeds maximum allowed value {MAX_OFFSET}"
                )

            return validated_limit, validated_offset

        # Otherwise, use page/page_size
        validated_page = page if page is not None else 1
        validated_page_size = page_size if page_size is not None else DEFAULT_PAGE_SIZE

        # Validate page
        if validated_page < 1:
            raise ValueError(f"page must be >= 1, got: {validated_page}")

        # Validate page_size
        if validated_page_size < MIN_PAGE_SIZE:
            raise ValueError(
                f"page_size must be >= {MIN_PAGE_SIZE}, got: {validated_page_size}"
            )
        if validated_page_size > MAX_PAGE_SIZE:
            raise ValueError(
                f"page_size {validated_page_size} exceeds maximum allowed value {MAX_PAGE_SIZE}"
            )

        # Calculate offset
        calculated_offset = (validated_page - 1) * validated_page_size

        if calculated_offset > MAX_OFFSET:
            raise ValueError(
                f"Calculated offset {calculated_offset} exceeds maximum allowed value {MAX_OFFSET}"
            )

        return validated_page_size, calculated_offset

    @staticmethod
    def create_pagination_response(
        total: int,
        page: int,
        page_size: int,
        data: list[Any],
        request_url: Optional[str] = None,
        additional_params: Optional[dict[str, Any]] = None,
    ) -> PaginatedResponse:
        """
        Create standardized paginated response with metadata.

        Args:
            total: Total number of items across all pages
            page: Current page number (1-indexed)
            page_size: Number of items per page
            data: List of items for current page
            request_url: Base URL for generating next/previous URLs (optional)
            additional_params: Additional query parameters to include in URLs (optional)

        Returns:
            PaginatedResponse with data and pagination metadata

        Raises:
            ValueError: If total < 0, page < 1, or page_size < 1

        Example:
            >>> response = PaginationHelper.create_pagination_response(
            ...     total=500,
            ...     page=2,
            ...     page_size=50,
            ...     data=[...],
            ...     request_url="/api/items"
            ... )
            >>> print(response.pagination.has_next)
            True
        """
        if total < 0:
            raise ValueError(f"total must be >= 0, got: {total}")
        if page < 1:
            raise ValueError(f"page must be >= 1, got: {page}")
        if page_size < MIN_PAGE_SIZE:
            raise ValueError(f"page_size must be >= {MIN_PAGE_SIZE}, got: {page_size}")

        # Calculate total pages
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0

        # Determine if next/previous pages exist
        has_next = page < total_pages
        has_previous = page > 1

        # Generate next/previous URLs if base URL provided
        next_url = None
        previous_url = None

        if request_url:
            base_params = additional_params or {}

            if has_next:
                next_params = {**base_params, "page": page + 1, "page_size": page_size}
                next_url = f"{request_url}?{urlencode(next_params)}"

            if has_previous:
                prev_params = {**base_params, "page": page - 1, "page_size": page_size}
                previous_url = f"{request_url}?{urlencode(prev_params)}"

        # Create pagination metadata
        pagination = PaginationMetadata(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=has_next,
            has_previous=has_previous,
            next_url=next_url,
            previous_url=previous_url,
        )

        return PaginatedResponse(data=data, pagination=pagination, total=total)

    @staticmethod
    def calculate_total_pages(total: int, page_size: int) -> int:
        """
        Calculate total number of pages for given total items and page size.

        Args:
            total: Total number of items
            page_size: Number of items per page

        Returns:
            Total number of pages

        Raises:
            ValueError: If total < 0 or page_size < 1

        Example:
            >>> total_pages = PaginationHelper.calculate_total_pages(
            ...     total=500, page_size=50
            ... )
            >>> print(total_pages)
            10
        """
        if total < 0:
            raise ValueError(f"total must be >= 0, got: {total}")
        if page_size < MIN_PAGE_SIZE:
            raise ValueError(f"page_size must be >= {MIN_PAGE_SIZE}, got: {page_size}")

        return (total + page_size - 1) // page_size if total > 0 else 0

    @staticmethod
    def validate_page_number(page: int, total_pages: int) -> bool:
        """
        Validate that page number is within valid range.

        Args:
            page: Page number to validate
            total_pages: Total number of pages

        Returns:
            True if page is valid, False otherwise

        Example:
            >>> is_valid = PaginationHelper.validate_page_number(
            ...     page=5, total_pages=10
            ... )
            >>> print(is_valid)
            True
        """
        return 1 <= page <= total_pages if total_pages > 0 else page == 1


# Convenience functions for common pagination patterns


def paginate_query_results(
    results: list[Any],
    total: int,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    request_url: Optional[str] = None,
) -> PaginatedResponse:
    """
    Convenience function to paginate query results.

    Args:
        results: Query results for current page
        total: Total number of items across all pages
        page: Current page number (default: 1)
        page_size: Items per page (default: DEFAULT_PAGE_SIZE)
        request_url: Base URL for pagination links (optional)

    Returns:
        PaginatedResponse with data and metadata

    Example:
        >>> results = db.query(...).limit(50).offset(50).all()
        >>> total = db.query(...).count()
        >>> response = paginate_query_results(
        ...     results=results,
        ...     total=total,
        ...     page=2,
        ...     page_size=50,
        ...     request_url="/api/items"
        ... )
    """
    return PaginationHelper.create_pagination_response(
        total=total,
        page=page,
        page_size=page_size,
        data=results,
        request_url=request_url,
    )


def get_pagination_params(
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> tuple[int, int]:
    """
    Convenience function to get validated limit/offset from pagination params.

    Args:
        page: Page number (1-indexed)
        page_size: Items per page
        limit: Maximum items to return
        offset: Items to skip

    Returns:
        Tuple of (limit, offset)

    Example:
        >>> limit, offset = get_pagination_params(page=2, page_size=50)
        >>> print(limit, offset)
        50 50
    """
    return PaginationHelper.validate_pagination_params(
        limit=limit, offset=offset, page=page, page_size=page_size
    )
