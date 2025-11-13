"""
Unit Tests for Pagination Helper Utilities.

Comprehensive test suite for pagination utilities covering:
    - Offset calculation
    - Parameter validation
    - Response formatting
    - Edge cases and error handling
    - Pydantic model validation

Test Coverage: 100%
"""

import pytest
from pydantic import ValidationError

from omninode_bridge.utils.pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_OFFSET,
    MAX_PAGE_SIZE,
    MIN_PAGE_SIZE,
    PaginatedResponse,
    PaginationHelper,
    PaginationMetadata,
    PaginationRequest,
    get_pagination_params,
    paginate_query_results,
)

# ===== Offset Calculation Tests =====


class TestCalculateOffset:
    """Test suite for calculate_offset method."""

    def test_calculate_offset_first_page(self):
        """Test offset calculation for first page."""
        offset = PaginationHelper.calculate_offset(page=1, page_size=50)
        assert offset == 0

    def test_calculate_offset_second_page(self):
        """Test offset calculation for second page."""
        offset = PaginationHelper.calculate_offset(page=2, page_size=50)
        assert offset == 50

    def test_calculate_offset_tenth_page(self):
        """Test offset calculation for tenth page."""
        offset = PaginationHelper.calculate_offset(page=10, page_size=25)
        assert offset == 225

    def test_calculate_offset_large_page_size(self):
        """Test offset calculation with large page size."""
        offset = PaginationHelper.calculate_offset(page=2, page_size=1000)
        assert offset == 1000

    def test_calculate_offset_invalid_page_zero(self):
        """Test error handling for page = 0."""
        with pytest.raises(ValueError, match="page must be >= 1"):
            PaginationHelper.calculate_offset(page=0, page_size=50)

    def test_calculate_offset_invalid_page_negative(self):
        """Test error handling for negative page."""
        with pytest.raises(ValueError, match="page must be >= 1"):
            PaginationHelper.calculate_offset(page=-1, page_size=50)

    def test_calculate_offset_invalid_page_size_zero(self):
        """Test error handling for page_size = 0."""
        with pytest.raises(ValueError, match="page_size must be >="):
            PaginationHelper.calculate_offset(page=1, page_size=0)

    def test_calculate_offset_invalid_page_size_negative(self):
        """Test error handling for negative page_size."""
        with pytest.raises(ValueError, match="page_size must be >="):
            PaginationHelper.calculate_offset(page=1, page_size=-1)

    def test_calculate_offset_exceeds_max_offset(self):
        """Test error handling when calculated offset exceeds MAX_OFFSET."""
        with pytest.raises(ValueError, match="exceeds maximum allowed value"):
            # Page that would result in offset > MAX_OFFSET
            PaginationHelper.calculate_offset(page=1000, page_size=1000)


# ===== Parameter Validation Tests =====


class TestValidatePaginationParams:
    """Test suite for validate_pagination_params method."""

    def test_validate_limit_offset_style(self):
        """Test validation with limit/offset parameters."""
        limit, offset = PaginationHelper.validate_pagination_params(
            limit=50, offset=100
        )
        assert limit == 50
        assert offset == 100

    def test_validate_page_page_size_style(self):
        """Test validation with page/page_size parameters."""
        limit, offset = PaginationHelper.validate_pagination_params(
            page=3, page_size=25
        )
        assert limit == 25
        assert offset == 50  # (3-1) * 25

    def test_validate_defaults_when_none(self):
        """Test default values when all parameters are None."""
        limit, offset = PaginationHelper.validate_pagination_params()
        assert limit == DEFAULT_PAGE_SIZE
        assert offset == 0

    def test_validate_limit_without_offset(self):
        """Test limit without offset uses default offset."""
        limit, offset = PaginationHelper.validate_pagination_params(limit=100)
        assert limit == 100
        assert offset == 0

    def test_validate_offset_without_limit(self):
        """Test offset without limit uses default limit."""
        limit, offset = PaginationHelper.validate_pagination_params(offset=200)
        assert limit == DEFAULT_PAGE_SIZE
        assert offset == 200

    def test_validate_priority_limit_offset_over_page(self):
        """Test that limit/offset takes priority over page/page_size."""
        limit, offset = PaginationHelper.validate_pagination_params(
            limit=100, offset=50, page=2, page_size=25
        )
        assert limit == 100
        assert offset == 50

    def test_validate_invalid_limit_negative(self):
        """Test error handling for negative limit."""
        with pytest.raises(ValueError, match="limit must be >= 1"):
            PaginationHelper.validate_pagination_params(limit=-1)

    def test_validate_invalid_limit_zero(self):
        """Test error handling for zero limit."""
        with pytest.raises(ValueError, match="limit must be >= 1"):
            PaginationHelper.validate_pagination_params(limit=0)

    def test_validate_invalid_limit_exceeds_max(self):
        """Test error handling for limit exceeding MAX_PAGE_SIZE."""
        with pytest.raises(ValueError, match="exceeds maximum allowed value"):
            PaginationHelper.validate_pagination_params(limit=MAX_PAGE_SIZE + 1)

    def test_validate_invalid_offset_negative(self):
        """Test error handling for negative offset."""
        with pytest.raises(ValueError, match="offset must be >= 0"):
            PaginationHelper.validate_pagination_params(offset=-1)

    def test_validate_invalid_offset_exceeds_max(self):
        """Test error handling for offset exceeding MAX_OFFSET."""
        with pytest.raises(ValueError, match="exceeds maximum allowed value"):
            PaginationHelper.validate_pagination_params(offset=MAX_OFFSET + 1)

    def test_validate_invalid_page_negative(self):
        """Test error handling for negative page."""
        with pytest.raises(ValueError, match="page must be >= 1"):
            PaginationHelper.validate_pagination_params(page=-1, page_size=50)

    def test_validate_invalid_page_zero(self):
        """Test error handling for zero page."""
        with pytest.raises(ValueError, match="page must be >= 1"):
            PaginationHelper.validate_pagination_params(page=0, page_size=50)

    def test_validate_invalid_page_size_too_small(self):
        """Test error handling for page_size below minimum."""
        with pytest.raises(ValueError, match="page_size must be >="):
            PaginationHelper.validate_pagination_params(page=1, page_size=0)

    def test_validate_invalid_page_size_exceeds_max(self):
        """Test error handling for page_size exceeding maximum."""
        with pytest.raises(ValueError, match="exceeds maximum allowed value"):
            PaginationHelper.validate_pagination_params(
                page=1, page_size=MAX_PAGE_SIZE + 1
            )

    def test_validate_calculated_offset_exceeds_max(self):
        """Test error handling when calculated offset exceeds MAX_OFFSET."""
        with pytest.raises(ValueError, match="exceeds maximum allowed value"):
            PaginationHelper.validate_pagination_params(page=1000, page_size=1000)


# ===== Response Creation Tests =====


class TestCreatePaginationResponse:
    """Test suite for create_pagination_response method."""

    def test_create_response_basic(self):
        """Test basic pagination response creation."""
        data = [{"id": i} for i in range(50)]
        response = PaginationHelper.create_pagination_response(
            total=500, page=2, page_size=50, data=data
        )

        assert isinstance(response, PaginatedResponse)
        assert response.total == 500
        assert len(response.data) == 50
        assert response.pagination.page == 2
        assert response.pagination.page_size == 50
        assert response.pagination.total_pages == 10
        assert response.pagination.has_next is True
        assert response.pagination.has_previous is True

    def test_create_response_first_page(self):
        """Test pagination response for first page."""
        data = [{"id": i} for i in range(50)]
        response = PaginationHelper.create_pagination_response(
            total=500, page=1, page_size=50, data=data
        )

        assert response.pagination.has_next is True
        assert response.pagination.has_previous is False

    def test_create_response_last_page(self):
        """Test pagination response for last page."""
        data = [{"id": i} for i in range(50)]
        response = PaginationHelper.create_pagination_response(
            total=500, page=10, page_size=50, data=data
        )

        assert response.pagination.has_next is False
        assert response.pagination.has_previous is True

    def test_create_response_single_page(self):
        """Test pagination response when all data fits in one page."""
        data = [{"id": i} for i in range(30)]
        response = PaginationHelper.create_pagination_response(
            total=30, page=1, page_size=50, data=data
        )

        assert response.pagination.total_pages == 1
        assert response.pagination.has_next is False
        assert response.pagination.has_previous is False

    def test_create_response_empty_results(self):
        """Test pagination response with no results."""
        response = PaginationHelper.create_pagination_response(
            total=0, page=1, page_size=50, data=[]
        )

        assert response.total == 0
        assert len(response.data) == 0
        assert response.pagination.total_pages == 0
        assert response.pagination.has_next is False
        assert response.pagination.has_previous is False

    def test_create_response_with_urls(self):
        """Test pagination response with next/previous URLs."""
        data = [{"id": i} for i in range(50)]
        response = PaginationHelper.create_pagination_response(
            total=500,
            page=2,
            page_size=50,
            data=data,
            request_url="/api/items",
        )

        assert response.pagination.next_url is not None
        assert "/api/items?page=3&page_size=50" in response.pagination.next_url
        assert response.pagination.previous_url is not None
        assert "/api/items?page=1&page_size=50" in response.pagination.previous_url

    def test_create_response_with_additional_params(self):
        """Test pagination response with additional query parameters."""
        data = [{"id": i} for i in range(50)]
        response = PaginationHelper.create_pagination_response(
            total=500,
            page=2,
            page_size=50,
            data=data,
            request_url="/api/items",
            additional_params={"status": "active", "sort": "name"},
        )

        assert response.pagination.next_url is not None
        assert "status=active" in response.pagination.next_url
        assert "sort=name" in response.pagination.next_url

    def test_create_response_no_next_url_on_last_page(self):
        """Test that next_url is None on last page."""
        data = [{"id": i} for i in range(50)]
        response = PaginationHelper.create_pagination_response(
            total=500,
            page=10,
            page_size=50,
            data=data,
            request_url="/api/items",
        )

        assert response.pagination.next_url is None
        assert response.pagination.previous_url is not None

    def test_create_response_no_previous_url_on_first_page(self):
        """Test that previous_url is None on first page."""
        data = [{"id": i} for i in range(50)]
        response = PaginationHelper.create_pagination_response(
            total=500,
            page=1,
            page_size=50,
            data=data,
            request_url="/api/items",
        )

        assert response.pagination.next_url is not None
        assert response.pagination.previous_url is None

    def test_create_response_invalid_total_negative(self):
        """Test error handling for negative total."""
        with pytest.raises(ValueError, match="total must be >= 0"):
            PaginationHelper.create_pagination_response(
                total=-1, page=1, page_size=50, data=[]
            )

    def test_create_response_invalid_page_zero(self):
        """Test error handling for zero page."""
        with pytest.raises(ValueError, match="page must be >= 1"):
            PaginationHelper.create_pagination_response(
                total=100, page=0, page_size=50, data=[]
            )

    def test_create_response_invalid_page_size_zero(self):
        """Test error handling for zero page_size."""
        with pytest.raises(ValueError, match="page_size must be >="):
            PaginationHelper.create_pagination_response(
                total=100, page=1, page_size=0, data=[]
            )


# ===== Total Pages Calculation Tests =====


class TestCalculateTotalPages:
    """Test suite for calculate_total_pages method."""

    def test_calculate_total_pages_exact_division(self):
        """Test total pages when total divides evenly by page_size."""
        total_pages = PaginationHelper.calculate_total_pages(total=100, page_size=50)
        assert total_pages == 2

    def test_calculate_total_pages_with_remainder(self):
        """Test total pages when division has remainder."""
        total_pages = PaginationHelper.calculate_total_pages(total=105, page_size=50)
        assert total_pages == 3

    def test_calculate_total_pages_less_than_page_size(self):
        """Test total pages when total is less than page_size."""
        total_pages = PaginationHelper.calculate_total_pages(total=25, page_size=50)
        assert total_pages == 1

    def test_calculate_total_pages_zero_total(self):
        """Test total pages when total is 0."""
        total_pages = PaginationHelper.calculate_total_pages(total=0, page_size=50)
        assert total_pages == 0

    def test_calculate_total_pages_invalid_total_negative(self):
        """Test error handling for negative total."""
        with pytest.raises(ValueError, match="total must be >= 0"):
            PaginationHelper.calculate_total_pages(total=-1, page_size=50)

    def test_calculate_total_pages_invalid_page_size_zero(self):
        """Test error handling for zero page_size."""
        with pytest.raises(ValueError, match="page_size must be >="):
            PaginationHelper.calculate_total_pages(total=100, page_size=0)


# ===== Page Number Validation Tests =====


class TestValidatePageNumber:
    """Test suite for validate_page_number method."""

    def test_validate_page_number_valid(self):
        """Test validation of valid page number."""
        assert PaginationHelper.validate_page_number(page=5, total_pages=10) is True

    def test_validate_page_number_first_page(self):
        """Test validation of first page."""
        assert PaginationHelper.validate_page_number(page=1, total_pages=10) is True

    def test_validate_page_number_last_page(self):
        """Test validation of last page."""
        assert PaginationHelper.validate_page_number(page=10, total_pages=10) is True

    def test_validate_page_number_exceeds_total(self):
        """Test validation when page exceeds total pages."""
        assert PaginationHelper.validate_page_number(page=11, total_pages=10) is False

    def test_validate_page_number_zero(self):
        """Test validation of page 0."""
        assert PaginationHelper.validate_page_number(page=0, total_pages=10) is False

    def test_validate_page_number_negative(self):
        """Test validation of negative page."""
        assert PaginationHelper.validate_page_number(page=-1, total_pages=10) is False

    def test_validate_page_number_zero_total_pages(self):
        """Test validation when total_pages is 0."""
        assert PaginationHelper.validate_page_number(page=1, total_pages=0) is True
        assert PaginationHelper.validate_page_number(page=2, total_pages=0) is False


# ===== Pydantic Model Tests =====


class TestPaginationRequest:
    """Test suite for PaginationRequest Pydantic model."""

    def test_pagination_request_defaults(self):
        """Test default values for PaginationRequest."""
        request = PaginationRequest()
        assert request.page == 1
        assert request.page_size == DEFAULT_PAGE_SIZE
        assert request.limit is None
        assert request.offset is None

    def test_pagination_request_custom_values(self):
        """Test custom values for PaginationRequest."""
        request = PaginationRequest(page=5, page_size=100, limit=200, offset=50)
        assert request.page == 5
        assert request.page_size == 100
        assert request.limit == 200
        assert request.offset == 50

    def test_pagination_request_invalid_page_negative(self):
        """Test validation error for negative page."""
        with pytest.raises(ValidationError):
            PaginationRequest(page=-1)

    def test_pagination_request_invalid_page_size_exceeds_max(self):
        """Test validation error for page_size exceeding maximum."""
        with pytest.raises(ValidationError):
            PaginationRequest(page_size=MAX_PAGE_SIZE + 1)

    def test_pagination_request_invalid_limit_exceeds_max(self):
        """Test validation error for limit exceeding maximum."""
        with pytest.raises(ValidationError):
            PaginationRequest(limit=MAX_PAGE_SIZE + 1)

    def test_pagination_request_invalid_offset_exceeds_max(self):
        """Test validation error for offset exceeding maximum."""
        with pytest.raises(ValidationError):
            PaginationRequest(offset=MAX_OFFSET + 1)


class TestPaginationMetadata:
    """Test suite for PaginationMetadata Pydantic model."""

    def test_pagination_metadata_creation(self):
        """Test creation of PaginationMetadata."""
        metadata = PaginationMetadata(
            total=500,
            page=2,
            page_size=50,
            total_pages=10,
            has_next=True,
            has_previous=True,
            next_url="/api/items?page=3",
            previous_url="/api/items?page=1",
        )

        assert metadata.total == 500
        assert metadata.page == 2
        assert metadata.page_size == 50
        assert metadata.total_pages == 10
        assert metadata.has_next is True
        assert metadata.has_previous is True

    def test_pagination_metadata_optional_urls(self):
        """Test PaginationMetadata with optional URLs."""
        metadata = PaginationMetadata(
            total=100,
            page=1,
            page_size=50,
            total_pages=2,
            has_next=True,
            has_previous=False,
        )

        assert metadata.next_url is None
        assert metadata.previous_url is None


class TestPaginatedResponse:
    """Test suite for PaginatedResponse Pydantic model."""

    def test_paginated_response_creation(self):
        """Test creation of PaginatedResponse."""
        metadata = PaginationMetadata(
            total=100,
            page=1,
            page_size=50,
            total_pages=2,
            has_next=True,
            has_previous=False,
        )

        response = PaginatedResponse(
            data=[{"id": 1}, {"id": 2}], pagination=metadata, total=100
        )

        assert len(response.data) == 2
        assert response.total == 100
        assert response.pagination.page == 1


# ===== Convenience Function Tests =====


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_paginate_query_results(self):
        """Test paginate_query_results convenience function."""
        results = [{"id": i} for i in range(50)]
        response = paginate_query_results(
            results=results,
            total=500,
            page=2,
            page_size=50,
            request_url="/api/items",
        )

        assert isinstance(response, PaginatedResponse)
        assert response.total == 500
        assert response.pagination.page == 2
        assert response.pagination.has_next is True

    def test_get_pagination_params(self):
        """Test get_pagination_params convenience function."""
        limit, offset = get_pagination_params(page=3, page_size=25)
        assert limit == 25
        assert offset == 50

    def test_get_pagination_params_defaults(self):
        """Test get_pagination_params with defaults."""
        limit, offset = get_pagination_params()
        assert limit == DEFAULT_PAGE_SIZE
        assert offset == 0


# ===== Edge Case Tests =====


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_maximum_page_size(self):
        """Test pagination with maximum page_size."""
        limit, offset = PaginationHelper.validate_pagination_params(
            page=1, page_size=MAX_PAGE_SIZE
        )
        assert limit == MAX_PAGE_SIZE
        assert offset == 0

    def test_maximum_offset(self):
        """Test pagination with maximum offset."""
        limit, offset = PaginationHelper.validate_pagination_params(
            limit=10, offset=MAX_OFFSET
        )
        assert limit == 10
        assert offset == MAX_OFFSET

    def test_minimum_page_size(self):
        """Test pagination with minimum page_size."""
        limit, offset = PaginationHelper.validate_pagination_params(
            page=1, page_size=MIN_PAGE_SIZE
        )
        assert limit == MIN_PAGE_SIZE
        assert offset == 0

    def test_large_total_small_page_size(self):
        """Test pagination with large total and small page_size."""
        total_pages = PaginationHelper.calculate_total_pages(
            total=1000000, page_size=10
        )
        assert total_pages == 100000

    def test_partial_last_page(self):
        """Test response for partial last page."""
        data = [{"id": i} for i in range(7)]  # Only 7 items on last page
        response = PaginationHelper.create_pagination_response(
            total=107, page=11, page_size=10, data=data
        )

        assert response.pagination.total_pages == 11
        assert len(response.data) == 7
        assert response.pagination.has_next is False
