"""Test suite for PostgreSQL query result models."""

import pytest
from pydantic import ValidationError

from omnibase_infra.models.infrastructure.postgres.model_postgres_query_result import (
    ModelPostgresQueryResult,
)
from omnibase_infra.models.infrastructure.postgres.model_postgres_query_row import (
    ModelPostgresQueryRow,
)
from omnibase_infra.models.infrastructure.postgres.model_postgres_query_row_value import (
    ModelPostgresQueryRowValue,
)


class TestModelPostgresQueryRowValue:
    """Test cases for PostgreSQL query row value model."""

    def test_create_valid_row_value(self):
        """Test creating a valid row value."""
        row_value = ModelPostgresQueryRowValue(
            column_name="user_id",
            value=123,
            column_type="integer"
        )

        assert row_value.column_name == "user_id"
        assert row_value.value == 123
        assert row_value.column_type == "integer"

    def test_row_value_with_none_value(self):
        """Test row value with None value."""
        row_value = ModelPostgresQueryRowValue(
            column_name="optional_field",
            value=None,
            column_type="varchar"
        )

        assert row_value.value is None

    @pytest.mark.parametrize("value,column_type", [
        ("test_string", "varchar"),
        (42, "integer"),
        (3.14, "numeric"),
        (True, "boolean"),
        (False, "boolean"),
        (None, "varchar"),
    ])
    def test_supported_value_types(self, value, column_type):
        """Test that all supported value types work correctly."""
        row_value = ModelPostgresQueryRowValue(
            column_name="test_column",
            value=value,
            column_type=column_type
        )

        assert row_value.value == value


class TestModelPostgresQueryRow:
    """Test cases for PostgreSQL query row model."""

    def test_create_empty_row(self):
        """Test creating an empty row."""
        row = ModelPostgresQueryRow()
        assert row.values == {}

    def test_create_row_with_values(self):
        """Test creating a row with values."""
        test_values = {
            "id": 1,
            "name": "Test User",
            "active": True,
            "score": 95.5,
            "notes": None
        }

        row = ModelPostgresQueryRow(values=test_values)
        assert row.values == test_values

    def test_row_values_type_validation(self):
        """Test that row values are properly typed."""
        row = ModelPostgresQueryRow(values={
            "string_col": "text",
            "int_col": 42,
            "float_col": 3.14,
            "bool_col": True,
            "null_col": None
        })

        assert isinstance(row.values["string_col"], str)
        assert isinstance(row.values["int_col"], int)
        assert isinstance(row.values["float_col"], float)
        assert isinstance(row.values["bool_col"], bool)
        assert row.values["null_col"] is None


class TestModelPostgresQueryResult:
    """Test cases for PostgreSQL query result model."""

    def test_create_empty_result(self):
        """Test creating an empty query result."""
        result = ModelPostgresQueryResult(
            row_count=0
        )

        assert result.rows == []
        assert result.column_names == []
        assert result.row_count == 0
        assert result.has_more is False

    def test_create_result_with_rows(self):
        """Test creating a result with actual data."""
        rows = [
            ModelPostgresQueryRow(values={"id": 1, "name": "User 1"}),
            ModelPostgresQueryRow(values={"id": 2, "name": "User 2"})
        ]
        column_names = ["id", "name"]

        result = ModelPostgresQueryResult(
            rows=rows,
            column_names=column_names,
            row_count=2,
            has_more=True
        )

        assert len(result.rows) == 2
        assert result.column_names == column_names
        assert result.row_count == 2
        assert result.has_more is True

    def test_row_count_validation(self):
        """Test that row_count must be non-negative."""
        with pytest.raises(ValidationError):
            ModelPostgresQueryResult(row_count=-1)

    def test_pagination_support(self):
        """Test pagination functionality."""
        # Test first page
        result_page1 = ModelPostgresQueryResult(
            rows=[ModelPostgresQueryRow(values={"id": i}) for i in range(10)],
            column_names=["id"],
            row_count=10,
            has_more=True
        )

        assert len(result_page1.rows) == 10
        assert result_page1.has_more is True

        # Test last page
        result_page2 = ModelPostgresQueryResult(
            rows=[ModelPostgresQueryRow(values={"id": i}) for i in range(5)],
            column_names=["id"],
            row_count=5,
            has_more=False
        )

        assert len(result_page2.rows) == 5
        assert result_page2.has_more is False

    def test_model_integration(self):
        """Test that all three models work together correctly."""
        # Create row values
        row_values = {
            "user_id": 123,
            "username": "testuser",
            "is_active": True,
            "balance": 1500.50,
            "last_login": None
        }

        # Create row
        row = ModelPostgresQueryRow(values=row_values)

        # Create result
        result = ModelPostgresQueryResult(
            rows=[row],
            column_names=list(row_values.keys()),
            row_count=1,
            has_more=False
        )

        # Verify integration
        assert len(result.rows) == 1
        assert result.rows[0].values == row_values
        assert result.column_names == ["user_id", "username", "is_active", "balance", "last_login"]