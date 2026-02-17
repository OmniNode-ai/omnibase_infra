# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelTopicCatalogResponse.

Tests creation, defaults, nested entry handling, and warnings tuple.

Related Tickets:
    - OMN-2310: Topic Catalog model + suffix foundation
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.models.catalog.model_topic_catalog_entry import (
    ModelTopicCatalogEntry,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)


def _make_entry(
    suffix: str = "onex.evt.platform.test.v1",
    partitions: int = 1,
    publisher_count: int = 0,
    subscriber_count: int = 0,
) -> ModelTopicCatalogEntry:
    """Create a test catalog entry."""
    return ModelTopicCatalogEntry(
        topic_suffix=suffix,
        topic_name=suffix,
        partitions=partitions,
        publisher_count=publisher_count,
        subscriber_count=subscriber_count,
    )


class TestModelTopicCatalogResponseCreation:
    """Test basic creation and defaults."""

    def test_minimal_creation(self) -> None:
        """Test creation with only required fields."""
        cid = uuid4()
        now = datetime.now(UTC)
        response = ModelTopicCatalogResponse(
            correlation_id=cid,
            catalog_version=1,
            node_count=5,
            generated_at=now,
        )
        assert response.correlation_id == cid
        assert response.topics == ()
        assert response.catalog_version == 1
        assert response.node_count == 5
        assert response.generated_at == now
        assert response.warnings == ()
        assert response.schema_version == 1

    def test_full_creation_with_entries(self) -> None:
        """Test creation with topic entries and warnings."""
        entry1 = _make_entry(
            suffix="onex.evt.platform.node-registration.v1",
            partitions=6,
            publisher_count=2,
        )
        entry2 = _make_entry(
            suffix="onex.cmd.platform.request-introspection.v1",
            partitions=6,
            subscriber_count=1,
        )
        response = ModelTopicCatalogResponse(
            correlation_id=uuid4(),
            topics=(entry1, entry2),
            catalog_version=3,
            node_count=10,
            generated_at=datetime.now(UTC),
            warnings=("Partial metadata for topic X",),
        )
        assert len(response.topics) == 2
        assert response.topics[0].is_active is True
        assert response.topics[1].is_active is True
        assert response.warnings == ("Partial metadata for topic X",)

    def test_frozen_immutability(self) -> None:
        """Test that the model is frozen (immutable)."""
        response = ModelTopicCatalogResponse(
            correlation_id=uuid4(),
            catalog_version=1,
            node_count=0,
            generated_at=datetime.now(UTC),
        )
        with pytest.raises(ValidationError):
            response.catalog_version = 2  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogResponse(
                correlation_id=uuid4(),
                catalog_version=1,
                node_count=0,
                generated_at=datetime.now(UTC),
                unknown="value",  # type: ignore[call-arg]
            )


class TestModelTopicCatalogResponseValidation:
    """Test field validation."""

    def test_negative_catalog_version_rejected(self) -> None:
        """Test that negative catalog_version is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogResponse(
                correlation_id=uuid4(),
                catalog_version=-1,
                node_count=0,
                generated_at=datetime.now(UTC),
            )

    def test_zero_catalog_version_accepted(self) -> None:
        """Test that catalog_version=0 is accepted (initial state)."""
        response = ModelTopicCatalogResponse(
            correlation_id=uuid4(),
            catalog_version=0,
            node_count=0,
            generated_at=datetime.now(UTC),
        )
        assert response.catalog_version == 0

    def test_negative_node_count_rejected(self) -> None:
        """Test that negative node_count is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogResponse(
                correlation_id=uuid4(),
                catalog_version=1,
                node_count=-1,
                generated_at=datetime.now(UTC),
            )

    def test_schema_version_zero_rejected(self) -> None:
        """Test that schema_version=0 is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogResponse(
                correlation_id=uuid4(),
                catalog_version=1,
                node_count=0,
                generated_at=datetime.now(UTC),
                schema_version=0,
            )


class TestModelTopicCatalogResponseWarnings:
    """Test warnings tuple behavior."""

    def test_warnings_default_empty(self) -> None:
        """Test that warnings default to empty tuple."""
        response = ModelTopicCatalogResponse(
            correlation_id=uuid4(),
            catalog_version=1,
            node_count=0,
            generated_at=datetime.now(UTC),
        )
        assert response.warnings == ()

    def test_warnings_from_list_coerced(self) -> None:
        """Test that list input is coerced to tuple."""
        response = ModelTopicCatalogResponse(
            correlation_id=uuid4(),
            catalog_version=1,
            node_count=0,
            generated_at=datetime.now(UTC),
            warnings=["warning1", "warning2"],
        )
        assert response.warnings == ("warning1", "warning2")
        assert isinstance(response.warnings, tuple)

    def test_empty_response_no_topics(self) -> None:
        """Test response with zero topics is valid."""
        response = ModelTopicCatalogResponse(
            correlation_id=uuid4(),
            catalog_version=0,
            node_count=0,
            generated_at=datetime.now(UTC),
        )
        assert response.topics == ()
        assert len(response.topics) == 0
