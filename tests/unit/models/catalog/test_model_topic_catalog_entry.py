# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelTopicCatalogEntry.

Tests computed field enforcement (is_active), frozen immutability,
and edge cases for publisher/subscriber counts.

Related Tickets:
    - OMN-2310: Topic Catalog model + suffix foundation
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.catalog.model_topic_catalog_entry import (
    ModelTopicCatalogEntry,
)


class TestModelTopicCatalogEntryCreation:
    """Test basic creation and defaults."""

    def test_minimal_creation(self) -> None:
        """Test creation with only required fields."""
        entry = ModelTopicCatalogEntry(
            topic_suffix="onex.evt.platform.node-registration.v1",
            topic_name="onex.evt.platform.node-registration.v1",
            partitions=6,
        )
        assert entry.topic_suffix == "onex.evt.platform.node-registration.v1"
        assert entry.topic_name == "onex.evt.platform.node-registration.v1"
        assert entry.partitions == 6
        assert entry.publisher_count == 0
        assert entry.subscriber_count == 0
        assert entry.is_active is False
        assert entry.description == ""
        assert entry.tags == ()

    def test_full_creation(self) -> None:
        """Test creation with all fields specified."""
        entry = ModelTopicCatalogEntry(
            topic_suffix="onex.evt.platform.node-registration.v1",
            topic_name="onex.evt.platform.node-registration.v1",
            description="Node registration events",
            partitions=6,
            publisher_count=2,
            subscriber_count=3,
            tags=("platform", "lifecycle"),
        )
        assert entry.description == "Node registration events"
        assert entry.publisher_count == 2
        assert entry.subscriber_count == 3
        assert entry.is_active is True
        assert entry.tags == ("platform", "lifecycle")

    def test_frozen_immutability(self) -> None:
        """Test that the model is frozen (immutable)."""
        entry = ModelTopicCatalogEntry(
            topic_suffix="onex.evt.platform.test.v1",
            topic_name="onex.evt.platform.test.v1",
            partitions=1,
        )
        with pytest.raises(ValidationError):
            entry.publisher_count = 5  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogEntry(
                topic_suffix="onex.evt.platform.test.v1",
                topic_name="onex.evt.platform.test.v1",
                partitions=1,
                extra_field="value",  # type: ignore[call-arg]
            )


class TestModelTopicCatalogEntryComputedIsActive:
    """Test is_active computed field enforcement (D4)."""

    def test_active_with_publishers_only(self) -> None:
        """Test is_active=True when only publishers exist."""
        entry = ModelTopicCatalogEntry(
            topic_suffix="onex.evt.platform.test.v1",
            topic_name="onex.evt.platform.test.v1",
            partitions=1,
            publisher_count=1,
            subscriber_count=0,
        )
        assert entry.is_active is True

    def test_active_with_subscribers_only(self) -> None:
        """Test is_active=True when only subscribers exist."""
        entry = ModelTopicCatalogEntry(
            topic_suffix="onex.evt.platform.test.v1",
            topic_name="onex.evt.platform.test.v1",
            partitions=1,
            publisher_count=0,
            subscriber_count=1,
        )
        assert entry.is_active is True

    def test_active_with_both(self) -> None:
        """Test is_active=True when both publishers and subscribers exist."""
        entry = ModelTopicCatalogEntry(
            topic_suffix="onex.evt.platform.test.v1",
            topic_name="onex.evt.platform.test.v1",
            partitions=1,
            publisher_count=3,
            subscriber_count=5,
        )
        assert entry.is_active is True

    def test_inactive_with_zero_counts(self) -> None:
        """Test is_active=False when both counts are zero."""
        entry = ModelTopicCatalogEntry(
            topic_suffix="onex.evt.platform.test.v1",
            topic_name="onex.evt.platform.test.v1",
            partitions=1,
            publisher_count=0,
            subscriber_count=0,
        )
        assert entry.is_active is False

    def test_is_active_overridden_to_false(self) -> None:
        """Test that passing is_active=True with zero counts is corrected to False.

        D4: is_active is always computed, never accepted as input.
        """
        entry = ModelTopicCatalogEntry(
            topic_suffix="onex.evt.platform.test.v1",
            topic_name="onex.evt.platform.test.v1",
            partitions=1,
            publisher_count=0,
            subscriber_count=0,
            is_active=True,  # Should be overridden
        )
        assert entry.is_active is False

    def test_is_active_overridden_to_true(self) -> None:
        """Test that passing is_active=False with nonzero counts is corrected to True.

        D4: is_active is always computed, never accepted as input.
        """
        entry = ModelTopicCatalogEntry(
            topic_suffix="onex.evt.platform.test.v1",
            topic_name="onex.evt.platform.test.v1",
            partitions=1,
            publisher_count=1,
            subscriber_count=0,
            is_active=False,  # Should be overridden
        )
        assert entry.is_active is True


class TestModelTopicCatalogEntryValidation:
    """Test field validation."""

    def test_empty_topic_suffix_rejected(self) -> None:
        """Test that empty topic_suffix is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogEntry(
                topic_suffix="",
                topic_name="onex.evt.platform.test.v1",
                partitions=1,
            )

    def test_empty_topic_name_rejected(self) -> None:
        """Test that empty topic_name is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogEntry(
                topic_suffix="onex.evt.platform.test.v1",
                topic_name="",
                partitions=1,
            )

    def test_zero_partitions_rejected(self) -> None:
        """Test that zero partitions is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogEntry(
                topic_suffix="onex.evt.platform.test.v1",
                topic_name="onex.evt.platform.test.v1",
                partitions=0,
            )

    def test_negative_publisher_count_rejected(self) -> None:
        """Test that negative publisher_count is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogEntry(
                topic_suffix="onex.evt.platform.test.v1",
                topic_name="onex.evt.platform.test.v1",
                partitions=1,
                publisher_count=-1,
            )

    def test_negative_subscriber_count_rejected(self) -> None:
        """Test that negative subscriber_count is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogEntry(
                topic_suffix="onex.evt.platform.test.v1",
                topic_name="onex.evt.platform.test.v1",
                partitions=1,
                subscriber_count=-1,
            )


class TestModelTopicCatalogEntryTags:
    """Test tags field behavior."""

    def test_tags_default_empty_tuple(self) -> None:
        """Test that tags default to empty tuple."""
        entry = ModelTopicCatalogEntry(
            topic_suffix="onex.evt.platform.test.v1",
            topic_name="onex.evt.platform.test.v1",
            partitions=1,
        )
        assert entry.tags == ()

    def test_tags_from_list_coerced_to_tuple(self) -> None:
        """Test that list input is coerced to tuple."""
        entry = ModelTopicCatalogEntry(
            topic_suffix="onex.evt.platform.test.v1",
            topic_name="onex.evt.platform.test.v1",
            partitions=1,
            tags=["platform", "lifecycle"],  # type: ignore[arg-type]
        )
        assert entry.tags == ("platform", "lifecycle")
        assert isinstance(entry.tags, tuple)
