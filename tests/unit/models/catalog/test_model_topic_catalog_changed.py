# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelTopicCatalogChanged.

Tests sorted delta tuples (D7), creation, defaults, and validation.

Related Tickets:
    - OMN-2310: Topic Catalog model + suffix foundation
"""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.models.catalog.model_topic_catalog_changed import (
    ModelTopicCatalogChanged,
)


class TestModelTopicCatalogChangedCreation:
    """Test basic creation and defaults."""

    def test_minimal_creation(self) -> None:
        """Test creation with only required fields."""
        cid = uuid4()
        now = datetime.now(UTC)
        changed = ModelTopicCatalogChanged(
            correlation_id=cid,
            catalog_version=1,
            changed_at=now,
        )
        assert changed.correlation_id == cid
        assert changed.catalog_version == 1
        assert changed.topics_added == ()
        assert changed.topics_removed == ()
        assert changed.trigger_node_id is None
        assert changed.trigger_reason == ""
        assert changed.changed_at == now
        assert changed.schema_version == 1

    def test_full_creation(self) -> None:
        """Test creation with all fields specified."""
        changed = ModelTopicCatalogChanged(
            correlation_id=uuid4(),
            catalog_version=5,
            topics_added=("onex.evt.platform.new-topic.v1",),
            topics_removed=("onex.evt.platform.old-topic.v1",),
            trigger_node_id="node-abc-123",
            trigger_reason="Node registered with new topic",
            changed_at=datetime.now(UTC),
            schema_version=2,
        )
        assert len(changed.topics_added) == 1
        assert len(changed.topics_removed) == 1
        assert changed.trigger_node_id == "node-abc-123"
        assert changed.trigger_reason == "Node registered with new topic"

    def test_frozen_immutability(self) -> None:
        """Test that the model is frozen (immutable)."""
        changed = ModelTopicCatalogChanged(
            correlation_id=uuid4(),
            catalog_version=1,
            changed_at=datetime.now(UTC),
        )
        with pytest.raises(ValidationError):
            changed.catalog_version = 2  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogChanged(
                correlation_id=uuid4(),
                catalog_version=1,
                changed_at=datetime.now(UTC),
                unknown="value",  # type: ignore[call-arg]
            )


class TestModelTopicCatalogChangedSortedDeltas:
    """Test alphabetical sorting of delta tuples (D7)."""

    def test_topics_added_sorted(self) -> None:
        """Test that topics_added is sorted alphabetically."""
        changed = ModelTopicCatalogChanged(
            correlation_id=uuid4(),
            catalog_version=1,
            topics_added=(
                "onex.evt.platform.zebra.v1",
                "onex.evt.platform.alpha.v1",
                "onex.evt.platform.middle.v1",
            ),
            changed_at=datetime.now(UTC),
        )
        assert changed.topics_added == (
            "onex.evt.platform.alpha.v1",
            "onex.evt.platform.middle.v1",
            "onex.evt.platform.zebra.v1",
        )

    def test_topics_removed_sorted(self) -> None:
        """Test that topics_removed is sorted alphabetically."""
        changed = ModelTopicCatalogChanged(
            correlation_id=uuid4(),
            catalog_version=1,
            topics_removed=(
                "onex.evt.platform.zeta.v1",
                "onex.evt.platform.beta.v1",
            ),
            changed_at=datetime.now(UTC),
        )
        assert changed.topics_removed == (
            "onex.evt.platform.beta.v1",
            "onex.evt.platform.zeta.v1",
        )

    def test_already_sorted_unchanged(self) -> None:
        """Test that already-sorted tuples remain unchanged."""
        added = (
            "onex.evt.platform.alpha.v1",
            "onex.evt.platform.beta.v1",
        )
        changed = ModelTopicCatalogChanged(
            correlation_id=uuid4(),
            catalog_version=1,
            topics_added=added,
            changed_at=datetime.now(UTC),
        )
        assert changed.topics_added == added

    def test_empty_deltas_remain_empty(self) -> None:
        """Test that empty delta tuples remain empty after sorting."""
        changed = ModelTopicCatalogChanged(
            correlation_id=uuid4(),
            catalog_version=1,
            changed_at=datetime.now(UTC),
        )
        assert changed.topics_added == ()
        assert changed.topics_removed == ()

    def test_single_element_tuple_sorted(self) -> None:
        """Test that single-element tuples pass sorting without error."""
        changed = ModelTopicCatalogChanged(
            correlation_id=uuid4(),
            catalog_version=1,
            topics_added=("onex.evt.platform.only-one.v1",),
            changed_at=datetime.now(UTC),
        )
        assert changed.topics_added == ("onex.evt.platform.only-one.v1",)

    def test_sorting_is_deterministic(self) -> None:
        """Test that sorting produces identical results across invocations."""
        cid = uuid4()
        now = datetime.now(UTC)
        changed1 = ModelTopicCatalogChanged(
            correlation_id=cid,
            catalog_version=1,
            topics_added=(
                "onex.evt.platform.c.v1",
                "onex.evt.platform.a.v1",
                "onex.evt.platform.b.v1",
            ),
            changed_at=now,
        )
        changed2 = ModelTopicCatalogChanged(
            correlation_id=cid,
            catalog_version=1,
            topics_added=(
                "onex.evt.platform.c.v1",
                "onex.evt.platform.a.v1",
                "onex.evt.platform.b.v1",
            ),
            changed_at=now,
        )
        assert changed1.topics_added == changed2.topics_added


class TestModelTopicCatalogChangedValidation:
    """Test field validation."""

    def test_negative_catalog_version_rejected(self) -> None:
        """Test that negative catalog_version is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogChanged(
                correlation_id=uuid4(),
                catalog_version=-1,
                changed_at=datetime.now(UTC),
            )

    def test_zero_catalog_version_accepted(self) -> None:
        """Test that catalog_version=0 is accepted (initial state)."""
        changed = ModelTopicCatalogChanged(
            correlation_id=uuid4(),
            catalog_version=0,
            changed_at=datetime.now(UTC),
        )
        assert changed.catalog_version == 0

    def test_schema_version_zero_rejected(self) -> None:
        """Test that schema_version=0 is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogChanged(
                correlation_id=uuid4(),
                catalog_version=1,
                changed_at=datetime.now(UTC),
                schema_version=0,
            )

    def test_trigger_node_id_max_length(self) -> None:
        """Test that trigger_node_id exceeding 256 chars is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogChanged(
                correlation_id=uuid4(),
                catalog_version=1,
                changed_at=datetime.now(UTC),
                trigger_node_id="x" * 257,
            )

    def test_trigger_reason_max_length(self) -> None:
        """Test that trigger_reason exceeding 1024 chars is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogChanged(
                correlation_id=uuid4(),
                catalog_version=1,
                changed_at=datetime.now(UTC),
                trigger_reason="x" * 1025,
            )
