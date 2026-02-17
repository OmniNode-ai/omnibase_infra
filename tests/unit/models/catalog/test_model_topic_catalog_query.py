# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelTopicCatalogQuery.

Tests validation behavior including topic_pattern character validation,
length limits, and default values.

Related Tickets:
    - OMN-2310: Topic Catalog model + suffix foundation
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.models.catalog.model_topic_catalog_query import (
    ModelTopicCatalogQuery,
)


class TestModelTopicCatalogQueryCreation:
    """Test basic creation and defaults."""

    def test_minimal_creation(self) -> None:
        """Test creation with only required fields."""
        cid = uuid4()
        query = ModelTopicCatalogQuery(
            correlation_id=cid,
            client_id="test-client",
        )
        assert query.correlation_id == cid
        assert query.client_id == "test-client"
        assert query.include_inactive is False
        assert query.topic_pattern is None
        assert query.schema_version == 1

    def test_full_creation(self) -> None:
        """Test creation with all fields specified."""
        cid = uuid4()
        query = ModelTopicCatalogQuery(
            correlation_id=cid,
            client_id="node-registration-orchestrator",
            include_inactive=True,
            topic_pattern="onex.evt.platform.*",
            schema_version=2,
        )
        assert query.correlation_id == cid
        assert query.client_id == "node-registration-orchestrator"
        assert query.include_inactive is True
        assert query.topic_pattern == "onex.evt.platform.*"
        assert query.schema_version == 2

    def test_frozen_immutability(self) -> None:
        """Test that the model is frozen (immutable)."""
        query = ModelTopicCatalogQuery(
            correlation_id=uuid4(),
            client_id="test",
        )
        with pytest.raises(ValidationError):
            query.client_id = "modified"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogQuery(
                correlation_id=uuid4(),
                client_id="test",
                unknown_field="value",  # type: ignore[call-arg]
            )


class TestModelTopicCatalogQueryTopicPattern:
    """Test topic_pattern validation."""

    @pytest.mark.parametrize(
        "pattern",
        [
            "onex.evt.platform.*",
            "onex.cmd.*.node-registration.v1",
            "onex.evt.platform.node-registration.v?",
            "simple",
            "a-b_c.d",
            "onex.evt.platform.topic-catalog-query.v1",
        ],
    )
    def test_valid_patterns(self, pattern: str) -> None:
        """Test that valid fnmatch patterns are accepted."""
        query = ModelTopicCatalogQuery(
            correlation_id=uuid4(),
            client_id="test",
            topic_pattern=pattern,
        )
        assert query.topic_pattern == pattern

    @pytest.mark.parametrize(
        "pattern",
        [
            "onex.evt platform",  # space
            "topic@name",  # @
            "topic#name",  # #
            "topic/name",  # /
            "topic[0]",  # brackets
            "topic{name}",  # braces
            "topic$var",  # dollar sign
        ],
    )
    def test_invalid_pattern_characters(self, pattern: str) -> None:
        """Test that patterns with invalid characters are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelTopicCatalogQuery(
                correlation_id=uuid4(),
                client_id="test",
                topic_pattern=pattern,
            )
        assert "invalid characters" in str(exc_info.value).lower()

    def test_pattern_max_length(self) -> None:
        """Test that patterns exceeding 256 chars are rejected."""
        long_pattern = "a" * 257
        with pytest.raises(ValidationError):
            ModelTopicCatalogQuery(
                correlation_id=uuid4(),
                client_id="test",
                topic_pattern=long_pattern,
            )

    def test_pattern_at_max_length(self) -> None:
        """Test that a pattern at exactly 256 chars is accepted."""
        pattern = "a" * 256
        query = ModelTopicCatalogQuery(
            correlation_id=uuid4(),
            client_id="test",
            topic_pattern=pattern,
        )
        assert query.topic_pattern == pattern

    def test_pattern_none_default(self) -> None:
        """Test that topic_pattern defaults to None."""
        query = ModelTopicCatalogQuery(
            correlation_id=uuid4(),
            client_id="test",
        )
        assert query.topic_pattern is None


class TestModelTopicCatalogQueryClientId:
    """Test client_id validation."""

    def test_empty_client_id_rejected(self) -> None:
        """Test that empty client_id is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogQuery(
                correlation_id=uuid4(),
                client_id="",
            )

    def test_max_length_client_id(self) -> None:
        """Test that client_id exceeding 256 chars is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogQuery(
                correlation_id=uuid4(),
                client_id="x" * 257,
            )


class TestModelTopicCatalogQuerySchemaVersion:
    """Test schema_version validation."""

    def test_schema_version_zero_rejected(self) -> None:
        """Test that schema_version=0 is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogQuery(
                correlation_id=uuid4(),
                client_id="test",
                schema_version=0,
            )

    def test_schema_version_negative_rejected(self) -> None:
        """Test that negative schema_version is rejected."""
        with pytest.raises(ValidationError):
            ModelTopicCatalogQuery(
                correlation_id=uuid4(),
                client_id="test",
                schema_version=-1,
            )
