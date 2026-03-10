# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for ServiceTopicCatalog.

Tests cover the stub behaviour introduced in OMN-3540 (Consul removed):
    - build_catalog: always returns empty topics + CONSUL_UNAVAILABLE
    - get_catalog_version: always returns -1
    - increment_version: always returns -1

Related Tickets:
    - OMN-3540: Remove Consul from omnibase_infra
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.models.catalog.catalog_warning_codes import (
    CONSUL_UNAVAILABLE,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.services.service_topic_catalog import ServiceTopicCatalog
from omnibase_infra.topics.topic_resolver import TopicResolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_SUFFIX = "onex.evt.platform.node-registration.v1"


def _make_service(
    topic_resolver: TopicResolver | None = None,
) -> ServiceTopicCatalog:
    """Create a ServiceTopicCatalog with a mock container."""
    container = MagicMock()
    return ServiceTopicCatalog(
        container=container,
        topic_resolver=topic_resolver,
    )


# ---------------------------------------------------------------------------
# Test: build_catalog stub behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildCatalogStub:
    """Tests for build_catalog stub (Consul removed in OMN-3540)."""

    @pytest.mark.asyncio
    async def test_returns_empty_topics(self) -> None:
        """build_catalog should always return an empty topics tuple."""
        service = _make_service()
        response = await service.build_catalog(correlation_id=uuid4())

        assert isinstance(response, ModelTopicCatalogResponse)
        assert response.topics == ()

    @pytest.mark.asyncio
    async def test_returns_consul_unavailable_warning(self) -> None:
        """build_catalog should always emit CONSUL_UNAVAILABLE warning."""
        service = _make_service()
        response = await service.build_catalog(correlation_id=uuid4())

        assert CONSUL_UNAVAILABLE in response.warnings

    @pytest.mark.asyncio
    async def test_catalog_version_is_zero(self) -> None:
        """build_catalog should return catalog_version == 0."""
        service = _make_service()
        response = await service.build_catalog(correlation_id=uuid4())

        assert response.catalog_version == 0

    @pytest.mark.asyncio
    async def test_include_inactive_ignored(self) -> None:
        """include_inactive flag has no effect on the stub result."""
        service = _make_service()
        response = await service.build_catalog(
            correlation_id=uuid4(),
            include_inactive=True,
        )

        assert response.topics == ()
        assert CONSUL_UNAVAILABLE in response.warnings

    @pytest.mark.asyncio
    async def test_topic_pattern_ignored(self) -> None:
        """topic_pattern filter has no effect on the stub result."""
        service = _make_service()
        response = await service.build_catalog(
            correlation_id=uuid4(),
            topic_pattern="onex.*",
        )

        assert response.topics == ()
        assert CONSUL_UNAVAILABLE in response.warnings

    @pytest.mark.asyncio
    async def test_multiple_calls_all_return_empty(self) -> None:
        """Every call to build_catalog returns the same stub result."""
        service = _make_service()
        for _ in range(3):
            response = await service.build_catalog(correlation_id=uuid4())
            assert response.topics == ()
            assert CONSUL_UNAVAILABLE in response.warnings


# ---------------------------------------------------------------------------
# Test: get_catalog_version stub behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetCatalogVersionStub:
    """Tests for get_catalog_version stub (Consul removed in OMN-3540)."""

    @pytest.mark.asyncio
    async def test_always_returns_minus_one(self) -> None:
        """get_catalog_version should always return -1."""
        service = _make_service()
        version = await service.get_catalog_version(uuid4())
        assert version == -1

    @pytest.mark.asyncio
    async def test_returns_minus_one_on_repeated_calls(self) -> None:
        """get_catalog_version should return -1 on every call."""
        service = _make_service()
        for _ in range(3):
            assert await service.get_catalog_version(uuid4()) == -1


# ---------------------------------------------------------------------------
# Test: increment_version stub behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIncrementVersionStub:
    """Tests for increment_version stub (Consul removed in OMN-3540)."""

    @pytest.mark.asyncio
    async def test_always_returns_minus_one(self) -> None:
        """increment_version should always return -1."""
        service = _make_service()
        result = await service.increment_version(uuid4())
        assert result == -1

    @pytest.mark.asyncio
    async def test_returns_minus_one_on_repeated_calls(self) -> None:
        """increment_version should return -1 on every call."""
        service = _make_service()
        for _ in range(3):
            assert await service.increment_version(uuid4()) == -1
