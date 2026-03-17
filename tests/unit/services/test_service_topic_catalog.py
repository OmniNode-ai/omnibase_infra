# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for ServiceTopicCatalog — contract-driven implementation (OMN-5300).

Tests cover:
    - build_catalog with an empty contracts directory returns empty response
    - build_catalog with real contract YAMLs returns populated topics
    - include_inactive and topic_pattern filters work correctly
    - get_catalog_version reflects the number of contracts parsed
    - increment_version always returns -1 (not applicable to contract-driven impl)
    - Missing contracts directory returns empty catalog gracefully

Related Tickets:
    - OMN-5300: Replace ServiceTopicCatalog with contract-driven impl
    - OMN-3540: Remove Consul from omnibase_infra
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.services.service_topic_catalog import ServiceTopicCatalog
from omnibase_infra.topics.topic_resolver import TopicResolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(
    contracts_dir: Path | None = None,
    topic_resolver: TopicResolver | None = None,
) -> ServiceTopicCatalog:
    """Create a ServiceTopicCatalog with a mock container."""
    container = MagicMock()
    return ServiceTopicCatalog(
        container=container,
        topic_resolver=topic_resolver,
        contracts_dir=contracts_dir,
    )


def _write_contract(
    directory: Path, node_name: str, sub: list[str], pub: list[str]
) -> None:
    """Write a minimal contract.yaml with event_bus topics."""
    data = {
        "name": node_name,
        "event_bus": {
            "subscribe_topics": sub,
            "publish_topics": pub,
        },
    }
    node_dir = directory / node_name
    node_dir.mkdir(parents=True, exist_ok=True)
    with (node_dir / "contract.yaml").open("w") as f:
        yaml.dump(data, f)


# ---------------------------------------------------------------------------
# Test: empty / missing directory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmptyContracts:
    """Tests when no contract files are present."""

    @pytest.mark.asyncio
    async def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        """Non-existent contracts_dir yields an empty catalog."""
        service = _make_service(contracts_dir=tmp_path / "does_not_exist")
        response = await service.build_catalog(correlation_id=uuid4())

        assert isinstance(response, ModelTopicCatalogResponse)
        assert response.topics == ()

    @pytest.mark.asyncio
    async def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        """Empty contracts_dir yields an empty catalog."""
        service = _make_service(contracts_dir=tmp_path)
        response = await service.build_catalog(correlation_id=uuid4())

        assert response.topics == ()
        assert response.catalog_version == 0

    @pytest.mark.asyncio
    async def test_get_catalog_version_empty(self, tmp_path: Path) -> None:
        """get_catalog_version returns 0 when no contracts found."""
        service = _make_service(contracts_dir=tmp_path)
        version = await service.get_catalog_version(uuid4())
        assert version == 0


# ---------------------------------------------------------------------------
# Test: contract-driven catalog population
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContractDrivenCatalog:
    """Tests for catalog population from contract YAML files."""

    @pytest.mark.asyncio
    async def test_topics_from_contracts(self, tmp_path: Path) -> None:
        """build_catalog returns topics declared in event_bus sections."""
        _write_contract(
            tmp_path,
            "node_a",
            sub=["onex.evt.platform.node-heartbeat.v1"],
            pub=["onex.evt.platform.node-registration-result.v1"],
        )
        service = _make_service(contracts_dir=tmp_path)
        response = await service.build_catalog(correlation_id=uuid4())

        suffixes = {e.topic_suffix for e in response.topics}
        assert "onex.evt.platform.node-heartbeat.v1" in suffixes
        assert "onex.evt.platform.node-registration-result.v1" in suffixes

    @pytest.mark.asyncio
    async def test_publisher_subscriber_counts(self, tmp_path: Path) -> None:
        """Publisher and subscriber counts are correct across multiple nodes."""
        _write_contract(
            tmp_path,
            "node_publisher",
            sub=[],
            pub=["onex.evt.platform.some-event.v1"],
        )
        _write_contract(
            tmp_path,
            "node_subscriber_1",
            sub=["onex.evt.platform.some-event.v1"],
            pub=[],
        )
        _write_contract(
            tmp_path,
            "node_subscriber_2",
            sub=["onex.evt.platform.some-event.v1"],
            pub=[],
        )
        service = _make_service(contracts_dir=tmp_path)
        response = await service.build_catalog(correlation_id=uuid4())

        entry = next(
            (
                e
                for e in response.topics
                if e.topic_suffix == "onex.evt.platform.some-event.v1"
            ),
            None,
        )
        assert entry is not None
        assert entry.publisher_count == 1
        assert entry.subscriber_count == 2

    @pytest.mark.asyncio
    async def test_catalog_version_equals_contract_count(self, tmp_path: Path) -> None:
        """catalog_version equals the number of contract files successfully parsed."""
        _write_contract(tmp_path, "node_a", sub=[], pub=["onex.evt.x.v1"])
        _write_contract(tmp_path, "node_b", sub=["onex.evt.x.v1"], pub=[])
        service = _make_service(contracts_dir=tmp_path)
        response = await service.build_catalog(correlation_id=uuid4())
        assert response.catalog_version == 2

    @pytest.mark.asyncio
    async def test_catalog_cached_after_first_build(self, tmp_path: Path) -> None:
        """Subsequent calls return the cached catalog object (same generated_at)."""
        _write_contract(tmp_path, "node_a", sub=[], pub=["onex.evt.x.v1"])
        service = _make_service(contracts_dir=tmp_path)
        r1 = await service.build_catalog(correlation_id=uuid4())
        r2 = await service.build_catalog(correlation_id=uuid4())
        assert r1.generated_at == r2.generated_at

    @pytest.mark.asyncio
    async def test_no_event_bus_section_skipped(self, tmp_path: Path) -> None:
        """Contracts without event_bus section contribute no topics."""
        node_dir = tmp_path / "node_no_bus"
        node_dir.mkdir()
        (node_dir / "contract.yaml").write_text('name: "node_no_bus"\n')
        service = _make_service(contracts_dir=tmp_path)
        response = await service.build_catalog(correlation_id=uuid4())
        assert response.topics == ()

    @pytest.mark.asyncio
    async def test_malformed_yaml_emits_warning(self, tmp_path: Path) -> None:
        """A malformed contract YAML emits a parse_error warning."""
        node_dir = tmp_path / "bad_node"
        node_dir.mkdir()
        (node_dir / "contract.yaml").write_text("key: [unclosed bracket\n")
        service = _make_service(contracts_dir=tmp_path)
        response = await service.build_catalog(correlation_id=uuid4())
        assert any(w.startswith("parse_error:") for w in response.warnings)


# ---------------------------------------------------------------------------
# Test: filtering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCatalogFiltering:
    """Tests for include_inactive and topic_pattern filters."""

    @pytest.mark.asyncio
    async def test_include_inactive_does_not_crash(self, tmp_path: Path) -> None:
        """include_inactive=False call completes without error."""
        _write_contract(
            tmp_path,
            "node_a",
            sub=["onex.evt.platform.orphan.v1"],
            pub=[],
        )
        service = _make_service(contracts_dir=tmp_path)
        response = await service.build_catalog(
            correlation_id=uuid4(), include_inactive=False
        )
        assert isinstance(response, ModelTopicCatalogResponse)

    @pytest.mark.asyncio
    async def test_topic_pattern_filters_results(self, tmp_path: Path) -> None:
        """topic_pattern fnmatch glob restricts returned topics."""
        _write_contract(
            tmp_path,
            "node_a",
            sub=[],
            pub=[
                "onex.evt.platform.alpha.v1",
                "onex.evt.platform.beta.v1",
            ],
        )
        service = _make_service(contracts_dir=tmp_path)
        response = await service.build_catalog(
            correlation_id=uuid4(),
            include_inactive=True,
            topic_pattern="onex.evt.platform.alpha.*",
        )
        suffixes = {e.topic_suffix for e in response.topics}
        assert "onex.evt.platform.alpha.v1" in suffixes
        assert "onex.evt.platform.beta.v1" not in suffixes


# ---------------------------------------------------------------------------
# Test: get_catalog_version and increment_version
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVersionMethods:
    """Tests for get_catalog_version and increment_version."""

    @pytest.mark.asyncio
    async def test_increment_version_always_minus_one(self, tmp_path: Path) -> None:
        """increment_version always returns -1 (not applicable to contract-driven impl)."""
        service = _make_service(contracts_dir=tmp_path)
        for _ in range(3):
            assert await service.increment_version(uuid4()) == -1

    @pytest.mark.asyncio
    async def test_get_catalog_version_matches_build_catalog(
        self, tmp_path: Path
    ) -> None:
        """get_catalog_version returns the same version as build_catalog."""
        _write_contract(tmp_path, "node_a", sub=[], pub=["onex.evt.x.v1"])
        service = _make_service(contracts_dir=tmp_path)
        build_response = await service.build_catalog(correlation_id=uuid4())
        version = await service.get_catalog_version(uuid4())
        assert version == build_response.catalog_version
