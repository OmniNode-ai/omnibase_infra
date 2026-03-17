# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Topic Catalog Service — contract-driven implementation (OMN-5300).

Replaces the legacy Consul KV stub (OMN-3540) with a contract-driven
implementation that reads ``event_bus.subscribe_topics`` and
``event_bus.publish_topics`` from ONEX ``contract.yaml`` files under the
``nodes/`` directory.

Design:
    - Scans all ``contract.yaml`` files under ``contracts_dir`` (default:
      the ``nodes/`` package directory relative to this module).
    - Derives publisher/subscriber sets from each node's event_bus declaration.
    - No I/O at query time — catalog is built once at first ``build_catalog``
      call and cached in-process.
    - ``get_catalog_version`` and ``increment_version`` are stub-only: these
      methods were Consul CAS artefacts with no contract-driven equivalent.

Related Tickets:
    - OMN-2311: Topic Catalog: ServiceTopicCatalog + KV precedence + caching
    - OMN-3540: Remove Consul entirely from omnibase_infra runtime
    - OMN-5300: Replace ServiceTopicCatalog with contract-driven impl

.. versionadded:: 0.9.0
.. versionchanged:: 0.12.0  OMN-5300 — replaced Consul stub with contract-driven scan
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from uuid import UUID

import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.models.catalog.model_topic_catalog_entry import (
    ModelTopicCatalogEntry,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.topics.topic_resolver import TopicResolutionError, TopicResolver

logger = logging.getLogger(__name__)

# Default contracts directory: nodes/ sibling to this services/ package
_DEFAULT_CONTRACTS_DIR = Path(__file__).parent.parent / "nodes"

# Default partitions when not declared in contract
_DEFAULT_PARTITIONS = 1


class TopicAccumulator:
    """Internal mutable accumulator for per-topic catalog data."""

    __slots__ = ("partitions", "publishers", "subscribers", "tags")

    def __init__(self) -> None:
        self.publishers: set[str] = set()
        self.subscribers: set[str] = set()
        self.partitions: int = _DEFAULT_PARTITIONS
        self.tags: set[str] = set()


class ServiceTopicCatalog:
    """Contract-driven topic catalog service.

    Reads ``event_bus.subscribe_topics`` and ``event_bus.publish_topics``
    from all ``contract.yaml`` files found under ``contracts_dir`` to build
    the topic catalog at first access.  Results are cached for the lifetime
    of the instance.

    Coroutine Safety:
        All public methods are async and coroutine-safe. The catalog is
        built synchronously on first call (filesystem scan) but involves no
        blocking network I/O.

    Example:
        >>> service = ServiceTopicCatalog(container=container)
        >>> response = await service.build_catalog(correlation_id=uuid4())
        >>> print(response.catalog_version, len(response.topics))
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        topic_resolver: TopicResolver | None = None,
        contracts_dir: Path | None = None,
    ) -> None:
        """Initialise the topic catalog service.

        Args:
            container: ONEX container for dependency injection.
            topic_resolver: Optional resolver for mapping topic suffixes to
                Kafka topic names. Defaults to a plain ``TopicResolver()``
                (pass-through).
            contracts_dir: Root directory to scan for ``contract.yaml`` files.
                Defaults to the ``nodes/`` package directory inside
                ``omnibase_infra``.
        """
        self._container = container
        self._topic_resolver = topic_resolver or TopicResolver()
        self._contracts_dir = contracts_dir or _DEFAULT_CONTRACTS_DIR

        # Lazy-built cache: None = not yet built, value = full catalog
        self._catalog: ModelTopicCatalogResponse | None = None

        logger.info(
            "ServiceTopicCatalog initialised",
            extra={"contracts_dir": str(self._contracts_dir)},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_catalog(
        self,
        correlation_id: UUID,
        include_inactive: bool = False,
        topic_pattern: str | None = None,
    ) -> ModelTopicCatalogResponse:
        """Build (or return cached) topic catalog from contract YAML files.

        Args:
            correlation_id: Correlation ID for tracing.
            include_inactive: Include topics with no publishers/subscribers.
            topic_pattern: Optional fnmatch glob to filter topic suffixes.

        Returns:
            ModelTopicCatalogResponse with topics derived from contract declarations.
        """
        if self._catalog is None:
            self._catalog = self._build_from_contracts(correlation_id)

        return self._filter_response(
            self._catalog,
            correlation_id=correlation_id,
            include_inactive=include_inactive,
            topic_pattern=topic_pattern,
        )

    async def get_catalog_version(self, correlation_id: UUID) -> int:
        """Return the contract-derived catalog version.

        Returns:
            0 when no contracts have been scanned yet, otherwise the count of
            contract files successfully parsed (stable within an instance lifetime).
        """
        if self._catalog is None:
            self._catalog = self._build_from_contracts(correlation_id)
        return self._catalog.catalog_version

    async def increment_version(self, correlation_id: UUID) -> int:
        """Contract-driven catalogs do not support version increment.

        Returns:
            -1 always — version is derived from contracts, not mutable.
        """
        return -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_from_contracts(self, correlation_id: UUID) -> ModelTopicCatalogResponse:
        """Scan contracts_dir for contract.yaml files and build the catalog.

        Returns:
            Fully-built ModelTopicCatalogResponse.
        """
        warnings: list[str] = []
        topic_map: dict[str, TopicAccumulator] = {}
        node_count = 0
        contract_count = 0

        if not self._contracts_dir.exists():
            logger.warning(
                "ServiceTopicCatalog: contracts_dir does not exist, returning empty catalog",
                extra={
                    "contracts_dir": str(self._contracts_dir),
                    "correlation_id": str(correlation_id),
                },
            )
            return self._empty_response(
                correlation_id=correlation_id,
                catalog_version=0,
                warnings=warnings,
            )

        for contract_path in sorted(self._contracts_dir.rglob("contract.yaml")):
            try:
                with contract_path.open() as f:
                    data = yaml.safe_load(f)
            except Exception:  # noqa: BLE001 — boundary: log and skip bad YAML
                logger.warning(
                    "ServiceTopicCatalog: failed to parse contract YAML, skipping",
                    extra={
                        "path": str(contract_path),
                        "correlation_id": str(correlation_id),
                    },
                    exc_info=True,
                )
                warnings.append(f"parse_error:{contract_path.parent.name}")
                continue

            if not isinstance(data, dict):
                continue

            contract_count += 1
            node_name = str(data.get("name", contract_path.parent.name))

            event_bus = data.get("event_bus")
            if not isinstance(event_bus, dict):
                continue

            node_count += 1
            sub_topics = event_bus.get("subscribe_topics") or []
            pub_topics = event_bus.get("publish_topics") or []

            if not isinstance(sub_topics, list):
                sub_topics = []
            if not isinstance(pub_topics, list):
                pub_topics = []

            for suffix in pub_topics:
                if not isinstance(suffix, str):
                    continue
                if suffix not in topic_map:
                    topic_map[suffix] = TopicAccumulator()
                topic_map[suffix].publishers.add(node_name)

            for suffix in sub_topics:
                if not isinstance(suffix, str):
                    continue
                if suffix not in topic_map:
                    topic_map[suffix] = TopicAccumulator()
                topic_map[suffix].subscribers.add(node_name)

        entries: list[ModelTopicCatalogEntry] = []
        for topic_suffix, info in topic_map.items():
            resolved_name = self._safe_resolve(topic_suffix, correlation_id, warnings)
            entry = ModelTopicCatalogEntry(
                topic_suffix=topic_suffix,
                topic_name=resolved_name,
                description="",
                partitions=info.partitions,
                publisher_count=len(info.publishers),
                subscriber_count=len(info.subscribers),
                tags=tuple(sorted(info.tags)),
            )
            entries.append(entry)

        logger.info(
            "ServiceTopicCatalog: catalog built from contracts",
            extra={
                "contracts_dir": str(self._contracts_dir),
                "contract_count": contract_count,
                "node_count": node_count,
                "topic_count": len(entries),
                "correlation_id": str(correlation_id),
            },
        )

        return ModelTopicCatalogResponse(
            correlation_id=correlation_id,
            topics=tuple(sorted(entries, key=lambda e: e.topic_suffix)),
            catalog_version=contract_count,
            node_count=node_count,
            generated_at=datetime.now(UTC),
            warnings=tuple(warnings),
        )

    def _safe_resolve(
        self,
        topic_suffix: str,
        correlation_id: UUID,
        warnings: list[str],
    ) -> str:
        """Resolve topic suffix to Kafka topic name, falling back to suffix on error."""
        try:
            return self._topic_resolver.resolve(
                topic_suffix, correlation_id=correlation_id
            )
        except TopicResolutionError:
            warnings.append(f"unresolvable_topic:{topic_suffix}")
            return topic_suffix

    def _filter_response(
        self,
        source: ModelTopicCatalogResponse,
        correlation_id: UUID,
        include_inactive: bool,
        topic_pattern: str | None,
    ) -> ModelTopicCatalogResponse:
        """Apply caller-specific filters and return a new response object."""
        topics = source.topics

        if not include_inactive:
            topics = tuple(t for t in topics if t.is_active)

        if topic_pattern is not None:
            topics = tuple(t for t in topics if fnmatch(t.topic_suffix, topic_pattern))

        return ModelTopicCatalogResponse(
            correlation_id=correlation_id,
            topics=topics,
            catalog_version=source.catalog_version,
            node_count=source.node_count,
            generated_at=source.generated_at,
            warnings=source.warnings,
            schema_version=source.schema_version,
        )

    def _empty_response(
        self,
        correlation_id: UUID,
        catalog_version: int,
        warnings: list[str],
    ) -> ModelTopicCatalogResponse:
        """Return an empty catalog response."""
        return ModelTopicCatalogResponse(
            correlation_id=correlation_id,
            topics=(),
            catalog_version=catalog_version,
            node_count=0,
            generated_at=datetime.now(UTC),
            warnings=tuple(warnings),
        )


__all__: list[str] = ["ServiceTopicCatalog"]
