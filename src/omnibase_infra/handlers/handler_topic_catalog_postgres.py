# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""PostgreSQL-backed Topic Catalog Handler.

Replaces ``ServiceTopicCatalogPostgres`` with an ONEX handler that owns the
PostgreSQL I/O boundary for topic catalog reads.  Reads ``subscribe_topics``
and ``publish_topics`` JSONB columns from the ``node_registrations`` table and
satisfies the ``ProtocolTopicCatalogService`` interface.

Design Principles:
    - Implements ``ProtocolTopicCatalogService`` — drop-in replacement for the
      old service class, no changes required in ``HandlerTopicCatalogQuery``
    - Explicit lifecycle: ``initialize()`` / ``shutdown()`` called by the
      container; avoids silent resource leaks
    - Same ``ModelTopicCatalogResponse`` interface as ``ServiceTopicCatalog``
    - Version key = truncated MD5 hash of ``MAX(updated_at)`` across all rows
    - In-process cache keyed by ``catalog_version`` (same eviction logic)
    - No Consul dependency — PostgreSQL is the single source of truth
    - Partial success: empty response on DB error with ``DB_UNAVAILABLE`` warning

Handler vs. Service classification (OMN-4011):
    ``ServiceTopicCatalogPostgres`` scored 4/5 on the OMN-4004 classification
    rubric — it owns a direct Postgres I/O connection, has a lifecycle (cache +
    pool), exposes a single dispatch entry point, and is already injectable.
    Converting to ``HandlerTopicCatalogPostgres`` makes the I/O boundary
    explicit and enables container-managed lifecycle.

Ambiguity signals satisfied (OMN-4004 Section 5):
    - S2: Dispatch ownership clarified — ``build_catalog()`` is the single
      entry point, replacing the service-layer indirection.
    - S3: Lifecycle enforced — ``initialize()`` / ``shutdown()`` are now
      explicit handler lifecycle hooks managed by the container.

Related Tickets:
    - OMN-2746: Replace ServiceTopicCatalog Consul KV backend with PostgreSQL
    - OMN-4011: ServiceTopicCatalogPostgres -> handler (this ticket)

.. versionadded:: 0.11.0
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from fnmatch import fnmatch
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.catalog.model_topic_catalog_entry import (
    ModelTopicCatalogEntry,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.topics.topic_resolver import TopicResolutionError, TopicResolver

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

# Warning codes — DB-specific (no Consul codes used by this handler)
DB_UNAVAILABLE: str = "db_unavailable"

# Default partitions when unknown
_DEFAULT_PARTITIONS = 1

# Query: compute version hash + pull all topic data in one pass
_SQL_FETCH_TOPICS = """
SELECT
    node_id,
    node_type,
    subscribe_topics,
    publish_topics,
    MAX(updated_at) OVER () AS max_updated_at
FROM node_registrations
ORDER BY node_id
"""

_SQL_FETCH_VERSION = """
SELECT md5(MAX(updated_at)::text) AS version_hash
FROM node_registrations
"""


class HandlerTopicCatalogPostgres:
    """PostgreSQL-backed topic catalog handler.

    Provides the ``build_catalog`` interface matching ``ProtocolTopicCatalogService``
    and reads ``subscribe_topics`` / ``publish_topics`` JSONB columns from the
    ``node_registrations`` table.

    Lifecycle:
        Call ``initialize()`` after construction and ``shutdown()`` when the
        handler is no longer needed. The container manages this lifecycle.

    Cache Behaviour:
        Results are cached in-process by ``catalog_version``. The version is
        derived from ``md5(MAX(updated_at)::text)``. When the version is
        unchanged the cached response is returned immediately.

    Timeout Budget:
        A ``query_timeout_seconds`` (default 5.0s) budget applies to each DB
        query. If exceeded, ``DB_UNAVAILABLE`` is emitted and an empty response
        is returned.

    Thread Safety:
        All methods are async. The in-process cache relies on Python's GIL
        for protection under a single asyncio event loop.

    Example:
        >>> handler = HandlerTopicCatalogPostgres(
        ...     container=container,
        ...     pool=pool,
        ... )
        >>> await handler.initialize()
        >>> response = await handler.build_catalog(correlation_id=uuid4())
        >>> print(response.catalog_version, len(response.topics))
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        pool: asyncpg.Pool | None = None,
        topic_resolver: TopicResolver | None = None,
        query_timeout_seconds: float = 5.0,
    ) -> None:
        """Initialise the PostgreSQL topic catalog handler.

        Args:
            container: ONEX container for dependency injection.
            pool: Optional asyncpg connection pool. When ``None`` all catalog
                methods return empty results with a ``DB_UNAVAILABLE`` warning.
            topic_resolver: Optional resolver for mapping topic suffixes to
                Kafka topic names. Defaults to a plain ``TopicResolver()``
                (pass-through).
            query_timeout_seconds: Maximum seconds for a DB query before
                returning partial results. Defaults to 5.0.
        """
        self._container = container
        self._pool = pool
        self._topic_resolver = topic_resolver or TopicResolver()
        self._query_timeout_seconds = query_timeout_seconds

        # in-process cache: catalog_version (int) -> ModelTopicCatalogResponse
        self._cache: dict[int, ModelTopicCatalogResponse] = {}

        logger.info(
            "HandlerTopicCatalogPostgres initialised",
            extra={
                "has_pool": pool is not None,
                "query_timeout_seconds": query_timeout_seconds,
            },
        )

    # ------------------------------------------------------------------
    # Handler classification (OMN-4004: required handler metadata)
    # ------------------------------------------------------------------

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER — This is an infrastructure-layer
            handler that owns a PostgreSQL I/O boundary for topic catalog reads.
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler.

        Returns:
            EnumHandlerTypeCategory.EFFECT — This handler performs side-effecting
            database I/O operations. EFFECT handlers are not deterministic and
            interact with external systems (PostgreSQL).
        """
        return EnumHandlerTypeCategory.EFFECT

    # ------------------------------------------------------------------
    # Handler lifecycle (OMN-4011: explicit lifecycle enforcement)
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Lifecycle hook — called by the container after construction.

        Validates the pool connection is reachable. Logs a warning when no
        pool is configured (all catalog reads will return ``DB_UNAVAILABLE``).
        """
        if self._pool is None:
            logger.warning(
                "HandlerTopicCatalogPostgres: no pool configured — "
                "all catalog reads will return DB_UNAVAILABLE"
            )
        else:
            logger.debug("HandlerTopicCatalogPostgres initialized with pool")

    async def shutdown(self) -> None:
        """Lifecycle hook — called by the container on teardown.

        Clears the in-process cache. The pool itself is owned externally
        (passed in at construction), so it is not closed here.
        """
        self._cache.clear()
        logger.debug("HandlerTopicCatalogPostgres shut down — cache cleared")

    # ------------------------------------------------------------------
    # Public API (satisfies ProtocolTopicCatalogService)
    # ------------------------------------------------------------------

    async def build_catalog(
        self,
        correlation_id: UUID,
        include_inactive: bool = False,
        topic_pattern: str | None = None,
    ) -> ModelTopicCatalogResponse:
        """Build (or return cached) topic catalog snapshot from PostgreSQL.

        Steps:
        1. Query ``md5(MAX(updated_at)::text)`` for current version hash.
        2. Return cached result immediately if version matches.
        3. Full SELECT on ``node_registrations`` for ``subscribe_topics`` +
           ``publish_topics`` columns.
        4. Cross-reference: collect publisher/subscriber node_ids per topic.
        5. Apply ``TopicResolver.resolve()`` for ``topic_name``.
        6. Filter by ``topic_pattern`` (fnmatch) if provided.
        7. Filter by ``is_active`` if ``include_inactive=False``.
        8. Return ``ModelTopicCatalogResponse`` with warnings.

        Args:
            correlation_id: Correlation ID for tracing.
            include_inactive: Include topics with no publishers/subscribers.
            topic_pattern: Optional fnmatch glob to filter topic suffixes.

        Returns:
            ModelTopicCatalogResponse with topics and any partial-failure warnings.
        """
        warnings: list[str] = []

        if self._pool is None:
            warnings.append(DB_UNAVAILABLE)
            return self._empty_response(
                correlation_id=correlation_id,
                catalog_version=0,
                warnings=warnings,
            )

        # Step 1: get current catalog version
        catalog_version = await self._get_catalog_version(correlation_id)

        # Step 2: cache hit
        if catalog_version != -1 and catalog_version in self._cache:
            cached = self._cache[catalog_version]
            return self._filter_response(
                cached,
                correlation_id=correlation_id,
                include_inactive=include_inactive,
                topic_pattern=topic_pattern,
            )

        # Step 3: full query
        rows: list[dict[str, object]] = []
        try:
            rows = await asyncio.wait_for(
                self._fetch_topic_rows(correlation_id),
                timeout=self._query_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "PostgreSQL topic catalog query exceeded %.1fs budget",
                self._query_timeout_seconds,
                extra={"correlation_id": str(correlation_id)},
            )
            warnings.append(DB_UNAVAILABLE)
        except Exception:
            logger.warning(
                "PostgreSQL topic catalog query failed",
                extra={"correlation_id": str(correlation_id)},
                exc_info=True,
            )
            warnings.append(DB_UNAVAILABLE)

        # Steps 4-5: build topic map
        topic_map: dict[str, ModelTopicInfoPostgres] = {}
        node_count = 0

        for row in rows:
            node_id = str(row.get("node_id", ""))
            if not node_id:
                continue
            node_count += 1

            sub_topics = self._parse_json_list(
                row.get("subscribe_topics"), correlation_id
            )
            pub_topics = self._parse_json_list(
                row.get("publish_topics"), correlation_id
            )

            for suffix in pub_topics:
                if not isinstance(suffix, str):
                    continue
                if suffix not in topic_map:
                    topic_map[suffix] = ModelTopicInfoPostgres()
                topic_map[suffix].publishers.add(node_id)

            for suffix in sub_topics:
                if not isinstance(suffix, str):
                    continue
                if suffix not in topic_map:
                    topic_map[suffix] = ModelTopicInfoPostgres()
                topic_map[suffix].subscribers.add(node_id)

        # Step 5: resolve topic names
        entries: list[ModelTopicCatalogEntry] = []
        for topic_suffix, info in topic_map.items():
            resolved_name = self._safe_resolve(topic_suffix, correlation_id, warnings)
            entry = ModelTopicCatalogEntry(
                topic_suffix=topic_suffix,
                topic_name=resolved_name,
                description=info.description,
                partitions=info.partitions,
                publisher_count=len(info.publishers),
                subscriber_count=len(info.subscribers),
                tags=tuple(sorted(info.tags)),
            )
            entries.append(entry)

        full_response = ModelTopicCatalogResponse(
            correlation_id=correlation_id,
            topics=tuple(sorted(entries, key=lambda e: e.topic_suffix)),
            catalog_version=max(catalog_version, 0),
            node_count=node_count,
            generated_at=datetime.now(UTC),
            warnings=tuple(warnings),
        )

        # Cache result (skip when version unknown)
        if catalog_version != -1:
            self._cache[catalog_version] = full_response
            stale = [v for v in list(self._cache) if v < catalog_version]
            for v in stale:
                del self._cache[v]

        return self._filter_response(
            full_response,
            correlation_id=correlation_id,
            include_inactive=include_inactive,
            topic_pattern=topic_pattern,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_catalog_version(self, correlation_id: UUID) -> int:
        """Compute catalog version from md5(MAX(updated_at)).

        Returns:
            Non-negative integer version, or -1 on error / empty table.
        """
        if self._pool is None:
            return -1
        try:
            async with self._pool.acquire() as conn:
                row = await asyncio.wait_for(
                    conn.fetchrow(_SQL_FETCH_VERSION),
                    timeout=self._query_timeout_seconds,
                )
            if row is None or row["version_hash"] is None:
                return -1
            version_hash: str = row["version_hash"]
            # Take last 8 hex chars → unsigned 32-bit integer
            return int(version_hash[-8:], 16)
        except Exception:
            logger.debug(
                "Failed to compute catalog version from PostgreSQL",
                extra={"correlation_id": str(correlation_id)},
                exc_info=True,
            )
            return -1

    async def _fetch_topic_rows(
        self,
        correlation_id: UUID,
    ) -> list[dict[str, object]]:
        """Fetch all node_registrations rows with topic columns.

        Returns:
            List of dicts with keys: node_id, node_type,
            subscribe_topics (str), publish_topics (str).
        """
        if self._pool is None:
            return []
        async with self._pool.acquire() as conn:
            db_rows = await conn.fetch(_SQL_FETCH_TOPICS)
        result: list[dict[str, object]] = []
        for row in db_rows:
            result.append(
                {
                    "node_id": str(row["node_id"]),
                    "node_type": row["node_type"],
                    "subscribe_topics": row["subscribe_topics"],
                    "publish_topics": row["publish_topics"],
                }
            )
        return result

    def _parse_json_list(
        self,
        value: object,
        correlation_id: UUID,
    ) -> list[object]:
        """Parse a JSONB value (already decoded by asyncpg) into a list.

        asyncpg decodes JSONB automatically, so ``value`` may already be a list
        or a string (if the column stores a raw JSON string). Both cases are
        handled gracefully.
        """
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                logger.debug(
                    "Invalid JSON in subscribe/publish_topics column",
                    extra={"correlation_id": str(correlation_id)},
                )
        return []

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
        """Return an empty catalog response for error conditions."""
        return ModelTopicCatalogResponse(
            correlation_id=correlation_id,
            topics=(),
            catalog_version=catalog_version,
            node_count=0,
            generated_at=datetime.now(UTC),
            warnings=tuple(warnings),
        )


class ModelTopicInfoPostgres:
    """Internal mutable accumulator for per-topic catalog data."""

    __slots__ = ("description", "partitions", "publishers", "subscribers", "tags")

    def __init__(self) -> None:
        self.publishers: set[str] = set()
        self.subscribers: set[str] = set()
        self.description: str = ""
        self.partitions: int = _DEFAULT_PARTITIONS
        self.tags: set[str] = set()


__all__: list[str] = [
    "HandlerTopicCatalogPostgres",
    "DB_UNAVAILABLE",
]
