# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Topic Catalog Service.

Reads the topic catalog from Consul KV with caching, timeout budget, and
KV precedence rules (Phase 1.3).

Design Principles:
    - Partial success: Returns data even if enrichment fails
    - Warnings array: Communicates backend failures without crashing
    - Version-based in-process cache: TTL keyed by catalog_version
    - CAS increment with 3 retries + exponential backoff
    - 5-second scan budget with partial results on timeout
    - KV precedence: node arrays authoritative, reverse index is cache only

KV Structure:
    onex/catalog/version                                # monotonic version int
    onex/nodes/{node_id}/event_bus/subscribe_topics     # [topic strings] authoritative
    onex/nodes/{node_id}/event_bus/publish_topics       # [topic strings] authoritative
    onex/nodes/{node_id}/event_bus/subscribe_entries    # [full entries] enrichment only
    onex/nodes/{node_id}/event_bus/publish_entries      # [full entries] enrichment only
    onex/topics/{topic}/subscribers                     # [node_ids] derived cache only

Related Tickets:
    - OMN-2311: Topic Catalog: ServiceTopicCatalog + KV precedence + caching

.. versionadded:: 0.9.0
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
from omnibase_infra.models.catalog.catalog_warning_codes import (
    CONSUL_KV_MAX_KEYS_REACHED,
    CONSUL_SCAN_TIMEOUT,
    CONSUL_UNAVAILABLE,
    PARTIAL_NODE_DATA,
    VERSION_UNKNOWN,
)
from omnibase_infra.models.catalog.model_topic_catalog_entry import (
    ModelTopicCatalogEntry,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.topics.topic_resolver import TopicResolutionError, TopicResolver

if TYPE_CHECKING:
    from omnibase_infra.handlers.handler_consul import HandlerConsul

logger = logging.getLogger(__name__)

# Maximum Consul KV keys to scan per build_catalog invocation
_MAX_KV_KEYS = 10_000

# Maximum seconds for a full catalog scan before returning partial results
_SCAN_BUDGET_SECONDS = 5.0

# CAS retry configuration
_CAS_MAX_RETRIES = 3
_CAS_RETRY_DELAYS = (0.1, 0.2, 0.4)  # seconds per attempt
assert len(_CAS_RETRY_DELAYS) >= _CAS_MAX_RETRIES - 1, (
    "CAS_RETRY_DELAYS must have at least CAS_MAX_RETRIES-1 entries"
)

# Consul KV key constants
_KV_CATALOG_VERSION = "onex/catalog/version"
_KV_NODES_PREFIX = "onex/nodes/"

# Default partitions when unknown
_DEFAULT_PARTITIONS = 1


class ModelTopicInfo:
    """Internal mutable accumulator for per-topic catalog data.

    Not part of the public API. Converted to ModelTopicCatalogEntry at the end
    of a build pass.
    """

    __slots__ = ("description", "partitions", "publishers", "subscribers", "tags")

    def __init__(self) -> None:
        self.publishers: set[str] = set()
        self.subscribers: set[str] = set()
        self.description: str = ""
        self.partitions: int = _DEFAULT_PARTITIONS
        self.tags: set[str] = set()


class ServiceTopicCatalog:
    """Catalog service that reads ONEX topic metadata from Consul KV.

    Combines per-node ``subscribe_topics``/``publish_topics`` arrays
    (authoritative) with optional ``subscribe_entries``/``publish_entries``
    enrichment data to produce a unified ``ModelTopicCatalogResponse``.

    Cache Behaviour:
        Results are cached in-process by ``catalog_version``. When the version
        key in Consul is unchanged the cached response is returned immediately.
        When ``catalog_version == -1`` (version key absent/corrupt) caching is
        disabled and every call performs a full rebuild.

    Timeout Budget:
        A 5-second hard budget applies to the Consul KV recursive scan. If the
        budget is exceeded a partial response is returned with a
        ``CONSUL_SCAN_TIMEOUT`` warning in ``ModelTopicCatalogResponse.warnings``.

    Thread Safety:
        All methods are async. The in-process cache is a plain dict and relies
        on Python's GIL for protection under asyncio (single-threaded event loop).
        External callers that use threads must handle their own synchronisation.

    Example:
        >>> service = ServiceTopicCatalog(container=container, consul_handler=handler)
        >>> response = await service.build_catalog(
        ...     correlation_id=uuid4(),
        ... )
        >>> print(response.catalog_version, len(response.topics))
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        consul_handler: HandlerConsul | None = None,
        topic_resolver: TopicResolver | None = None,
    ) -> None:
        """Initialise the topic catalog service.

        Args:
            container: ONEX container for dependency injection.
            consul_handler: Optional Consul handler. When absent all catalog
                methods return empty results with a ``CONSUL_UNAVAILABLE``
                warning.
            topic_resolver: Optional resolver for mapping topic suffixes to
                Kafka topic names. Defaults to a plain ``TopicResolver()``
                (pass-through).
        """
        self._container = container
        self._consul_handler = consul_handler
        self._topic_resolver = topic_resolver or TopicResolver()

        # in-process cache: catalog_version (int) -> ModelTopicCatalogResponse
        self._cache: dict[int, ModelTopicCatalogResponse] = {}

        logger.info(
            "ServiceTopicCatalog initialised",
            extra={
                "has_consul_handler": consul_handler is not None,
            },
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
        """Build (or return cached) topic catalog snapshot.

        Steps:
        1. Read ``onex/catalog/version`` for current version.
        2. Return cached result immediately if version matches.
        3. Consul KV recursive get ``onex/nodes/`` prefix.
        4. Cross-reference: collect publisher/subscriber node_ids per topic.
        5. Optionally enrich with ``subscribe_entries``/``publish_entries``.
        6. Apply ``TopicResolver.resolve()`` for ``topic_name``.
        7. Filter by ``topic_pattern`` (fnmatch) if provided.
        8. Filter by ``is_active`` if ``include_inactive=False``.
        9. Build ``ModelTopicCatalogEntry`` per topic.
        10. Return ``ModelTopicCatalogResponse`` with warnings.

        Args:
            correlation_id: Correlation ID for tracing.
            include_inactive: Include topics with no publishers/subscribers.
            topic_pattern: Optional fnmatch glob to filter topic suffixes.

        Returns:
            ModelTopicCatalogResponse with topics and any partial-failure warnings.
        """
        warnings: list[str] = []

        if self._consul_handler is None:
            warnings.append(CONSUL_UNAVAILABLE)
            return self._empty_response(
                correlation_id=correlation_id,
                catalog_version=0,
                warnings=warnings,
            )

        # Step 1: get current catalog version
        catalog_version = await self.get_catalog_version(correlation_id)

        # Emit version_unknown warning when catalog version is indeterminate.
        # Note: when catalog_version == -1 AND _kv_get_recurse later returns
        # None, both VERSION_UNKNOWN and CONSUL_UNAVAILABLE will be emitted
        # together in the same response — this is intentional, as each warning
        # describes a distinct failure condition.
        if catalog_version == -1:
            warnings.append(VERSION_UNKNOWN)

        # Step 2: cache hit?
        if catalog_version != -1 and catalog_version in self._cache:
            cached = self._cache[catalog_version]
            # Re-apply caller-specific filters and return a fresh response
            return self._filter_response(
                cached,
                correlation_id=correlation_id,
                include_inactive=include_inactive,
                topic_pattern=topic_pattern,
            )

        # Steps 3-9: full rebuild with timeout budget
        # Only the network I/O (KV fetch) is wrapped in the timeout; processing
        # of whatever was fetched always runs so partial results are preserved.
        raw_kv_items: list[dict[str, object]] = []
        try:
            fetched = await asyncio.wait_for(
                self._kv_get_recurse(_KV_NODES_PREFIX, correlation_id),
                timeout=_SCAN_BUDGET_SECONDS,
            )
            if fetched is not None:
                raw_kv_items = fetched
            else:
                warnings.append(CONSUL_UNAVAILABLE)
        except TimeoutError:
            logger.warning(
                "Consul KV scan exceeded %ss budget, returning partial results",
                _SCAN_BUDGET_SECONDS,
                extra={"correlation_id": str(correlation_id)},
            )
            warnings.append(CONSUL_SCAN_TIMEOUT)

        topics, scan_warnings, node_count = self._process_raw_kv_items(
            raw_kv_items, correlation_id
        )
        warnings.extend(scan_warnings)

        # Step 6: build entries
        entries: list[ModelTopicCatalogEntry] = []
        for topic_suffix, info in topics.items():
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

        # Cache result (skip cache when version unknown)
        if catalog_version != -1:
            self._cache[catalog_version] = full_response
            # Evict any older cached versions to avoid unbounded growth
            stale = [v for v in list(self._cache) if v < catalog_version]
            for v in stale:
                del self._cache[v]

        return self._filter_response(
            full_response,
            correlation_id=correlation_id,
            include_inactive=include_inactive,
            topic_pattern=topic_pattern,
        )

    async def get_catalog_version(self, correlation_id: UUID) -> int:
        """Read the current catalog version from Consul KV.

        Returns:
            Current catalog version (>= 0) or -1 when the key is absent or
            the value is corrupt.
        """
        if self._consul_handler is None:
            return -1

        raw = await self._kv_get_raw(_KV_CATALOG_VERSION, correlation_id)
        if raw is None:
            return -1

        try:
            version = int(raw.strip())
            return max(version, 0)
        except (ValueError, AttributeError):
            logger.warning(
                "Consul catalog version key has invalid value: %r",
                raw,
                extra={"correlation_id": str(correlation_id)},
            )
            return -1

    async def increment_version(self, correlation_id: UUID) -> int:
        """Atomically increment the catalog version using CAS.

        Uses Consul's check-and-set (CAS) to guarantee atomicity. On CAS
        failure retries up to ``_CAS_MAX_RETRIES`` times with exponential
        backoff (100 ms / 200 ms / 400 ms).

        Returns:
            New catalog version on success, -1 if all retries are exhausted.
        """
        if self._consul_handler is None:
            return -1

        for attempt in range(_CAS_MAX_RETRIES):
            new_version = await self._try_cas_increment(correlation_id)
            if new_version != -1:
                return new_version

            if attempt < _CAS_MAX_RETRIES - 1:
                delay = _CAS_RETRY_DELAYS[attempt]
                logger.debug(
                    "CAS increment failed (attempt %d/%d), retrying in %.3fs",
                    attempt + 1,
                    _CAS_MAX_RETRIES,
                    delay,
                    extra={"correlation_id": str(correlation_id)},
                )
                await asyncio.sleep(delay)

        logger.warning(
            "CAS increment exhausted %d retries, returning -1",
            _CAS_MAX_RETRIES,
            extra={"correlation_id": str(correlation_id)},
        )
        return -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_raw_kv_items(
        self,
        raw_items: list[dict[str, object]],
        correlation_id: UUID,
    ) -> tuple[dict[str, ModelTopicInfo], list[str], int]:
        """Process a list of raw Consul KV items into a topic map.

        This is pure CPU work with no I/O; it is intentionally synchronous so
        that it always runs to completion regardless of any prior timeout on the
        network fetch.

        Args:
            raw_items: Items returned by ``_kv_get_recurse`` (may be empty when
                the fetch timed out or returned nothing). Each dict must contain
                at least ``"key"`` (``str``) and ``"value"`` (``str | None``).
            correlation_id: Correlation ID forwarded to ``_parse_json_list`` for
                logging and warning token generation.

        Returns:
            Three-element tuple ``(topic_map, warnings, node_count)`` where:

            - ``topic_map`` maps each topic suffix (``str``) to a
              ``ModelTopicInfo`` accumulator holding publisher/subscriber node
              IDs plus enrichment data (description, partitions, tags).
            - ``warnings`` is a list of string tokens describing any non-fatal
              issues encountered during processing (e.g.
              ``CONSUL_KV_MAX_KEYS_REACHED``, ``PARTIAL_NODE_DATA``,
              ``"invalid_json_at:<key>"``).
            - ``node_count`` is the number of distinct node IDs discovered in
              the KV data.
        """
        warnings: list[str] = []

        if len(raw_items) >= _MAX_KV_KEYS:
            warnings.append(CONSUL_KV_MAX_KEYS_REACHED)

        # Build per-node lookup: node_id -> sub_key -> parsed list
        node_data: dict[str, dict[str, list[object]]] = {}

        # Track node IDs that had at least one malformed KV entry so we can
        # emit a partial_node_data summary warning after scanning all items.
        nodes_with_bad_data: set[str] = set()

        for item in raw_items[:_MAX_KV_KEYS]:
            raw_key = item.get("key")
            raw_value = item.get("value")

            # Narrow types from the KV item dict
            key: str = raw_key if isinstance(raw_key, str) else ""
            value: str | None = raw_value if isinstance(raw_value, str) else None

            # onex/nodes/{node_id}/event_bus/{sub_key}
            if not key.startswith(_KV_NODES_PREFIX):
                continue

            remainder = key[len(_KV_NODES_PREFIX) :]
            parts = remainder.split("/")
            # Expect: node_id / event_bus / sub_key
            if len(parts) < 3 or parts[1] != "event_bus":
                continue

            node_id = parts[0]
            sub_key = "/".join(parts[2:])

            if node_id not in node_data:
                node_data[node_id] = {}

            warnings_before = len(warnings)
            parsed = self._parse_json_list(value, key, correlation_id, warnings)
            node_data[node_id][sub_key] = parsed

            # If _parse_json_list appended a new warning, this node had bad data
            if len(warnings) > warnings_before:
                nodes_with_bad_data.add(node_id)

        node_count = len(node_data)

        # Emit a single partial_node_data summary token when any node had
        # malformed KV entries. The per-key "invalid_json_at:<key>" tokens
        # remain for detailed diagnosis; this summary token lets consumers
        # detect the condition without scanning all warning tokens.
        if nodes_with_bad_data:
            warnings.append(PARTIAL_NODE_DATA)

        # Cross-reference: build topic -> ModelTopicInfo
        topic_map: dict[str, ModelTopicInfo] = {}

        for node_id, data in node_data.items():
            # Authoritative: subscribe_topics and publish_topics arrays
            raw_subscribe = data.get("subscribe_topics", [])
            raw_publish = data.get("publish_topics", [])

            subscribe_topics: list[str] = [
                t for t in raw_subscribe if isinstance(t, str)
            ]
            publish_topics: list[str] = [t for t in raw_publish if isinstance(t, str)]

            # Enrichment: entries (description, partitions, tags)
            raw_sub_entries = data.get("subscribe_entries", [])
            raw_pub_entries = data.get("publish_entries", [])

            subscribe_entries: list[dict[str, object]] = [
                e for e in raw_sub_entries if isinstance(e, dict)
            ]
            publish_entries: list[dict[str, object]] = [
                e for e in raw_pub_entries if isinstance(e, dict)
            ]

            # Build enrichment lookup by topic suffix
            enrichment_by_suffix: dict[str, dict[str, object]] = {}
            for entry in subscribe_entries + publish_entries:
                raw_suffix = entry.get("topic_suffix") or entry.get("topic")
                if isinstance(raw_suffix, str) and raw_suffix:
                    # Intentional last-write-wins: publish_entries override subscribe_entries for the same suffix
                    enrichment_by_suffix[raw_suffix] = entry

            for suffix in publish_topics:
                if suffix not in topic_map:
                    topic_map[suffix] = ModelTopicInfo()
                topic_map[suffix].publishers.add(node_id)
                self._apply_enrichment(
                    topic_map[suffix], enrichment_by_suffix.get(suffix)
                )

            for suffix in subscribe_topics:
                if suffix not in topic_map:
                    topic_map[suffix] = ModelTopicInfo()
                topic_map[suffix].subscribers.add(node_id)
                self._apply_enrichment(
                    topic_map[suffix], enrichment_by_suffix.get(suffix)
                )

        return topic_map, warnings, node_count

    def _apply_enrichment(
        self,
        topic_info: ModelTopicInfo,
        entry: dict[str, object] | None,
    ) -> None:
        """Merge enrichment entry data into topic_info (in-place, non-destructive).

        Only fills in fields that are currently at their default values so that
        the first enrichment entry wins for each field. Fields already set by an
        earlier enrichment pass are left unchanged.

        Args:
            topic_info: Mutable accumulator for a single topic. Modified in-place.
            entry: Optional enrichment dict parsed from a ``subscribe_entries`` or
                ``publish_entries`` KV value. When ``None`` this method is a no-op.

        Returns:
            None. All updates are applied directly to ``topic_info``.
        """
        if entry is None:
            return

        desc = entry.get("description")
        if isinstance(desc, str) and desc and not topic_info.description:
            topic_info.description = desc

        partitions = entry.get("partitions")
        if (
            isinstance(partitions, int)
            and partitions > 0
            and topic_info.partitions == _DEFAULT_PARTITIONS
        ):
            topic_info.partitions = partitions

        raw_tags = entry.get("tags")
        if isinstance(raw_tags, list):
            for tag in raw_tags:
                if isinstance(tag, str):
                    topic_info.tags.add(tag)

    def _safe_resolve(
        self,
        topic_suffix: str,
        correlation_id: UUID,
        warnings: list[str],
    ) -> str:
        """Resolve topic suffix to Kafka topic name, falling back to suffix on error.

        Calls ``TopicResolver.resolve`` and suppresses ``TopicResolutionError`` so
        that a single unresolvable topic does not abort the entire catalog build.
        An ``"unresolvable_topic:<suffix>"`` warning is appended when resolution
        fails.

        Args:
            topic_suffix: Raw topic suffix string from the Consul KV node array
                (e.g. ``"my.service.events.v1"``).
            correlation_id: Correlation ID forwarded to the resolver for tracing.
            warnings: Mutable list that receives an error token when resolution
                fails. Modified in-place.

        Returns:
            Fully-qualified Kafka topic name on success, or ``topic_suffix``
            unchanged when ``TopicResolutionError`` is raised.
        """
        try:
            return self._topic_resolver.resolve(
                topic_suffix, correlation_id=correlation_id
            )
        except TopicResolutionError:
            warnings.append(f"unresolvable_topic:{topic_suffix}")
            return topic_suffix

    def _parse_json_list(
        self,
        value: str | None,
        key: str,
        correlation_id: UUID,
        warnings: list[str],
    ) -> list[object]:
        """Parse a JSON value that is expected to be a list.

        Implements partial-success semantics: any parse failure is recorded as a
        warning and an empty list is returned rather than propagating an exception.
        A ``DEBUG``-level log entry is emitted for every skipped key to aid
        diagnosis without polluting production logs.

        Args:
            value: Raw string value retrieved from Consul KV, or ``None`` when
                the key had no value (Consul returns ``null`` for empty keys).
            key: The Consul KV key path used only for logging and the warning
                token (e.g. ``"onex/nodes/my-node/event_bus/subscribe_topics"``).
            correlation_id: Correlation ID included in the log record for
                distributed tracing.
            warnings: Mutable list that receives an ``"invalid_json_at:<key>"``
                token when parsing fails. Modified in-place.

        Returns:
            Parsed list of JSON values when the value is a valid JSON array.
            Empty list when ``value`` is ``None``, not a JSON array, or
            malformed JSON.
        """
        if value is None:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            return []
        except (json.JSONDecodeError, ValueError):
            logger.debug(
                "Invalid JSON at Consul key %r, skipping",
                key,
                extra={"correlation_id": str(correlation_id)},
            )
            warnings.append(f"invalid_json_at:{key}")
            return []

    def _filter_response(
        self,
        source: ModelTopicCatalogResponse,
        correlation_id: UUID,
        include_inactive: bool,
        topic_pattern: str | None,
    ) -> ModelTopicCatalogResponse:
        """Apply caller-specific filters and return a new response object.

        Creates a new ``ModelTopicCatalogResponse`` from ``source``, optionally
        removing inactive topics (those with no publishers and no subscribers) and
        restricting results to topics whose ``topic_suffix`` matches a shell-style
        glob pattern. All other fields (``catalog_version``, ``node_count``,
        ``generated_at``, ``warnings``, ``schema_version``) are copied verbatim.

        Args:
            source: Fully-built catalog response to filter (typically the cached
                full-catalog object).
            correlation_id: Correlation ID written into the returned response for
                the caller's trace context.
            include_inactive: When ``False`` (default), topics where
                ``ModelTopicCatalogEntry.is_active`` is ``False`` are excluded.
            topic_pattern: Optional :func:`fnmatch.fnmatch` glob matched against
                each entry's ``topic_suffix``. ``None`` disables pattern filtering.

        Returns:
            A new ``ModelTopicCatalogResponse`` containing only the entries that
            pass both the active-status and pattern filters.
        """
        topics = source.topics

        # Filter by active status
        if not include_inactive:
            topics = tuple(t for t in topics if t.is_active)

        # Filter by pattern (fnmatch)
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
        """Return an empty catalog response with zero topics.

        Used as a fast-path return when no Consul handler is configured (emits
        ``CONSUL_UNAVAILABLE`` warning) or when the handler cannot be reached.
        The ``generated_at`` timestamp reflects
        the time of the call so that callers can detect stale responses by age.

        Args:
            correlation_id: Correlation ID written into the returned response.
            catalog_version: Version value to embed (typically ``0`` when the
                version key is absent or the handler is unavailable).
            warnings: List of warning tokens accumulated before the early return
                (e.g. ``[CONSUL_UNAVAILABLE]``). Copied into the response tuple.

        Returns:
            A ``ModelTopicCatalogResponse`` with an empty ``topics`` tuple,
            ``node_count`` of ``0``, and the supplied ``warnings``.
        """
        return ModelTopicCatalogResponse(
            correlation_id=correlation_id,
            topics=(),
            catalog_version=catalog_version,
            node_count=0,
            generated_at=datetime.now(UTC),
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------
    # Low-level Consul KV helpers (delegate to HandlerConsul internals)
    # ------------------------------------------------------------------

    async def _kv_get_raw(self, key: str, correlation_id: UUID) -> str | None:
        """Get raw string value from Consul KV via HandlerConsul's mixin method.

        Delegates to ``MixinConsulTopicIndex.kv_get_raw`` which is already defined
        on ``HandlerConsul``. Any exception from the handler is caught and logged
        at ``DEBUG`` level so that a single KV failure does not abort a broader
        operation.

        Args:
            key: Fully-qualified Consul KV key path
                (e.g. ``"onex/catalog/version"``).
            correlation_id: Correlation ID included in the log record for
                distributed tracing.

        Returns:
            Decoded string value of the key, or ``None`` when the key is absent,
            the handler is unavailable, or any exception is raised.
        """
        if self._consul_handler is None:
            return None
        try:
            # HandlerConsul inherits kv_get_raw from MixinConsulTopicIndex
            return await self._consul_handler.kv_get_raw(key, correlation_id)
        except Exception:
            logger.debug(
                "KV get failed for key %r",
                key,
                extra={"correlation_id": str(correlation_id)},
            )
            return None

    async def _kv_put_raw_with_cas(
        self,
        key: str,
        value: str,
        cas: int,
        correlation_id: UUID,
    ) -> bool:
        """Put value to Consul KV with a check-and-set (CAS) guard.

        Delegates to ``HandlerConsul.kv_put_raw_with_cas``, which routes through
        the handler's retry machinery and circuit breaker. Any exception from the
        handler is caught and logged at ``DEBUG`` level; the caller receives
        ``False`` and should treat this identically to a CAS conflict.

        Args:
            key: Fully-qualified Consul KV key path to write
                (e.g. ``"onex/catalog/version"``).
            value: String value to store. The caller is responsible for encoding
                (e.g. converting an integer to its decimal string representation).
            cas: Consul ``ModifyIndex`` obtained from a prior
                ``_kv_get_with_modify_index`` call. Pass ``0`` to create a key
                only when it does not yet exist.
            correlation_id: Correlation ID included in the log record for
                distributed tracing.

        Returns:
            ``True`` when the write was accepted (CAS index matched the current
            ``ModifyIndex`` in Consul). ``False`` when the CAS check failed
            (another writer modified the key first), the handler is unavailable,
            or any exception is raised.
        """
        if self._consul_handler is None:
            return False
        try:
            return await self._consul_handler.kv_put_raw_with_cas(
                key, value, cas, correlation_id
            )
        except Exception:
            logger.debug(
                "CAS put failed for key %r",
                key,
                extra={"correlation_id": str(correlation_id)},
            )
            return False

    async def _kv_get_with_modify_index(
        self,
        key: str,
        correlation_id: UUID,
    ) -> tuple[str | None, int]:
        """Get value and ModifyIndex for a KV key (required for CAS writes).

        Delegates to ``HandlerConsul.kv_get_with_modify_index``, which routes
        through the handler's retry machinery and circuit breaker. Any exception
        from the handler is caught and logged at ``DEBUG`` level; the caller
        receives ``(None, 0)`` and should treat this as a key-absent condition.

        Args:
            key: Fully-qualified Consul KV key path
                (e.g. ``"onex/catalog/version"``).
            correlation_id: Correlation ID included in the log record for
                distributed tracing.

        Returns:
            Two-element tuple ``(value, modify_index)`` where ``value`` is the
            decoded string value of the key (or ``None`` when absent) and
            ``modify_index`` is the Consul ``ModifyIndex`` at the time of the
            read. A ``modify_index`` of ``0`` signals that the key does not exist
            yet; passing ``cas=0`` to a subsequent CAS write creates the key only
            if it is still absent.
        """
        if self._consul_handler is None:
            return None, 0
        try:
            return await self._consul_handler.kv_get_with_modify_index(
                key, correlation_id
            )
        except Exception:
            logger.debug(
                "KV get with modify index failed for key %r",
                key,
                extra={"correlation_id": str(correlation_id)},
            )
            return None, 0

    async def _kv_get_recurse(
        self,
        prefix: str,
        correlation_id: UUID,
    ) -> list[dict[str, object]] | None:
        """Perform a recursive Consul KV get for all keys under a prefix.

        Delegates to ``HandlerConsul.kv_get_recurse``, which routes through the
        handler's retry machinery and circuit breaker. Any exception from the
        handler is caught and logged at ``DEBUG`` level.

        Args:
            prefix: Consul KV key prefix to scan recursively
                (e.g. ``"onex/nodes/"``). All keys that begin with this string
                are returned.
            correlation_id: Correlation ID included in the log record for
                distributed tracing.

        Returns:
            List of dicts, each containing at least ``"key"`` (``str``),
            ``"value"`` (``str | None``), and ``"modify_index"`` (``int``) fields,
            representing every KV entry under the prefix. Returns ``None`` when
            the handler is unavailable, the prefix does not exist, or any
            exception is raised.
        """
        if self._consul_handler is None:
            return None
        try:
            return await self._consul_handler.kv_get_recurse(prefix, correlation_id)
        except Exception:
            logger.debug(
                "KV recurse get failed for prefix %r",
                prefix,
                extra={"correlation_id": str(correlation_id)},
            )
            return None

    async def _try_cas_increment(self, correlation_id: UUID) -> int:
        """Attempt a single check-and-set increment of the catalog version key.

        Reads the current value and ``ModifyIndex`` of ``onex/catalog/version``
        via ``_kv_get_with_modify_index``, computes ``new_version = current + 1``
        (defaulting to ``1`` when the key is absent or its value is not a valid
        integer), then writes ``new_version`` back using ``_kv_put_raw_with_cas``.

        If the CAS write fails (another writer incremented the key between the
        read and write), the method returns ``-1`` immediately. The caller
        (``increment_version``) is responsible for retrying.

        Args:
            correlation_id: Correlation ID forwarded to all underlying KV helpers
                for distributed tracing.

        Returns:
            The newly written version integer (>= 1) when the CAS write succeeds.
            ``-1`` when the CAS check fails (concurrent modification) or any
            underlying KV helper returns an error sentinel.
        """
        current_str, modify_index = await self._kv_get_with_modify_index(
            _KV_CATALOG_VERSION, correlation_id
        )

        # Parse current value
        if current_str is None:
            new_version = 1
        else:
            try:
                new_version = int(current_str.strip()) + 1
            except (ValueError, AttributeError):
                new_version = 1

        success = await self._kv_put_raw_with_cas(
            _KV_CATALOG_VERSION,
            str(new_version),
            modify_index,
            correlation_id,
        )

        if success:
            return new_version

        return -1


__all__: list[str] = ["ServiceTopicCatalog"]
