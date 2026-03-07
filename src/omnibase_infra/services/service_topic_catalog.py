# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Topic Catalog Service (stub after OMN-3540).

Consul KV was removed in OMN-3540. This service is now a stub that always
returns empty results with a ``CONSUL_UNAVAILABLE`` warning. The internal
helpers for KV parsing and enrichment are retained for reference but are
no longer exercised at runtime.

Original Design Principles (pre-OMN-3540):
    - Partial success: Returns data even if enrichment fails
    - Warnings array: Communicates backend failures without crashing
    - Version-based in-process cache: TTL keyed by catalog_version

Related Tickets:
    - OMN-2311: Topic Catalog: ServiceTopicCatalog + KV precedence + caching
    - OMN-3540: Remove Consul entirely from omnibase_infra runtime

.. versionadded:: 0.9.0
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from fnmatch import fnmatch
from uuid import UUID

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.models.catalog.catalog_warning_codes import (
    CONSUL_KV_MAX_KEYS_REACHED,
    CONSUL_SCAN_TIMEOUT,
    CONSUL_UNAVAILABLE,
    PARTIAL_NODE_DATA,
    UNRESOLVABLE_TOPIC_PREFIX,
    VERSION_UNKNOWN,
)
from omnibase_infra.models.catalog.model_topic_catalog_entry import (
    ModelTopicCatalogEntry,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.topics.topic_resolver import TopicResolutionError, TopicResolver

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
    """Catalog service stub for ONEX topic metadata.

    Consul KV was removed in OMN-3540. All public methods return empty or
    sentinel results immediately without performing any I/O.

    Coroutine Safety:
        All public methods are async and coroutine-safe. They perform no
        blocking I/O and hold no locks.

    Example:
        >>> service = ServiceTopicCatalog(container=container)
        >>> response = await service.build_catalog(
        ...     correlation_id=uuid4(),
        ... )
        >>> print(response.catalog_version, len(response.topics))
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        topic_resolver: TopicResolver | None = None,
    ) -> None:
        """Initialise the topic catalog service.

        Args:
            container: ONEX container for dependency injection.
            topic_resolver: Optional resolver for mapping topic suffixes to
                Kafka topic names. Defaults to a plain ``TopicResolver()``
                (pass-through).
        """
        self._container = container
        self._topic_resolver = topic_resolver or TopicResolver()

        # in-process cache: catalog_version (int) -> ModelTopicCatalogResponse
        self._cache: dict[int, ModelTopicCatalogResponse] = {}

        logger.info("ServiceTopicCatalog initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_catalog(
        self,
        correlation_id: UUID,
        include_inactive: bool = False,
        topic_pattern: str | None = None,
    ) -> ModelTopicCatalogResponse:
        """Build topic catalog snapshot (stub -- always returns empty).

        Consul KV was removed in OMN-3540. This method returns an empty
        response with a ``CONSUL_UNAVAILABLE`` warning immediately.

        Args:
            correlation_id: Correlation ID for tracing.
            include_inactive: Ignored (retained for API compatibility).
            topic_pattern: Ignored (retained for API compatibility).

        Returns:
            ModelTopicCatalogResponse with no topics and a ``CONSUL_UNAVAILABLE`` warning.
        """
        # Consul removed (OMN-3540): always return empty with CONSUL_UNAVAILABLE.
        warnings: list[str] = [CONSUL_UNAVAILABLE]
        return self._empty_response(
            correlation_id=correlation_id,
            catalog_version=0,
            warnings=warnings,
        )

    async def get_catalog_version(self, correlation_id: UUID) -> int:
        """Read the current catalog version (stub -- always returns -1).

        Returns:
            -1 always (Consul KV removed in OMN-3540).
        """
        # Consul removed (OMN-3540): always return -1.
        return -1

    async def increment_version(self, correlation_id: UUID) -> int:
        """Atomically increment the catalog version.

        Returns:
            -1 always (Consul removed in OMN-3540).
        """
        # Consul removed (OMN-3540): always return -1.
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
            warnings.append(f"{UNRESOLVABLE_TOPIC_PREFIX}{topic_suffix}")
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
            # Valid JSON but not a list (e.g. object or scalar): treat as bad data
            # so that the caller can detect this node had a malformed KV entry and
            # emit PARTIAL_NODE_DATA, just as it would for unparseable JSON.
            logger.debug(
                "Non-list JSON at Consul key %r (got %s), skipping",
                key,
                type(parsed).__name__,
                extra={"correlation_id": str(correlation_id)},
            )
            warnings.append(f"invalid_json_at:{key}")
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
    # Low-level KV helpers (no-ops after OMN-3540 Consul removal)
    # ------------------------------------------------------------------

    async def _kv_get_raw(self, key: str, correlation_id: UUID) -> str | None:
        """Consul KV get — no-op after OMN-3540 Consul removal."""
        return None

    async def _kv_put_raw_with_cas(
        self,
        key: str,
        value: str,
        cas: int,
        correlation_id: UUID,
    ) -> bool:
        """Consul KV CAS put — no-op after OMN-3540 Consul removal."""
        return False

    async def _kv_get_with_modify_index(
        self,
        key: str,
        correlation_id: UUID,
    ) -> tuple[str | None, int]:
        """Consul KV get with ModifyIndex — no-op after OMN-3540 Consul removal."""
        return None, 0

    async def _kv_get_recurse(
        self,
        prefix: str,
        correlation_id: UUID,
    ) -> list[dict[str, object]] | None:
        """Consul KV recursive get — no-op after OMN-3540 Consul removal."""
        return None

    async def _try_cas_increment(self, correlation_id: UUID) -> int:
        """CAS increment — no-op after OMN-3540 Consul removal."""
        return -1


__all__: list[str] = ["ServiceTopicCatalog"]
