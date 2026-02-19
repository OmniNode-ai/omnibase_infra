# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Topic catalog warning code constants.

Defines the canonical string constants for all partial-failure warning codes
that may appear in ``ModelTopicCatalogResponse.warnings``. Each warning code
is a short, stable identifier—never a free-form message—so that consumers
(dashboard, tests, alerting) can match against them without string parsing.

Warning Code Reference:

    ``consul_unavailable``
        Consul connection failure prevented the catalog from being built.
        Emitted by ``ServiceTopicCatalog.build_catalog()`` in two scenarios:
        (1) a Consul KV recursive scan returns ``None`` due to a transport or
        availability error; (2) the fast-path return when no ``consul_handler``
        is configured at all.

    ``consul_scan_timeout``
        The 5-second scan budget was exceeded before the KV scan completed.
        A partial response (with whatever data was available) is returned.
        Emitted by ``ServiceTopicCatalog.build_catalog()``.

    ``consul_kv_max_keys_reached``
        The KV scan returned at least ``_MAX_KV_KEYS`` items, indicating the
        result set was capped. The catalog is built from the truncated result.
        Emitted by ``ServiceTopicCatalog._process_raw_kv_items()``.

    ``internal_error``
        An unexpected exception was raised during catalog retrieval. An empty
        catalog response is returned. Emitted by
        ``HandlerTopicCatalogQuery.handle()``.

    ``invalid_query_payload``
        The incoming query payload was malformed (wrong type, missing required
        fields, or failed validation). An empty catalog response is returned.
        Emitted by ``HandlerTopicCatalogQuery.handle()``.

    ``partial_node_data``
        One or more nodes had malformed KV entries that could not be parsed.
        The affected entries are skipped and the catalog is built from the
        remaining valid data. Emitted by
        ``ServiceTopicCatalog.build_catalog()`` via ``_process_raw_kv_items()``.

    ``version_unknown``
        The catalog version key is absent or corrupt (``catalog_version == -1``).
        Caching is disabled for this response. Emitted by
        ``ServiceTopicCatalog.build_catalog()``.

Related Tickets:
    - OMN-2312: Topic Catalog: response warnings channel
    - OMN-2311: Topic Catalog: ServiceTopicCatalog + KV precedence + caching
    - OMN-2310: Topic Catalog model + suffix foundation

.. versionadded:: 0.9.0
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Warning code constants
# ---------------------------------------------------------------------------

#: Consul connection failure during catalog build (no handler configured, or KV scan returned None).
CONSUL_UNAVAILABLE: str = "consul_unavailable"

#: 5-second scan budget exceeded; partial results returned.
CONSUL_SCAN_TIMEOUT: str = "consul_scan_timeout"

#: KV scan result set was capped at _MAX_KV_KEYS; catalog built from truncated data.
CONSUL_KV_MAX_KEYS_REACHED: str = "consul_kv_max_keys_reached"

#: Unexpected exception during catalog retrieval; empty response returned.
INTERNAL_ERROR: str = "internal_error"

#: Malformed query payload (wrong type or missing required fields).
INVALID_QUERY_PAYLOAD: str = "invalid_query_payload"

#: One or more nodes had malformed KV entries that were logged and skipped.
PARTIAL_NODE_DATA: str = "partial_node_data"

#: Catalog version key absent or corrupt (CAS failure / version == -1).
VERSION_UNKNOWN: str = "version_unknown"


__all__: list[str] = [
    "CONSUL_UNAVAILABLE",
    "CONSUL_SCAN_TIMEOUT",
    "CONSUL_KV_MAX_KEYS_REACHED",
    "INTERNAL_ERROR",
    "INVALID_QUERY_PAYLOAD",
    "PARTIAL_NODE_DATA",
    "VERSION_UNKNOWN",
]
