# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic catalog models for the ONEX platform.

Provides Pydantic models for querying, responding to, and notifying about
topic catalog changes, plus the canonical warning code constants for partial-
success scenarios.

Related Tickets:
    - OMN-2310: Topic Catalog model + suffix foundation
    - OMN-2312: Topic Catalog: response warnings channel
"""

from omnibase_infra.models.catalog.catalog_warning_codes import (
    CONSUL_SCAN_TIMEOUT,
    CONSUL_UNAVAILABLE,
    INVALID_QUERY_PAYLOAD,
    PARTIAL_NODE_DATA,
    VERSION_UNKNOWN,
)
from omnibase_infra.models.catalog.model_topic_catalog_changed import (
    ModelTopicCatalogChanged,
)
from omnibase_infra.models.catalog.model_topic_catalog_entry import (
    ModelTopicCatalogEntry,
)
from omnibase_infra.models.catalog.model_topic_catalog_query import (
    ModelTopicCatalogQuery,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)

__all__: list[str] = [
    "CONSUL_SCAN_TIMEOUT",
    "CONSUL_UNAVAILABLE",
    "INVALID_QUERY_PAYLOAD",
    "PARTIAL_NODE_DATA",
    "VERSION_UNKNOWN",
    "ModelTopicCatalogChanged",
    "ModelTopicCatalogEntry",
    "ModelTopicCatalogQuery",
    "ModelTopicCatalogResponse",
]
