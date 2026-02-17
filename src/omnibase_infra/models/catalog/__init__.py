# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic catalog models for the ONEX platform.

Provides Pydantic models for querying, responding to, and notifying about
topic catalog changes. Used by the topic catalog coordination protocol.

Related Tickets:
    - OMN-2310: Topic Catalog model + suffix foundation
"""

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
    "ModelTopicCatalogChanged",
    "ModelTopicCatalogEntry",
    "ModelTopicCatalogQuery",
    "ModelTopicCatalogResponse",
]
