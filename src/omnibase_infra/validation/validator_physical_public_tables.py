# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Logical schemas whose declared tables physically live in ``public``.

Deployable SQL must target the physical relation, while ownership manifests
declare the logical schema. This maps each logical schema to the tables of it
that still resolve in ``public``.

OMN-26151: ``plan_tiers`` and ``plan_entitlements`` are declared at the logical
``platform_catalog`` schema, but the dev ``omninode_cloud`` database has no
``platform_catalog`` schema and holds them in ``public``. The gateway_runtime
grants seam therefore names ``public.<name>``.
"""

from __future__ import annotations

from omnibase_infra.topology.physical_schema_mapping import (
    INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
)

PLATFORM_CATALOG_TABLES_PHYSICALLY_IN_PUBLIC: frozenset[str] = frozenset(
    {"plan_entitlements", "plan_tiers"}
)

# OMN-26121: tenant_event_outbox is declared at the logical ``omninode_internal``
# schema but the tenant identity seam creates it in ``public`` on omninode_cloud.
INTERNAL_SEAM_TABLES_PHYSICALLY_IN_PUBLIC: frozenset[str] = frozenset(
    {"tenant_event_outbox"}
)

PHYSICALLY_PUBLIC_TABLES: dict[str, frozenset[str]] = {
    "omninode_internal": INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359.union(
        INTERNAL_SEAM_TABLES_PHYSICALLY_IN_PUBLIC
    ),
    "platform_catalog": PLATFORM_CATALOG_TABLES_PHYSICALLY_IN_PUBLIC,
}
